import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.llm_factory import BACKUP_FIRST_STRATEGY, PRIMARY_STRATEGY, normalize_llm_strategy
from utils.logger import logger


_ACQUIRE_PERMIT_SCRIPT = """
local active_key = KEYS[1]
local lease_prefix = KEYS[2]
local queue_key = KEYS[3]

local permit_id = ARGV[1]
local limit = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local request_id = ARGV[4]

local members = redis.call('SMEMBERS', active_key)
for _, member in ipairs(members) do
    if redis.call('EXISTS', lease_prefix .. member) == 0 then
        redis.call('SREM', active_key, member)
    end
end

local active_count = redis.call('SCARD', active_key)
if active_count >= limit then
    return 0
end

redis.call('SET', lease_prefix .. permit_id, request_id, 'EX', ttl)
redis.call('SADD', active_key, permit_id)
redis.call('ZREM', queue_key, request_id)
return active_count + 1
"""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class GraphConcurrencyTimeout(RuntimeError):
    pass


@dataclass
class GraphConcurrencyPermit:
    gate: "RedisGraphConcurrencyGate"
    request_id: str
    permit_id: str
    llm_strategy: str
    wait_seconds: float
    queue_depth: int
    active_count: int
    _heartbeat_task: Optional[asyncio.Task] = field(default=None, repr=False)

    async def release(self) -> None:
        await self.gate.release(self)


class RedisGraphConcurrencyGate:
    def __init__(
        self,
        redis_client,
        *,
        name: str = "graph_runtime",
        limit: Optional[int] = None,
        acquire_timeout: Optional[float] = None,
        lease_ttl: Optional[int] = None,
        queue_poll_interval: Optional[float] = None,
        backup_queue_threshold: Optional[int] = None,
        backup_wait_threshold: Optional[float] = None,
    ) -> None:
        self.redis = redis_client
        self.name = name
        self.limit = limit or _env_int("GRAPH_CONCURRENCY_LIMIT", 50)
        self.acquire_timeout = acquire_timeout or _env_float("GRAPH_CONCURRENCY_ACQUIRE_TIMEOUT", 90.0)
        self.lease_ttl = lease_ttl or _env_int("GRAPH_CONCURRENCY_LEASE_TTL", 120)
        self.queue_poll_interval = queue_poll_interval or _env_float("GRAPH_QUEUE_POLL_INTERVAL", 0.25)
        self.backup_queue_threshold = backup_queue_threshold or _env_int(
            "GRAPH_BACKUP_QUEUE_THRESHOLD",
            max(1, self.limit // 2),
        )
        self.backup_wait_threshold = backup_wait_threshold or _env_float(
            "GRAPH_BACKUP_WAIT_THRESHOLD",
            2.5,
        )
        self._queue_key = f"{self.name}:queue"
        self._sequence_key = f"{self.name}:queue_seq"
        self._active_key = f"{self.name}:active"
        self._lease_prefix = f"{self.name}:lease:"

    def _lease_key(self, permit_id: str) -> str:
        return f"{self._lease_prefix}{permit_id}"

    async def acquire(self, *, session_key: str, channel: str) -> GraphConcurrencyPermit:
        request_id = f"{channel}:{session_key}:{uuid.uuid4().hex}"
        position = await self.redis.incr(self._sequence_key)
        await self.redis.zadd(self._queue_key, {request_id: position})
        start = time.monotonic()

        try:
            while True:
                wait_seconds = time.monotonic() - start
                if wait_seconds >= self.acquire_timeout:
                    raise GraphConcurrencyTimeout(
                        f"Graph concurrency queue timed out after {wait_seconds:.1f}s"
                    )

                rank = await self.redis.zrank(self._queue_key, request_id)
                if rank is None:
                    await self.redis.zadd(self._queue_key, {request_id: position})
                    rank = await self.redis.zrank(self._queue_key, request_id)

                if rank == 0:
                    permit_id = uuid.uuid4().hex
                    active_count = await self.redis.eval(
                        _ACQUIRE_PERMIT_SCRIPT,
                        3,
                        self._active_key,
                        self._lease_prefix,
                        self._queue_key,
                        permit_id,
                        self.limit,
                        self.lease_ttl,
                        request_id,
                    )
                    if int(active_count or 0) > 0:
                        queue_depth = int(await self.redis.zcard(self._queue_key))
                        llm_strategy = self._select_llm_strategy(
                            wait_seconds=wait_seconds,
                            queue_depth=queue_depth,
                        )
                        permit = GraphConcurrencyPermit(
                            gate=self,
                            request_id=request_id,
                            permit_id=permit_id,
                            llm_strategy=llm_strategy,
                            wait_seconds=wait_seconds,
                            queue_depth=queue_depth,
                            active_count=int(active_count),
                        )
                        permit._heartbeat_task = asyncio.create_task(self._heartbeat(permit))
                        return permit

                await asyncio.sleep(self.queue_poll_interval)
        except Exception:
            await self.redis.zrem(self._queue_key, request_id)
            raise

    async def release(self, permit: GraphConcurrencyPermit) -> None:
        if permit._heartbeat_task is not None:
            permit._heartbeat_task.cancel()
            try:
                await permit._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self.redis.delete(self._lease_key(permit.permit_id))
        await self.redis.srem(self._active_key, permit.permit_id)

    def _select_llm_strategy(self, *, wait_seconds: float, queue_depth: int) -> str:
        if wait_seconds >= self.backup_wait_threshold or queue_depth >= self.backup_queue_threshold:
            return BACKUP_FIRST_STRATEGY
        return PRIMARY_STRATEGY

    async def _heartbeat(self, permit: GraphConcurrencyPermit) -> None:
        lease_key = self._lease_key(permit.permit_id)
        interval = max(1.0, self.lease_ttl / 3)
        while True:
            try:
                await asyncio.sleep(interval)
                if await self.redis.get(lease_key) != permit.request_id:
                    return
                await self.redis.expire(lease_key, self.lease_ttl)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(f"⚠️ Graph permit heartbeat failed: {exc}")


class GraphRuntimeController:
    def __init__(self, redis_client, *, gate: Optional[RedisGraphConcurrencyGate] = None) -> None:
        self.gate = gate or RedisGraphConcurrencyGate(redis_client)
        self._pressure_log_interval = _env_float("GRAPH_PRESSURE_LOG_INTERVAL", 5.0)
        self._last_pressure_log_at = 0.0

    async def invoke(
        self,
        graph_app,
        *,
        inputs: dict[str, Any],
        config: Optional[dict[str, Any]],
        session_key: str,
        channel: str,
    ) -> Any:
        permit = await self.gate.acquire(session_key=session_key, channel=channel)
        effective_inputs = self._apply_runtime_strategy(inputs, permit.llm_strategy)
        now = time.monotonic()
        if (
            permit.llm_strategy == BACKUP_FIRST_STRATEGY
            and (now - self._last_pressure_log_at >= self._pressure_log_interval)
        ):
            self._last_pressure_log_at = now
            logger.warning(
                "⚠️ Queue pressure detected, using backup-first chain "
                f"({channel}:{session_key}, waited={permit.wait_seconds:.2f}s, queue={permit.queue_depth}, active={permit.active_count})"
            )

        try:
            return await asyncio.to_thread(graph_app.invoke, effective_inputs, config=config)
        finally:
            await permit.release()

    @staticmethod
    def _apply_runtime_strategy(inputs: dict[str, Any], strategy: str) -> dict[str, Any]:
        normalized_strategy = normalize_llm_strategy(strategy)
        payload = dict(inputs)
        runtime_config = dict(payload.get("runtime_config") or {})
        runtime_config["llm_strategy"] = normalized_strategy
        payload["runtime_config"] = runtime_config
        return payload
