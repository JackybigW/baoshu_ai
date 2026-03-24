import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import redis.asyncio as aioredis

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.runtime_control import GraphRuntimeController, RedisGraphConcurrencyGate
from utils.logger import logger as app_logger


class FakeGraphApp:
    def __init__(self, *, service_time: float, jitter: float = 0.0) -> None:
        self.service_time = service_time
        self.jitter = jitter
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.strategy_counts: Counter[str] = Counter()

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        started_at = time.perf_counter()
        runtime_config = inputs.get("runtime_config") or {}
        llm_strategy = str(runtime_config.get("llm_strategy") or "primary")
        sleep_time = self.service_time
        if self.jitter > 0:
            sleep_time += random.uniform(0, self.jitter)

        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.strategy_counts[llm_strategy] += 1

        try:
            time.sleep(sleep_time)
        finally:
            with self._lock:
                self._active -= 1

        finished_at = time.perf_counter()
        return {
            "messages": [],
            "_benchmark": {
                "started_at": started_at,
                "finished_at": finished_at,
                "service_time": sleep_time,
                "llm_strategy": llm_strategy,
                "thread_id": ((config or {}).get("configurable") or {}).get("thread_id"),
            },
        }


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * ratio))
    return sorted_values[index]


async def _cleanup_keys(redis_client, prefix: str) -> None:
    cursor = 0
    pattern = f"{prefix}:*"
    keys: list[str] = []
    while True:
        cursor, batch = await redis_client.scan(cursor=cursor, match=pattern, count=200)
        keys.extend(batch)
        if cursor == 0:
            break
    if keys:
        await redis_client.delete(*keys)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=max(args.limit * 2, 64),
        thread_name_prefix="graph-benchmark",
    )
    loop.set_default_executor(executor)

    redis_client = aioredis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        decode_responses=True,
    )
    prefix = f"graph_runtime_benchmark:{uuid.uuid4().hex}"
    gate = RedisGraphConcurrencyGate(
        redis_client,
        name=prefix,
        limit=args.limit,
        acquire_timeout=args.acquire_timeout,
        lease_ttl=args.lease_ttl,
        queue_poll_interval=args.poll_interval,
        backup_queue_threshold=args.backup_queue_threshold,
        backup_wait_threshold=args.backup_wait_threshold,
    )
    controller = GraphRuntimeController(redis_client, gate=gate)
    fake_graph = FakeGraphApp(service_time=args.service_time, jitter=args.jitter)

    await _cleanup_keys(redis_client, prefix)

    semaphore = asyncio.Semaphore(args.client_concurrency)
    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    async def worker(index: int) -> None:
        session_key = f"bench_{index:05d}"
        request_started_at = time.perf_counter()
        config = {"configurable": {"thread_id": session_key}}
        async with semaphore:
            try:
                result = await controller.invoke(
                    fake_graph,
                    inputs={"messages": [f"message-{index}"]},
                    config=config,
                    session_key=session_key,
                    channel="benchmark",
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                return

        benchmark = result["_benchmark"]
        results.append(
            {
                "session_key": session_key,
                "latency": benchmark["finished_at"] - request_started_at,
                "queue_wait": benchmark["started_at"] - request_started_at,
                "service_time": benchmark["service_time"],
                "llm_strategy": benchmark["llm_strategy"],
            }
        )

    try:
        await asyncio.gather(*(worker(i) for i in range(args.requests)))
    finally:
        await _cleanup_keys(redis_client, prefix)
        close_fn = getattr(redis_client, "aclose", None) or redis_client.close
        await close_fn()
        executor.shutdown(wait=False, cancel_futures=True)

    finished_at = time.perf_counter()
    latencies = [item["latency"] for item in results]
    queue_waits = [item["queue_wait"] for item in results]
    strategy_counts = Counter(item["llm_strategy"] for item in results)

    summary = {
        "requests": args.requests,
        "client_concurrency": args.client_concurrency,
        "graph_limit": args.limit,
        "service_time_seconds": args.service_time,
        "jitter_seconds": args.jitter,
        "total_duration_seconds": round(finished_at - started_at, 3),
        "throughput_rps": round((len(results) / max(finished_at - started_at, 0.001)), 2),
        "max_active_graph_invocations": fake_graph.max_active,
        "backup_first_count": strategy_counts.get("backup_first", 0),
        "primary_count": strategy_counts.get("primary", 0),
        "latency_p50_seconds": round(_percentile(latencies, 0.50), 3),
        "latency_p95_seconds": round(_percentile(latencies, 0.95), 3),
        "latency_p99_seconds": round(_percentile(latencies, 0.99), 3),
        "queue_wait_p50_seconds": round(_percentile(queue_waits, 0.50), 3),
        "queue_wait_p95_seconds": round(_percentile(queue_waits, 0.95), 3),
        "queue_wait_p99_seconds": round(_percentile(queue_waits, 0.99), 3),
        "queue_wait_avg_seconds": round(statistics.mean(queue_waits) if queue_waits else 0.0, 3),
        "errors": errors[:10],
        "error_count": len(errors),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the Redis-backed graph runtime gate.")
    parser.add_argument("--requests", type=int, default=500, help="Total requests to simulate.")
    parser.add_argument(
        "--client-concurrency",
        type=int,
        default=500,
        help="How many client coroutines fire at the same time.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Global graph concurrency limit.")
    parser.add_argument("--service-time", type=float, default=0.35, help="Simulated graph execution time.")
    parser.add_argument("--jitter", type=float, default=0.05, help="Extra random execution time.")
    parser.add_argument("--acquire-timeout", type=float, default=90.0, help="Queue timeout in seconds.")
    parser.add_argument("--lease-ttl", type=int, default=120, help="Permit lease TTL in seconds.")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="Queue polling interval.")
    parser.add_argument("--backup-queue-threshold", type=int, default=25, help="Queue depth to trigger backup-first.")
    parser.add_argument("--backup-wait-threshold", type=float, default=2.5, help="Wait time to trigger backup-first.")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument(
        "--log-level",
        default="ERROR",
        help="Console log level for benchmark internals. Use NONE to silence all logs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    app_logger.remove()
    if str(args.log_level).upper() != "NONE":
        app_logger.add(sys.stderr, level=str(args.log_level).upper(), format="{message}")
    summary = asyncio.run(run_benchmark(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary["error_count"] > 0:
        return 1
    if summary["max_active_graph_invocations"] > args.limit:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
