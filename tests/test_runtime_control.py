import pytest

from utils.runtime_control import GraphRuntimeController, RedisGraphConcurrencyGate


class FakePermit:
    def __init__(self, *, strategy: str = "primary", wait_seconds: float = 0.0, queue_depth: int = 0):
        self.llm_strategy = strategy
        self.wait_seconds = wait_seconds
        self.queue_depth = queue_depth
        self.active_count = 1
        self.released = False

    async def release(self):
        self.released = True


class FakeGate:
    def __init__(self, permit: FakePermit):
        self.permit = permit
        self.calls = []

    async def acquire(self, *, session_key: str, channel: str):
        self.calls.append({"session_key": session_key, "channel": channel})
        return self.permit


class FakeGraphApp:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, inputs, config=None):
        self.calls.append({"inputs": inputs, "config": config})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_runtime_controller_applies_backup_strategy_from_permit():
    permit = FakePermit(strategy="backup_first", wait_seconds=3.0, queue_depth=18)
    controller = GraphRuntimeController(redis_client=None, gate=FakeGate(permit))
    graph_app = FakeGraphApp([{"ok": True}])

    result = await controller.invoke(
        graph_app,
        inputs={"messages": []},
        config={"configurable": {"thread_id": "session_a"}},
        session_key="session_a",
        channel="web",
    )

    assert result == {"ok": True}
    assert graph_app.calls[0]["inputs"]["runtime_config"]["llm_strategy"] == "backup_first"
    assert permit.released is True


@pytest.mark.asyncio
async def test_runtime_controller_applies_primary_strategy_without_retry():
    permit = FakePermit(strategy="primary", wait_seconds=0.1, queue_depth=0)
    controller = GraphRuntimeController(redis_client=None, gate=FakeGate(permit))
    graph_app = FakeGraphApp([{"ok": True}])

    result = await controller.invoke(
        graph_app,
        inputs={"messages": ["hello"]},
        config={"configurable": {"thread_id": "session_b"}},
        session_key="session_b",
        channel="wecom",
    )

    assert result == {"ok": True}
    assert len(graph_app.calls) == 1
    assert graph_app.calls[0]["inputs"]["runtime_config"]["llm_strategy"] == "primary"
    assert permit.released is True


@pytest.mark.asyncio
async def test_runtime_controller_propagates_errors_without_replaying_graph():
    permit = FakePermit(strategy="primary")
    controller = GraphRuntimeController(redis_client=None, gate=FakeGate(permit))
    graph_app = FakeGraphApp([RuntimeError("429 rate limit from upstream")])

    with pytest.raises(RuntimeError, match="429 rate limit"):
        await controller.invoke(
            graph_app,
            inputs={"messages": ["hello"]},
            config={"configurable": {"thread_id": "session_c"}},
            session_key="session_c",
            channel="wecom",
        )

    assert len(graph_app.calls) == 1
    assert permit.released is True


def test_gate_switches_to_backup_under_queue_pressure():
    gate = RedisGraphConcurrencyGate(
        redis_client=None,
        limit=50,
        acquire_timeout=30,
        lease_ttl=60,
        queue_poll_interval=0.1,
        backup_queue_threshold=10,
        backup_wait_threshold=1.5,
    )

    assert gate._select_llm_strategy(wait_seconds=0.2, queue_depth=3) == "primary"
    assert gate._select_llm_strategy(wait_seconds=2.0, queue_depth=3) == "backup_first"
    assert gate._select_llm_strategy(wait_seconds=0.2, queue_depth=12) == "backup_first"
