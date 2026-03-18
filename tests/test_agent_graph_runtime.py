import agent_graph


def teardown_function():
    agent_graph.close_graph()


def test_initialize_graph_requires_database_url():
    try:
        agent_graph.initialize_graph(database_url="")
    except RuntimeError as exc:
        assert "DATABASE_URL 未配置" in str(exc)
    else:
        raise AssertionError("initialize_graph should fail when DATABASE_URL is missing")


def test_initialize_graph_uses_postgres_checkpointer(monkeypatch):
    class FakePool:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_pool = FakePool()
    fake_checkpointer = object()
    compiled = object()

    monkeypatch.setattr(
        agent_graph,
        "_build_postgres_checkpointer",
        lambda _: (fake_checkpointer, fake_pool),
    )
    monkeypatch.setattr(agent_graph.workflow, "compile", lambda checkpointer: compiled)

    agent_graph.initialize_graph(database_url="postgresql://example")

    assert agent_graph.get_graph_backend() == "postgres"
    assert agent_graph.app._graph is compiled

    agent_graph.close_graph()
    assert fake_pool.closed is True
