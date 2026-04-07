from nodes_eval.common import build_backend_model_configs, build_frontend_model_configs


def test_backend_default_uses_resolved_model_chain_names():
    config = build_backend_model_configs([])[0]

    assert config.label == "backend_default"
    assert config.resolved_model == "deepseek-v3-2-251201 -> gemini-3.1-flash-lite-preview -> doubao-seed-2-0-pro-260215"


def test_frontend_default_uses_resolved_model_chain_names():
    config = build_frontend_model_configs([])[0]

    assert config.label == "frontend_default"
    assert config.resolved_model == "deepseek-v3-2-251201 -> gemini-3.1-pro-preview -> doubao-seed-2-0-pro-260215"
