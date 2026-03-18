# Uncle Bao AI

**English** | [中文](./README.md)

Uncle Bao AI is a study-abroad consultation agent system deployed on Enterprise WeChat and the web. It handles message debouncing, profile extraction, intent classification, deterministic routing, human handoff, and node-level eval pipelines.

## Core Capabilities

- Dual entry points: Enterprise WeChat and web
- Redis-based message buffering and debouncing
- Parallel `classifier` and `extractor` perception
- Deterministic Python routing
- Specialist consultant nodes: general, high-value, art, low-budget, and profile completion
- Extractor eval pipeline with failure analysis

## Tech Stack

- Python 3.10+
- FastAPI
- LangGraph / LangChain
- PostgreSQL
- Redis
- Pydantic v2
- DeepSeek / OpenAI / Gemini

## Repository Layout

```text
.
├── main.py                      # FastAPI entrypoint
├── agent_graph.py               # LangGraph workflow
├── state.py                     # Shared state and profile merge logic
├── router.py                    # Deterministic router
├── nodes/                       # Business nodes
├── utils/                       # Buffer, logging, WeCom API, LLM factory
├── db/                          # Postgres store and schema
├── config/                      # Prompts and settings
├── tests/                       # Unit and integration tests
├── nodes_eval/extractor_eval/   # Extractor eval data and scripts
├── nodes_eval/classifier_eval/  # Classifier eval data and scripts
├── nodes_eval/router_eval/      # Router eval data and scripts
├── nodes_eval/execution_eval/   # Execution-node eval data and scripts
├── scripts/                     # Environment bootstrap scripts
├── static/                      # Web assets
└── data/                        # Shared data files
```

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Tests

```bash
PYTHONPATH=. pytest tests -q
```

## Eval Pipelines

```bash
PYTHONPATH=. python nodes_eval/extractor_eval/generate_dataset.py
PYTHONPATH=. python nodes_eval/extractor_eval/run_eval.py --concurrency 8
PYTHONPATH=. python nodes_eval/classifier_eval/run_eval.py --concurrency 8
PYTHONPATH=. python nodes_eval/router_eval/run_eval.py
PYTHONPATH=. python nodes_eval/execution_eval/run_eval.py
```

Key files:

- [golden_dataset.json](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/golden_dataset.json)
- [benchmark.py](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/benchmark.py)
- [failure_analysis.py](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/failure_analysis.py)
- [eval_progress_20260317.md](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_progress_20260317.md)
- [classifier_eval](/Users/jackywang/Documents/baoshu_ai/nodes_eval/classifier_eval/README.md)
- [router_eval](/Users/jackywang/Documents/baoshu_ai/nodes_eval/router_eval/README.md)
- [execution_eval](/Users/jackywang/Documents/baoshu_ai/nodes_eval/execution_eval/README.md)
