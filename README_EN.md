# Uncle Bao AI

**English** | [中文](./README.md)

Uncle Bao AI is a study-abroad consultation agent system deployed on Enterprise WeChat and the web. It handles message debouncing, profile extraction, intent classification, deterministic routing, human handoff, and extractor evaluation.

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
├── scripts/                     # Environment bootstrap scripts
├── static/                      # Web assets
└── data/                        # Shared data files
```

## Environment Setup

The default development environment is the conda environment `agent`.

```bash
conda activate agent
pip install -r requirements.txt
```

Recommended environment variables in the project root `.env`:

```bash
DEEPSEEK_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
DATABASE_URL=...
WECOM_CORPID=...
WECOM_SECRET=...
WECOM_TOKEN=...
WECOM_AES_KEY=...
WECOM_KF_ID=...
```

## Local Run

```bash
conda activate agent
python main.py
```

Default listen address:

- `http://0.0.0.0:8000`

Main endpoints:

- Web: `/`
- Web API: `/chat`
- Enterprise WeChat callback: `/api/wecom/callback`

## Tests

Run the full suite:

```bash
conda activate agent
PYTHONPATH=. pytest tests -q
```

Common focused tests:

```bash
conda activate agent
PYTHONPATH=. pytest tests/test_nodes_unit.py tests/test_profile_state.py -q
```

## Extractor Eval

Artifacts and scripts live in [nodes_eval/extractor_eval](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval).

Generate the dataset:

```bash
conda activate agent
PYTHONPATH=. python nodes_eval/extractor_eval/generate_dataset.py
```

Run eval:

```bash
conda activate agent
PYTHONPATH=. python nodes_eval/extractor_eval/run_eval.py --concurrency 8
```

Key files:

- [golden_dataset.json](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/golden_dataset.json)
- [benchmark.py](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/benchmark.py)
- [failure_analysis.py](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/failure_analysis.py)
- [eval_progress_20260317.md](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_progress_20260317.md)

## Deployment

Use [deploy.sh](/Users/jackywang/Documents/baoshu_ai/deploy.sh) as the default release entrypoint. The detailed release workflow now lives in the `baoshu-git-deploy` skill instead of repository docs.

## File Hygiene Rules

- Local workspace files such as `.obsidian/` and `README.pdf` are not tracked
- Shared runtime data under `data/` should be tracked when required by the app or tests
- Tooling artifacts such as `.ipynb_checkpoints/`, `.mypy_cache/`, and `__pycache__/` are not tracked
- Intermediate failure-analysis outputs are treated as local artifacts unless explicitly promoted

## Related Scripts

- [deploy.sh](/Users/jackywang/Documents/baoshu_ai/deploy.sh): release entrypoint
- [scripts/setup_postgres_server.sh](/Users/jackywang/Documents/baoshu_ai/scripts/setup_postgres_server.sh): PostgreSQL bootstrap

## Notes

- Do not modify business `system_prompt` unless explicitly requested
- Prefer tests or regression coverage before changing business logic
- Open a dedicated branch for multi-file structural work
