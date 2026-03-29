# 暴叔 AI

## Project Context

- This repository is a FastAPI + LangGraph system for Baoshu's enterprise WeCom and Web consultation flows.
- Python runtime and dependency operations must use conda environment `agent`.
- Core files: `main.py`, `agent_graph.py`, `state.py`, `router.py`, `nodes/`, `deploy.sh`.

## Working Conventions

- Use repo-local gstack skills from `.agents/skills/`.
- Prefer `/gstack-review` for pre-merge review.
- Prefer `/gstack-ship` for branch checks, test execution, push, and PR creation.
- Prefer `/gstack-land-and-deploy` only after the branch is approved and ready to merge.
- Do not change business prompts unless the user explicitly asks.

## Commands

- Install deps: `conda run -n agent pip install -r requirements.txt`
- Run app: `conda run -n agent python main.py`
- Run tests: `conda run -n agent python -m pytest tests -q`
- Quick deploy tests are already encoded in `deploy.sh`

## Deploy Configuration (configured for gstack)
- Platform: custom deploy script via `deploy.sh`
- Production URL: `https://ai.afyedu.cn`
- Deploy workflow: custom local deploy script
- Deploy status command: HTTP health check
- Merge method: squash
- Project type: web app / API
- Post-deploy health check: `https://ai.afyedu.cn/api/chat-config`

### Custom deploy hooks
- Pre-merge: `conda run -n agent python -m pytest tests -q`
- Deploy trigger: `bash deploy.sh`
- Deploy status: `curl -sf https://ai.afyedu.cn/api/chat-config >/dev/null`
- Health check: `https://ai.afyedu.cn/api/chat-config`
