# 暴叔AI

千万级留学网红“暴叔”的咨询智能体，部署在企业微信和 Web，负责初聊、画像提取、分流和转人工。

## 架构

- 感知层：`classifier` 和 `extractor` 并行。
- 决策层：`router.py` 做纯逻辑路由。
- 执行层：`high_value`、`art`、`low_budget`、`interviewer` 等顾问节点。

## 关键文件

- `main.py`：FastAPI 入口，含消息防抖。
- `agent_graph.py`：LangGraph 工作流。
- `state.py`：状态与画像模型。
- `router.py`：纯 Python 路由逻辑。
- `nodes/`：业务节点实现。
- `deploy.sh`：唯一发布入口。

## 工作约束

1. 默认使用项目内 gstack 技能，不再依赖通用 `git-workflow` 或 `code-review-expert`。
2. 代码审查优先用 `/gstack-review`，提测和开 PR 优先用 `/gstack-ship`。
3. 合并与发布优先用 `/gstack-land-and-deploy`，但实际发布执行必须走仓库根目录的 `deploy.sh`。
4. 如果发布配置变化，优先更新 `CLAUDE.md` 里的 `## Deploy Configuration`，必要时再用 `/gstack-setup-deploy` 重配。
5. 运行 Python 和 `pip install` 必须使用 conda 环境 `agent`。
6. 除非用户明确要求，不改业务 `system_prompt` 或其他顾问提示词。
7. 非必要不要改 `.agents/skills/gstack` 源码；它是 vendored 的上游技能包，升级优先整体替换或重新同步。
