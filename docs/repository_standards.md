# Repository Standards

## 目标

本文件定义当前仓库的最小工程治理约定，目的是降低工作区噪音、提升部署可重复性，并减少“本地能跑、服务器不一致”的问题。

## 分支策略

- `main` 仅保留可部署状态
- 多文件结构调整、部署治理、评测体系改造，必须单开分支
- 合并前至少完成一次相关测试

## 依赖与环境

- 默认 Python 环境为 conda `agent`
- `.env` 存放本地和服务器运行所需密钥，不入库
- 共享运行依赖文件应纳入版本控制，例如 `data/products_intro.xlsx`

## 文件追踪原则

应入库：

- 源代码
- 测试代码
- 共享脚本
- 运行和测试都依赖的数据文件
- 说明文档

不应入库：

- 本地编辑器配置，如 `.obsidian/`
- 本地导出文件，如 `README.pdf`
- 工具缓存，如 `__pycache__/`、`.pytest_cache/`、`.mypy_cache/`
- Jupyter checkpoint，如 `.ipynb_checkpoints/`
- 平台垃圾文件，如 `.DS_Store`、`._*`
- 中间态 eval 产物，除非明确要作为基线保留

## Eval 约定

- 黄金集位于 `nodes_eval/extractor_eval/golden_dataset.json`
- 评分逻辑位于 `nodes_eval/extractor_eval/benchmark.py`
- 其他节点评测统一位于 `nodes_eval/classifier_eval/`、`nodes_eval/router_eval/`、`nodes_eval/execution_eval/`
- failure analysis 目录只保留少量关键节点，其余中间 run 做本地归档
- 重要分数变化应写入单独 summary 文档，避免只存在日志里

## 发布

- 默认入口仍然是 `deploy.sh`
- 默认 review / ship / merge 流程改为项目内 gstack：
  - `/gstack-review`
  - `/gstack-ship`
  - `/gstack-land-and-deploy`
- `CLAUDE.md` 中的 `## Deploy Configuration` 是 gstack deploy 技能的仓库内事实来源
- 如果 gstack 的 deploy 配置与 `deploy.sh` 不一致，以 `deploy.sh` 的真实行为为准，并同步更新 `CLAUDE.md`
