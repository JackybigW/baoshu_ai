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

## Extractor Eval 约定

- 黄金集位于 `nodes_eval/extractor_eval/golden_dataset.json`
- 评分逻辑位于 `nodes_eval/extractor_eval/benchmark.py`
- failure analysis 目录只保留少量关键节点，其余中间 run 做本地归档
- 重要分数变化应写入单独 summary 文档，避免只存在日志里

## 部署约定

推荐顺序：

1. 本地修改
2. 本地测试
3. 推送 GitHub
4. 从本地工作区直接同步服务器
5. 服务器验证版本、测试与进程状态

不推荐把服务器 `git pull GitHub` 作为唯一发布方式，原因是网络链路不稳定时容易出现代码版本和本地验证结果不一致。

## 发布后检查

至少确认以下三项：

- 本地 `git rev-parse --short HEAD`
- `git rev-parse --short origin/main`
- 服务器仓库 `git rev-parse --short HEAD`

如果服务器仓库是脏工作树：

- 先备份
- 再 reset 或定向同步
- 不直接无备份清理
