# Deployment Runbook

## 目标

发布流程以“本地验证后的工作区内容”为准，不把服务器直接 `git pull GitHub` 作为唯一发布路径。

## 推荐流程

1. 在本地分支完成修改
2. 在 conda `agent` 环境跑测试
3. 提交并推送 GitHub
4. 从本地工作区直接同步到服务器
5. 在服务器上跑快速回归并重启服务
6. 核对本地、GitHub、服务器三端版本

## 使用 `deploy.sh`

```bash
bash deploy.sh
```

脚本会执行：

- 可选本地测试：`skip / quick / full`
- 本地生成 git bundle 并连同工作区一起直传服务器
- 如果服务器工作树已脏，先备份 `git status` 和 `git diff`
- 服务器从 bundle 导入本次提交，并把 `HEAD` 对齐到部署版本
- 可选服务器依赖安装
- 可选服务器快速回归
- 重启 `main.py`
- 打印本地与服务器版本信息

## 本地测试

快速回归：

```bash
conda activate agent
PYTHONPATH=. pytest \
  tests/test_profile_state.py \
  tests/test_nodes_unit.py \
  tests/test_extractor_eval.py \
  tests/test_intent_guardrails.py \
  -q
```

全量测试：

```bash
conda activate agent
PYTHONPATH=. pytest tests -q
```

## 服务器校验

常用检查：

```bash
ssh root@<server> "cd /home/ubuntu/baoshu_ai && git branch --show-current && git rev-parse --short HEAD && git status --short --branch"
ssh root@<server> "pgrep -af '/root/miniconda3/envs/agent/bin/python main.py'"
ssh root@<server> "tail -n 20 /home/ubuntu/baoshu_ai/output.log"
```

## 注意事项

- 旧版脚本只做 `rsync`，会直接覆盖服务器 repo 里的 tracked 文件，因此服务代码虽然更新了，但 `git HEAD` 不变，`git status` 会变脏
- 新版脚本会在本地生成 git bundle，并在服务器先备份脏工作树，再把服务器仓库 `HEAD` 对齐到同一个 commit
- 如果只是为了保证服务代码一致，仍然优先使用本地直传；bundle 对齐的目的是让服务器 Git 状态也一致
- 如果服务器无法直接访问 GitHub，新版 bundle 流程仍可工作，不依赖服务器 `git pull`
