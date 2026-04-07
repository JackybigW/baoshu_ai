# Execution Eval Pipeline

这个目录统一评测执行层回复节点：

- `first_greeting`
- `interviewer`
- `consultant`
- `high_value`
- `low_budget`
- `art`
- `human_handoff`
- `chit_chat`

## 数据集格式

黄金集位于 [golden_dataset.json](/Users/jackywang/Documents/baoshu_ai/nodes_eval/execution_eval/golden_dataset.json)。

每条样本包含：

- `node_name`: 要执行的节点
- `input.messages`: 原始对话片段
- `input.profile / last_intent / dialog_status`
- `expected`: 回复合同

回复合同把硬约束和软质量分开：

- 硬约束：是否必须拉群、消息段数、单段长度、消息类型、状态流转
- 软质量：关键点覆盖、节点目标、rubric 备注

## 评分逻辑

优先使用确定性规则：

- 该不该拉群
- 格式是否越界
- 关键要点是否覆盖
- 状态是否正确
- 消息类型是否符合协议

其余自然语言质量由 rubric judge 打分；若 judge 不可用，则退化为关键词覆盖 fallback。

## 运行

请在 `agent` conda 环境里执行：

```bash
python nodes_eval/execution_eval/run_eval.py
```

常用参数：

```bash
python nodes_eval/execution_eval/run_eval.py --limit 6
python nodes_eval/execution_eval/run_eval.py --case-id exe_005
python nodes_eval/execution_eval/run_eval.py --tag handoff
python nodes_eval/execution_eval/run_eval.py --llm deepseek --llm qwen
python nodes_eval/execution_eval/run_eval.py --no-judge
python nodes_eval/execution_eval/run_eval.py --output-json /tmp/execution_eval.json
```

如果不传 `--llm`，默认跑 `frontend_default`。

## 数据集维护

执行层黄金集现在由 6 个 shard 文件拼出来：

- `nodes_eval/execution_eval/datasets/consultant_cases.json`
- `nodes_eval/execution_eval/datasets/interviewer_cases.json`
- `nodes_eval/execution_eval/datasets/high_value_cases.json`
- `nodes_eval/execution_eval/datasets/low_budget_cases.json`
- `nodes_eval/execution_eval/datasets/art_cases.json`
- `nodes_eval/execution_eval/datasets/chit_chat_cases.json`

重建总集：

```bash
python nodes_eval/execution_eval/build_dataset.py
```

如果你只是想在 shard 还没写满时预览合并结果：

```bash
python nodes_eval/execution_eval/build_dataset.py --allow-partial
```

常用 smoke：

```bash
python nodes_eval/execution_eval/run_eval.py --tag consultant --limit 5 --llm deepseek --no-judge
python nodes_eval/execution_eval/run_eval.py --tag chit_chat --limit 5 --llm qwen
python nodes_eval/execution_eval/run_eval.py --tag interviewer --limit 5 --llm gemini_flash
```

说明：

- shard 用于多人并行编写，`golden_dataset.json` 是构建产物
- builder 默认就是 strict，会检查每个目标节点至少 20 条
- `--strict` 仍可用，但只是兼容旧命令
- `consultant` 可使用 `required_context_terms` 强制知识库路径命中
- `chit_chat` 可使用 `forbidden_regexes` 拦截括号式情绪/动作文本
