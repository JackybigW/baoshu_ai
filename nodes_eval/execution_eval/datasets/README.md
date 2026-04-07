# Execution Eval Shards

这里是 execution eval 的分片数据目录。当前只允许这 6 个节点持有 shard：

- `consultant`
- `interviewer`
- `high_value`
- `low_budget`
- `art`
- `chit_chat`

每个 shard 文件名必须与节点名一致，例如 `consultant_cases.json`、`art_cases.json`。
`nodes_eval/execution_eval/build_dataset.py` 会按这个注册表合并 shard，并生成
`nodes_eval/execution_eval/golden_dataset.json`。
默认 builder 以 scaffold 模式运行，允许这些文件先为空；正式 authoring 完成后，
请使用 `--strict` 或 `strict=True` 强制检查每个 shard 至少 20 条 case。

## 所有权

每个 shard 只由对应节点的作者维护，禁止跨节点改写。
`case_id` 必须在所有 shard 中全局唯一，建议按节点使用稳定前缀，便于审阅和后续追踪：

- `consultant` -> `con_`
- `interviewer` -> `int_`
- `high_value` -> `hiv_`
- `low_budget` -> `low_`
- `art` -> `art_`
- `chit_chat` -> `chat_`

## 编写规则

- scaffold 阶段允许空数组；正式提交前每个 shard 至少 20 条 case。
- 每条 case 的 `node_name` 必须和 shard 名完全一致。
- `case_id` 只负责唯一性，不要复用旧 case。
- 排序交给 builder，作者可以按主题或难度组织原始顺序。
- 如果 case 使用 `required_context_terms`，要写出能稳定覆盖这些上下文点的输入和输出目标。
- 如果 case 使用 `forbidden_regexes`，要明确列出模型最容易漏掉的格式噪音，尤其是 chit_chat 里常见的括号式情绪文本。
- `consultant` shard 可以依赖 `search_products()` 背景，但 case 仍然要自洽，不能把关键信息留给隐式上下文。

## 提交流程

1. 在对应 shard 文件里新增或修改 case。
2. 运行 `python nodes_eval/execution_eval/build_dataset.py` 合并检查。
3. 完成后加上 `--strict` 复查，确认最终输出满足每个节点最少 20 条、`case_id` 无重复、`node_name` 无漂移。
