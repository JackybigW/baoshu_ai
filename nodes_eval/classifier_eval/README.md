# Classifier Eval Pipeline

这个目录评测 `nodes.perception.classifier_node`，目标是验证真实分类结果是否符合业务分层，而不是只看 prompt 表面命中率。

## 数据集格式

黄金集文件是 [golden_dataset.json](/Users/jackywang/Documents/baoshu_ai/nodes_eval/classifier_eval/golden_dataset.json)。

每条样本包含：

- `input.messages`: 最近对话片段，按真实消息顺序给出
- `input.profile`: 进入 classifier 前已有画像
- `input.last_intent`: 上一轮 sticky 意图
- `expected.intent`: 期望分类
- `expected.dialog_status`: 若该分类应该触发状态切换，则显式标注

## 评分逻辑

总分 100 分：

- 意图得分 85 分：精确命中最高，近邻误判按业务成本降权
- 状态得分 15 分：只在该 case 有明确状态约束时计分

高成本错误优先惩罚：

- 漏转人工
- 销售机会漏判
- 决策支持漏判
- sticky 赛道掉回普通咨询

## 运行

请在 `agent` conda 环境里执行：

```bash
python nodes_eval/classifier_eval/run_eval.py
```

常用参数：

```bash
python nodes_eval/classifier_eval/run_eval.py --limit 8
python nodes_eval/classifier_eval/run_eval.py --case-id clf_011
python nodes_eval/classifier_eval/run_eval.py --tag sticky
python nodes_eval/classifier_eval/run_eval.py --llm deepseek --llm qwen
python nodes_eval/classifier_eval/run_eval.py --output-json /tmp/classifier_eval.json
```

如果不传 `--llm`，默认跑 `backend_default`。

传多个 `--llm` 时，会在同一份黄金集上逐个模型运行；每个模型内部按 case 并发，并输出统一 leaderboard。

## 输出产物

- 主日志：`nodes_eval/classifier_eval/eval_classifier.log`
- 分模型日志目录：`nodes_eval/classifier_eval/logs/`
- 失败分析目录：`nodes_eval/classifier_eval/failure_analyses/<timestamp>/`
