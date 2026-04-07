# Extractor Eval Pipeline

这个目录只服务 `nodes.perception.extractor_node` 的自动化评测，目标是直接复用线上 extractor 逻辑，不另写一套“假评测器”。

## 数据集格式

黄金数据集文件是 [golden_dataset.json](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/golden_dataset.json)，每条样本至少包含两个顶级字段：

```json
{
  "case_id": "case_001",
  "tags": ["anxious_parent", "unit_confusion"],
  "input": {
    "last_ai_msg": "可以把孩子当前年级、成绩、预算说一下吗？",
    "last_user_msg": "..."
  },
  "expected": {
    "user_role": "家长",
    "educationStage": "高中",
    "budget": {
      "amount": 36,
      "period": "TOTAL"
    }
  }
}
```

`input.current_profile` 是可选字段，用于覆盖多轮合并、已有画像更新、隐式确认等场景。

## 评分基准

总分 100 分，按以下权重汇总：

- 精确召回率 60 分：按字段级单元计算。`destination_preference` 按国家逐项计分，`budget.amount` 和 `budget.period` 分开计分。
- 幻觉惩罚 30 分：统计模型实际输出里不该出现的信息占比，再换算成剩余分数。预测越多无中生有，扣分越重。
- 模糊语义匹配 10 分：只对 `target_school`、`target_major`、`academic_background`、`language_level` 打分。优先调用 DeepSeek，缺 key 时退化为 token overlap。

最终公式：

```text
overall = exact_recall * 60 + (1 - hallucination_rate) * 30 + fuzzy_semantic * 10
```

## 汇率约定

为了让“5 万美金 / 5000 磅”这类陷阱有统一标准，黄金集采用固定换算口径，统一折算到“人民币万元”：

- 1 USD = 7.2 CNY
- 1 GBP = 9.2 CNY
- 1 EUR = 7.8 CNY
- 1 HKD = 0.92 CNY
- 1 JPY = 0.048 CNY
- 1 KRW = 0.0053 CNY

金额保留到“万”为单位后四舍五入成整数。

## 运行

请在 `agent` conda 环境里执行：

```bash
python nodes_eval/extractor_eval/run_eval.py
```

常用参数：

```bash
python nodes_eval/extractor_eval/run_eval.py --limit 10
python nodes_eval/extractor_eval/run_eval.py --case-id case_017
python nodes_eval/extractor_eval/run_eval.py --concurrency 8
python nodes_eval/extractor_eval/run_eval.py --tag currency_conversion
python nodes_eval/extractor_eval/run_eval.py --llm deepseek --llm gemini_flash --llm qwen
python nodes_eval/extractor_eval/run_eval.py --output-json /tmp/extractor_eval_result.json
```

如果不传 `--llm`，默认跑 `backend_default`，也就是线上 extractor 当前 backend fallback chain。

传多个 `--llm` 时，会在同一份黄金集和 benchmark 上逐个模型运行；每个模型内部按 case 并发，并输出统一 leaderboard。

## 输出产物

每次运行都会把成绩追加写入以下日志：

- 主日志：[eval_extracot.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extracot.log)
- 分模型日志目录：[logs/](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/logs)

失败分析目录会按“运行时间 / LLM”区分，例如：

```text
nodes_eval/extractor_eval/failure_analyses/20260318_153000/
├── README.md
├── leaderboard.json
├── backend_default/
│   ├── summary.md
│   ├── abnormal_cases.md
│   └── abnormal_cases.json
└── qwen/
    ├── summary.md
    ├── abnormal_cases.md
    └── abnormal_cases.json
```

其中：

- 根目录 `README.md` 记录这次 multi-LLM run 的 leaderboard
- 各 LLM 子目录分别保存自己的异常样本、错误分类和得分摘要

说明：

- `run_eval.py` 现在会按模型顺序运行，每个模型内部按 case 并发执行；默认单模型并发 `--concurrency 8`
- 单 case 请求失败不会中断整批，会在 summary 里累计 `error_count`
- `--output-json` 会写出完整 run payload，包含 leaderboard、各 LLM 的 summary、日志路径和 failure analysis 目录
