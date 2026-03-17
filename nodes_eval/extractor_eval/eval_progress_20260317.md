# Extractor Eval Progress - 2026-03-17

## 结论

今天这一轮 extractor 优化，按全量 `100` case 的结果看，最终成绩从早期基线 `82.6` 提升到 `87.98`。

最稳妥的对比口径：

- 基线：`82.6`
- 最终：`87.98`
- `overall_score` 提升：`+5.38`
- `exact_recall`：`0.7919 -> 0.8643`，提升 `+0.0724`
- `hallucination_rate`：`0.101 -> 0.0844`，下降 `-0.0166`
- `pass_rate`：`0.75 -> 0.86`，提升 `+0.11`

如果以人工补完黄金集后的稳定基线来算：

- 中期基线：`83.5`
- 最终：`87.98`
- `overall_score` 提升：`+4.48`
- `exact_recall`：`0.7991 -> 0.8643`，提升 `+0.0652`
- `hallucination_rate`：`0.0906 -> 0.0844`，下降 `-0.0062`
- `pass_rate`：`0.76 -> 0.86`，提升 `+0.10`

## 关键 Run

### 1. 早期全量基线

- 时间：`2026-03-17 17:24:00`
- 日志文件：[eval_extractor.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extractor.log)
- 结果：
  - `overall_score=82.6`
  - `exact_recall=0.7919`
  - `hallucination_rate=0.101`
  - `fuzzy_semantic=0.8115`
  - `pass_rate=0.75`

### 2. 人工修正黄金集后的基线

- 时间：`2026-03-17 17:40:43`
- 日志文件：[eval_extractor.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extractor.log)
- 结果：
  - `overall_score=83.5`
  - `exact_recall=0.7991`
  - `hallucination_rate=0.0906`
  - `fuzzy_semantic=0.798`
  - `pass_rate=0.76`

### 3. Prompt 优化后的高点

- 时间：`2026-03-17 21:00:16`
- 日志文件：[eval_extracot.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extracot.log)
- 结果：
  - `overall_score=88.41`
  - `exact_recall=0.8647`
  - `hallucination_rate=0.0728`
  - `fuzzy_semantic=0.8331`
  - `pass_rate=0.86`
  - `error_count=0`

### 4. `budget=None` 迁移中的中间异常

- 时间：`2026-03-17 20:58:11`
- 日志文件：[eval_extracot.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extracot.log)
- 结果：
  - `overall_score=61.38`
  - `error_count=28`
- 原因：
  - `BudgetInfo.amount` 改成 `None` 后，Pydantic 校验还没完全打通，导致大量 `ValidationError`
- 这条 run **不应作为业务效果对比基线**

### 5. `budget=None` 收敛后的最终结果

- 时间：`2026-03-17 21:40:48`
- 日志文件：[eval_extracot.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extracot.log)
- 结果：
  - `overall_score=87.98`
  - `exact_recall=0.8643`
  - `hallucination_rate=0.0844`
  - `fuzzy_semantic=0.8289`
  - `pass_rate=0.86`
  - `error_count=0`

## 体现在哪里

主要体现在这几个地方：

- 日志对比：
  - [eval_extractor.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extractor.log)
  - [eval_extracot.log](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/eval_extracot.log)
- 最终 failure analysis：
  - [20260317_214048](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/failure_analyses/20260317_214048)
- 当前黄金集和评分逻辑：
  - [golden_dataset.json](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/golden_dataset.json)
  - [benchmark.py](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/benchmark.py)
  - [failure_analysis.py](/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/failure_analysis.py)

## 解释

今天的提升主要不是来自某一个单点，而是三类改动叠加：

- 黄金数据集更贴近业务，不再大量误伤正常输出
- 评分逻辑更公平，DeepSeek 语义评分看到了更完整上下文
- extractor prompt 和后处理对预算换算、未知预算、幻觉控制更稳

剩余最顽固的问题仍然集中在：

- `all_missing`
- `target_conflict`
- `jpy_budget`
- `overseas_inference`
