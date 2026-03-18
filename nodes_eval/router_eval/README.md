# Router Eval Pipeline

这个目录评测 `router.core_router` 的确定性分流逻辑，关注业务优先级是否稳定。

## 数据集格式

黄金集位于 [golden_dataset.json](/Users/jackywang/Documents/baoshu_ai/nodes_eval/router_eval/golden_dataset.json)。

每条样本包含：

- `input.last_intent`
- `input.dialog_status`
- `input.profile`
- `expected.route`

## 评分逻辑

总分 100 分，按最终路由正确性计分。

严重错误优先级：

- 漏转人工
- 成交信号被资料补全拦截
- 高价值 / 艺术 / 低预算赛道走错
- 资料补全漏拦截

## 运行

请在 `agent` conda 环境里执行：

```bash
python nodes_eval/router_eval/run_eval.py
```

常用参数：

```bash
python nodes_eval/router_eval/run_eval.py --limit 8
python nodes_eval/router_eval/run_eval.py --case-id rtr_010
python nodes_eval/router_eval/run_eval.py --tag priority
python nodes_eval/router_eval/run_eval.py --output-json /tmp/router_eval.json
```
