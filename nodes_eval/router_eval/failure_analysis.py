from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SEVERITY_ORDER = {
    "漏转人工": 0,
    "成交信号被资料补全拦截": 1,
    "低预算分流错误": 2,
    "高价值分流错误": 3,
    "艺术赛道分流错误": 4,
    "资料补全漏拦截": 5,
    "闲聊误入业务流": 6,
    "路由优先级错误": 7,
}


def generate_failure_analysis(payload: Dict[str, Any], output_root: Path, run_timestamp: Optional[datetime] = None) -> Path:
    timestamp = (run_timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / "failure_analyses" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    abnormal_cases: List[Dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    for item in payload["results"]:
        failure_tag = item["score"].get("failure_tag")
        if item["score"]["overall_score"] < 100 or failure_tag:
            abnormal_cases.append(item)
            if failure_tag:
                category_counter.update([failure_tag])

    abnormal_cases.sort(
        key=lambda item: (
            SEVERITY_ORDER.get(item["score"].get("failure_tag") or "", 99),
            item["score"]["overall_score"],
            item["case_id"],
        )
    )

    summary_lines = [
        f"# Router Failure Analysis {timestamp}",
        "",
        f"- abnormal_case_count: `{len(abnormal_cases)}` / `{payload['summary']['case_count']}`",
        f"- overall_score: `{payload['summary']['overall_score']}`",
        f"- route_accuracy: `{payload['summary']['route_accuracy']}`",
        f"- pass_rate: `{payload['summary']['pass_rate']}`",
        "",
        "## Category Counts",
        "",
    ]
    for name, count in sorted(category_counter.items(), key=lambda item: (SEVERITY_ORDER.get(item[0], 99), -item[1], item[0])):
        summary_lines.append(f"- `{name}`: `{count}`")
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    abnormal_lines = [
        f"# Abnormal Cases {timestamp}",
        "",
        f"共 `{len(abnormal_cases)}` 条，已按严重程度排序。",
        "",
    ]
    for item in abnormal_cases:
        abnormal_lines.extend(
            [
                f"## {item['case_id']} | score={item['score']['overall_score']}",
                "",
                f"- tags: `{', '.join(item['tags'])}`",
                f"- failure_tag: `{item['score'].get('failure_tag') or 'None'}`",
                "",
                "### Input",
                "",
                "```json",
                json.dumps(item["input"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Expected",
                "",
                "```json",
                json.dumps(item["expected"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Actual",
                "",
                "```json",
                json.dumps(item["actual"], ensure_ascii=False, indent=2),
                "```",
                "",
                "### Score",
                "",
                "```json",
                json.dumps(item["score"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    (output_dir / "abnormal_cases.md").write_text("\n".join(abnormal_lines), encoding="utf-8")
    (output_dir / "abnormal_cases.json").write_text(json.dumps(abnormal_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir
