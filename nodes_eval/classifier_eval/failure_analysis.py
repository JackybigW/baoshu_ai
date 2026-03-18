from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SEVERITY_ORDER = {
    "API/运行异常": 0,
    "漏转人工": 1,
    "销售信号漏判": 2,
    "决策支持漏判": 3,
    "赛道漏判": 4,
    "过度升级": 5,
    "状态流转错误": 6,
    "类别误判": 7,
}


def generate_failure_analysis(
    payload: Dict[str, Any],
    output_root: Path,
    run_timestamp: Optional[datetime] = None,
    model_label: Optional[str] = None,
) -> Path:
    timestamp = (run_timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / "failure_analyses" / timestamp
    if model_label:
        output_dir = output_dir / model_label
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
        f"# Classifier Failure Analysis {timestamp}",
        "",
        f"- abnormal_case_count: `{len(abnormal_cases)}` / `{payload['summary']['case_count']}`",
        f"- overall_score: `{payload['summary']['overall_score']}`",
        f"- intent_accuracy: `{payload['summary']['intent_accuracy']}`",
        f"- status_accuracy: `{payload['summary']['status_accuracy']}`",
        f"- pass_rate: `{payload['summary']['pass_rate']}`",
        f"- error_count: `{payload['summary']['error_count']}`",
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
                f"- error: `{item.get('error') or 'None'}`",
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
