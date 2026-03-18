from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SEVERITY_ORDER = {
    "空回复": 0,
    "该拉群未拉群": 1,
    "误拉群/过早拉群": 2,
    "状态流转错误": 3,
    "输出消息类型错误": 4,
    "关键要点缺失": 5,
    "格式违规": 6,
    "命中禁用表达": 7,
    "回复质量偏低": 8,
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
        failure_tags = item["score"].get("failure_tags") or []
        if item["score"]["overall_score"] < 100 or failure_tags:
            abnormal_cases.append(item)
            category_counter.update(failure_tags)

    abnormal_cases.sort(
        key=lambda item: (
            min((SEVERITY_ORDER.get(tag, 99) for tag in item["score"].get("failure_tags") or []), default=99),
            item["score"]["overall_score"],
            item["case_id"],
        )
    )

    summary_lines = [
        f"# Execution Failure Analysis {timestamp}",
        "",
        f"- abnormal_case_count: `{len(abnormal_cases)}` / `{payload['summary']['case_count']}`",
        f"- overall_score: `{payload['summary']['overall_score']}`",
        f"- tool_success_rate: `{payload['summary']['tool_success_rate']}`",
        f"- format_score: `{payload['summary']['format_score']}`",
        f"- keyword_score: `{payload['summary']['keyword_score']}`",
        f"- rubric_score: `{payload['summary']['rubric_score']}`",
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
        f"共 `{len(abnormal_cases)}` 条，按严重度优先排序。",
        "",
    ]
    for item in abnormal_cases:
        abnormal_lines.extend(
            [
                f"## {item['case_id']} | node={item['node_name']} | score={item['score']['overall_score']}",
                "",
                f"- tags: `{', '.join(item['tags'])}`",
                f"- failure_tags: `{', '.join(item['score'].get('failure_tags') or ['None'])}`",
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
