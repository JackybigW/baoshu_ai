from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SEVERITY_ORDER = {
    "API/运行异常": 0,
    "单位/汇率/预算周期崩塌": 1,
    "预算周期错误": 2,
    "预算漏抓": 3,
    "预算幻觉": 4,
    "关键信息抓漏": 5,
    "纯幻觉/过提取": 6,
    "冲突意图混写": 7,
    "轻微语义偏差": 8,
}


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value).strip("._").lower() or "unknown_llm"


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, dict):
        if "amount" in value and "period" in value:
            return value.get("amount") is not None or value.get("period") != "UNKNOWN"
        return any(has_value(item) for item in value.values())
    return True


def classify_case(item: Dict[str, Any]) -> List[str]:
    expected = item["expected"]
    actual = item["actual"]
    last_user_msg = item["input"]["last_user_msg"]
    reasons: List[str] = []

    if item.get("error"):
        reasons.append("API/运行异常")

    expected_amount = expected["budget"]["amount"]
    actual_amount = actual["budget"]["amount"]
    expected_period = expected["budget"]["period"]
    actual_period = actual["budget"]["period"]
    budget_signal = any(
        token in last_user_msg.lower()
        for token in [
            "美金",
            "美元",
            "镑",
            "£",
            "欧元",
            "港币",
            "日元",
            "韩币",
            "hkd",
            "usd",
            "jpy",
            "krw",
            "一年",
            "/year",
            "year",
            "annual",
            "总共",
            "all-in",
            "total",
        ]
    )
    if expected_amount is not None and actual_amount is not None and expected_amount != actual_amount and budget_signal:
        reasons.append("单位/汇率/预算周期崩塌")
    elif expected_amount is not None and actual_amount is None:
        reasons.append("预算漏抓")
    elif expected_amount is None and actual_amount is not None:
        reasons.append("预算幻觉")

    if expected_period != actual_period and expected_period != "UNKNOWN":
        reasons.append("预算周期错误")

    for field_name in [
        "user_role",
        "educationStage",
        "abroad_readiness",
        "target_school",
        "target_major",
        "academic_background",
        "language_level",
    ]:
        expected_value = expected.get(field_name)
        actual_value = actual.get(field_name)
        if has_value(expected_value) and not has_value(actual_value):
            reasons.append("关键信息抓漏")
        if not has_value(expected_value) and has_value(actual_value):
            reasons.append("纯幻觉/过提取")

    expected_destinations = expected.get("destination_preference") or []
    actual_destinations = actual.get("destination_preference") or []
    if expected_destinations and not actual_destinations:
        reasons.append("关键信息抓漏")
    if not expected_destinations and actual_destinations:
        reasons.append("纯幻觉/过提取")
    if set(actual_destinations) - set(expected_destinations):
        reasons.append("纯幻觉/过提取")

    actual_major = actual.get("target_major")
    expected_major = expected.get("target_major")
    if (
        isinstance(actual_major, str)
        and any(token in actual_major for token in ["（家长意向）", "(家长意向)", "或", "/", "；"])
        and expected_major
        and actual_major != expected_major
    ):
        reasons.append("冲突意图混写")

    if not reasons and item["score"]["overall_score"] < 100:
        reasons.append("轻微语义偏差")

    return list(dict.fromkeys(reasons))


def generate_failure_analysis(
    payload: Dict[str, Any],
    output_root: Path,
    run_timestamp: Optional[datetime] = None,
    model_label: Optional[str] = None,
) -> Path:
    timestamp = (run_timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / "failure_analyses" / timestamp
    if model_label:
        output_dir = output_dir / _slugify(model_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_meta = payload.get("llm") or {}
    llm_header = []
    if llm_meta:
        llm_header.extend(
            [
                f"- llm_label: `{llm_meta.get('label', '')}`",
                f"- llm_canonical: `{llm_meta.get('canonical_id', '')}`",
                f"- llm_provider: `{llm_meta.get('provider', '')}`",
                f"- llm_model: `{llm_meta.get('resolved_model', '')}`",
            ]
        )

    abnormal_cases: List[Dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    for item in payload["results"]:
        reasons = classify_case(item)
        if item["score"]["overall_score"] < 100 or reasons:
            enriched = dict(item)
            enriched["failure_tags"] = reasons
            abnormal_cases.append(enriched)
            category_counter.update(reasons)

    abnormal_cases.sort(
        key=lambda item: (
            min(SEVERITY_ORDER.get(tag, 99) for tag in item["failure_tags"]) if item["failure_tags"] else 99,
            item["score"]["overall_score"],
            item["case_id"],
        )
    )

    summary_lines = [
        f"# Failure Analysis {timestamp}",
        "",
        f"- abnormal_case_count: `{len(abnormal_cases)}` / `{payload['summary']['case_count']}`",
        f"- overall_score: `{payload['summary']['overall_score']}`",
        f"- exact_recall: `{payload['summary']['exact_recall']}`",
        f"- hallucination_rate: `{payload['summary']['hallucination_rate']}`",
        f"- fuzzy_semantic: `{payload['summary']['fuzzy_semantic']}`",
        f"- pass_rate: `{payload['summary']['pass_rate']}`",
        f"- error_count: `{payload['summary']['error_count']}`",
        *llm_header,
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
        f"共 `{len(abnormal_cases)}` 条。已按严重程度优先排序，严重问题在前，轻微语义偏差在后。",
        "",
    ]
    for item in abnormal_cases:
        abnormal_lines.extend(
            [
                f"## {item['case_id']} | score={item['score']['overall_score']}",
                "",
                f"- tags: `{', '.join(item['tags'])}`",
                f"- failure_tags: `{', '.join(item['failure_tags'])}`",
                f"- error: `{item.get('error') or 'None'}`",
                "",
                "### 对话",
                "",
                f"- AI: {item['input'].get('last_ai_msg', '')}",
                f"- User: {item['input'].get('last_user_msg', '')}",
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
