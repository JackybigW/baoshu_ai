from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from state import BudgetPeriod, CustomerProfile

load_dotenv(find_dotenv())

EXACT_SCALAR_FIELDS = ("user_role", "educationStage", "abroad_readiness")
LIST_FIELDS = ("destination_preference",)
SEMANTIC_FIELDS = ("target_school", "target_major", "academic_background", "language_level")


class SemanticMatcher(Protocol):
    def __call__(self, field_name: str, expected: str, actual: str) -> Tuple[float, str]:
        ...


class SemanticJudgement(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="")


@dataclass
class ScoreBreakdown:
    overall_score: float
    exact_recall: float
    hallucination_rate: float
    fuzzy_semantic: float
    exact_points: float
    hallucination_points: float
    fuzzy_points: float
    expected_units: int
    matched_units: int
    actual_units: int
    hallucinated_units: int
    active_weight_total: int
    field_details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "exact_recall": round(self.exact_recall, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "fuzzy_semantic": round(self.fuzzy_semantic, 4),
            "exact_points": round(self.exact_points, 2),
            "hallucination_points": round(self.hallucination_points, 2),
            "fuzzy_points": round(self.fuzzy_points, 2),
            "expected_units": self.expected_units,
            "matched_units": self.matched_units,
            "actual_units": self.actual_units,
            "hallucinated_units": self.hallucinated_units,
            "active_weight_total": self.active_weight_total,
            "field_details": self.field_details,
        }


def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,、/|]+", ";", text)
    text = re.sub(r"[；]+", ";", text)
    return text


def _normalize_segments(value: Optional[str]) -> List[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return []
    parts = [segment for segment in normalized.split(";") if segment]
    return sorted(dict.fromkeys(parts))


def _normalize_list(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    normalized = []
    for item in values:
        token = _normalize_text(item)
        if token and token not in normalized:
            normalized.append(token)
    return normalized


def _text_exact_match(expected: Optional[str], actual: Optional[str]) -> bool:
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if _normalize_text(expected) == _normalize_text(actual):
        return True
    return _normalize_segments(expected) == _normalize_segments(actual)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _profile_from_any(data: Any) -> CustomerProfile:
    if isinstance(data, CustomerProfile):
        return data
    if data is None:
        return CustomerProfile()
    return CustomerProfile.model_validate(data)


class DeepSeekSemanticMatcher:
    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.enabled = bool(api_key)
        self._judge = None
        if not self.enabled:
            return
        llm = init_chat_model(
            "deepseek-chat",
            model_provider="deepseek",
            api_key=api_key,
            temperature=0,
        )
        self._judge = llm.with_structured_output(SemanticJudgement)

    def __call__(self, field_name: str, expected: str, actual: str) -> Tuple[float, str]:
        if not self.enabled or self._judge is None:
            return self._fallback(field_name, expected, actual)

        prompt = (
            "你是 extractor eval 审稿器。请比较 expected 和 actual 在字段层面的语义一致度。\n"
            "评分范围 0 到 1，只看该字段是否抓到了同一事实，不要因为措辞简短就扣太多分。\n"
            "如果 actual 引入 expected 没提到的新事实，需要降低分数。\n"
            f"字段: {field_name}\n"
            f"expected: {expected}\n"
            f"actual: {actual}\n"
        )
        try:
            result = self._judge.invoke(prompt)
        except Exception:
            return self._fallback(field_name, expected, actual)
        return float(result.score), result.reason

    @staticmethod
    def _fallback(field_name: str, expected: str, actual: str) -> Tuple[float, str]:
        expected_tokens = set(_normalize_segments(expected) or [_normalize_text(expected) or ""])
        actual_tokens = set(_normalize_segments(actual) or [_normalize_text(actual) or ""])
        if not expected_tokens or not actual_tokens:
            return 0.0, f"{field_name}: missing text"
        overlap = len(expected_tokens & actual_tokens)
        union = len(expected_tokens | actual_tokens)
        return _safe_ratio(overlap, union), "fallback_token_overlap"


def score_profiles(
    expected_profile: Any,
    actual_profile: Any,
    semantic_matcher: Optional[SemanticMatcher] = None,
) -> ScoreBreakdown:
    expected = _profile_from_any(expected_profile)
    actual = _profile_from_any(actual_profile)
    matcher = semantic_matcher or DeepSeekSemanticMatcher()

    field_details: List[Dict[str, Any]] = []
    expected_units = 0
    matched_units = 0
    actual_units = 0
    hallucinated_units = 0
    fuzzy_scores: List[float] = []

    for field_name in EXACT_SCALAR_FIELDS:
        expected_value = getattr(expected, field_name)
        actual_value = getattr(actual, field_name)

        if expected_value is not None:
            expected_units += 1
            if actual_value is not None:
                actual_units += 1
            is_match = expected_value == actual_value
            if is_match:
                matched_units += 1
            field_details.append(
                {
                    "field": field_name,
                    "mode": "exact",
                    "expected": expected_value,
                    "actual": actual_value,
                    "matched": is_match,
                }
            )
        elif actual_value is not None:
            actual_units += 1
            hallucinated_units += 1
            field_details.append(
                {
                    "field": field_name,
                    "mode": "hallucination",
                    "expected": None,
                    "actual": actual_value,
                    "matched": False,
                }
            )

    expected_destinations = _normalize_list(expected.destination_preference)
    actual_destinations = _normalize_list(actual.destination_preference)
    if expected_destinations:
        expected_units += len(expected_destinations)
        actual_units += len(actual_destinations)
        matched_destinations = sorted(set(expected_destinations) & set(actual_destinations))
        matched_units += len(matched_destinations)
        hallucinated_destinations = sorted(set(actual_destinations) - set(expected_destinations))
        hallucinated_units += len(hallucinated_destinations)
        field_details.append(
            {
                "field": "destination_preference",
                "mode": "exact_list",
                "expected": expected_destinations,
                "actual": actual_destinations,
                "matched": matched_destinations,
                "hallucinated": hallucinated_destinations,
            }
        )
    elif actual_destinations:
        actual_units += len(actual_destinations)
        hallucinated_units += len(actual_destinations)
        field_details.append(
            {
                "field": "destination_preference",
                "mode": "hallucination_list",
                "expected": [],
                "actual": actual_destinations,
                "matched": [],
                "hallucinated": actual_destinations,
            }
        )

    budget_expectations = (
        ("budget.amount", expected.budget.amount, actual.budget.amount, -1),
        ("budget.period", expected.budget.period.value, actual.budget.period.value, BudgetPeriod.UNKNOWN.value),
    )
    for field_name, expected_value, actual_value, empty_value in budget_expectations:
        if expected_value != empty_value:
            expected_units += 1
            if actual_value != empty_value:
                actual_units += 1
            is_match = expected_value == actual_value
            if is_match:
                matched_units += 1
            field_details.append(
                {
                    "field": field_name,
                    "mode": "exact",
                    "expected": expected_value,
                    "actual": actual_value,
                    "matched": is_match,
                }
            )
        elif actual_value != empty_value:
            actual_units += 1
            hallucinated_units += 1
            field_details.append(
                {
                    "field": field_name,
                    "mode": "hallucination",
                    "expected": empty_value,
                    "actual": actual_value,
                    "matched": False,
                }
            )

    for field_name in SEMANTIC_FIELDS:
        expected_value = getattr(expected, field_name)
        actual_value = getattr(actual, field_name)

        if expected_value:
            expected_units += 1
            if actual_value:
                actual_units += 1
            is_match = _text_exact_match(expected_value, actual_value)
            semantic_score = 1.0 if is_match else 0.0
            reason = "exact_match" if is_match else "missing_actual"
            if actual_value and not is_match:
                semantic_score, reason = matcher(field_name, expected_value, actual_value)
            if is_match:
                matched_units += 1
            fuzzy_scores.append(semantic_score)
            field_details.append(
                {
                    "field": field_name,
                    "mode": "semantic",
                    "expected": expected_value,
                    "actual": actual_value,
                    "matched": is_match,
                    "semantic_score": round(semantic_score, 4),
                    "reason": reason,
                }
            )
        elif actual_value:
            actual_units += 1
            hallucinated_units += 1
            field_details.append(
                {
                    "field": field_name,
                    "mode": "hallucination",
                    "expected": None,
                    "actual": actual_value,
                    "matched": False,
                }
            )

    exact_recall = _safe_ratio(matched_units, expected_units)
    hallucination_rate = _safe_ratio(hallucinated_units, actual_units)
    fuzzy_semantic = _safe_ratio(sum(fuzzy_scores), len(fuzzy_scores))

    exact_weight = 60 if expected_units > 0 else 0
    hallucination_weight = 30
    fuzzy_weight = 10 if fuzzy_scores else 0

    exact_points = exact_recall * exact_weight
    hallucination_points = (1 - hallucination_rate) * hallucination_weight
    fuzzy_points = fuzzy_semantic * fuzzy_weight

    active_weight_total = exact_weight + hallucination_weight + fuzzy_weight
    overall_score = _safe_ratio(exact_points + hallucination_points + fuzzy_points, active_weight_total) * 100

    return ScoreBreakdown(
        overall_score=overall_score,
        exact_recall=exact_recall,
        hallucination_rate=hallucination_rate,
        fuzzy_semantic=fuzzy_semantic,
        exact_points=exact_points,
        hallucination_points=hallucination_points,
        fuzzy_points=fuzzy_points,
        expected_units=expected_units,
        matched_units=matched_units,
        actual_units=actual_units,
        hallucinated_units=hallucinated_units,
        active_weight_total=active_weight_total,
        field_details=field_details,
    )


def dump_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
