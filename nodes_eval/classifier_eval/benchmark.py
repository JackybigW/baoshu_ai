from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


CASUAL_INTENTS = {"GREETING", "CHIT_CHAT"}

PAIRWISE_INTENT_SCORES = {
    ("GREETING", "CHIT_CHAT"): 0.9,
    ("CHIT_CHAT", "GREETING"): 0.9,
    ("SALES_READY", "DECISION_SUPPORT"): 0.55,
    ("DECISION_SUPPORT", "SALES_READY"): 0.6,
    ("HIGH_VALUE", "NEED_CONSULTING"): 0.45,
    ("LOW_BUDGET", "NEED_CONSULTING"): 0.3,
    ("ART_CONSULTING", "HIGH_VALUE"): 0.1,
    ("HIGH_VALUE", "ART_CONSULTING"): 0.35,
    ("TRANSFER_TO_HUMAN", "SALES_READY"): 0.0,
    ("TRANSFER_TO_HUMAN", "DECISION_SUPPORT"): 0.0,
    ("TRANSFER_TO_HUMAN", "NEED_CONSULTING"): 0.0,
    ("TRANSFER_TO_HUMAN", "HIGH_VALUE"): 0.0,
    ("TRANSFER_TO_HUMAN", "LOW_BUDGET"): 0.0,
    ("TRANSFER_TO_HUMAN", "ART_CONSULTING"): 0.0,
    ("SALES_READY", "TRANSFER_TO_HUMAN"): 0.7,
    ("DECISION_SUPPORT", "TRANSFER_TO_HUMAN"): 0.6,
    ("HIGH_VALUE", "TRANSFER_TO_HUMAN"): 0.55,
    ("LOW_BUDGET", "TRANSFER_TO_HUMAN"): 0.45,
    ("ART_CONSULTING", "TRANSFER_TO_HUMAN"): 0.5,
}


@dataclass
class ClassifierScoreBreakdown:
    overall_score: float
    intent_score: float
    status_score: float
    expected_intent: str
    actual_intent: Optional[str]
    expected_status: Optional[str]
    actual_status: Optional[str]
    failure_tag: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "intent_score": round(self.intent_score, 4),
            "status_score": round(self.status_score, 4),
            "expected_intent": self.expected_intent,
            "actual_intent": self.actual_intent,
            "expected_status": self.expected_status,
            "actual_status": self.actual_status,
            "failure_tag": self.failure_tag,
        }


def _intent_score(expected: str, actual: Optional[str]) -> float:
    if not actual:
        return 0.0
    if expected == actual:
        return 1.0
    if expected in CASUAL_INTENTS and actual in CASUAL_INTENTS:
        return 0.9
    return PAIRWISE_INTENT_SCORES.get((expected, actual), 0.2)


def _status_score(expected: Optional[str], actual: Optional[str]) -> float:
    if expected is None:
        return 1.0
    return 1.0 if expected == actual else 0.0


def classify_failure(expected_intent: str, actual_intent: Optional[str], expected_status: Optional[str], actual_status: Optional[str]) -> Optional[str]:
    if actual_intent is None:
        return "API/运行异常"
    if expected_intent == actual_intent and expected_status == actual_status:
        return None
    if expected_intent == "TRANSFER_TO_HUMAN" and actual_intent != "TRANSFER_TO_HUMAN":
        return "漏转人工"
    if actual_intent == "TRANSFER_TO_HUMAN" and expected_intent != "TRANSFER_TO_HUMAN":
        return "过度升级"
    if expected_intent in {"HIGH_VALUE", "ART_CONSULTING", "LOW_BUDGET"} and actual_intent == "NEED_CONSULTING":
        return "赛道漏判"
    if expected_intent == "SALES_READY" and actual_intent != "SALES_READY":
        return "销售信号漏判"
    if expected_intent == "DECISION_SUPPORT" and actual_intent != "DECISION_SUPPORT":
        return "决策支持漏判"
    if expected_status and expected_status != actual_status:
        return "状态流转错误"
    return "类别误判"


def score_classifier_result(
    *,
    expected_intent: str,
    actual_intent: Optional[str],
    expected_status: Optional[str],
    actual_status: Optional[str],
) -> ClassifierScoreBreakdown:
    intent_score = _intent_score(expected_intent, actual_intent)
    status_score = _status_score(expected_status, actual_status)
    overall_score = intent_score * 85 + status_score * 15
    failure_tag = classify_failure(expected_intent, actual_intent, expected_status, actual_status)
    return ClassifierScoreBreakdown(
        overall_score=overall_score,
        intent_score=intent_score,
        status_score=status_score,
        expected_intent=expected_intent,
        actual_intent=actual_intent,
        expected_status=expected_status,
        actual_status=actual_status,
        failure_tag=failure_tag,
    )
