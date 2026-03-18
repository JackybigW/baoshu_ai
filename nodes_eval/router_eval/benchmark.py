from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


PAIRWISE_ROUTE_SCORES = {
    ("human_handoff", "consultant"): 0.0,
    ("human_handoff", "interviewer"): 0.0,
    ("consultant", "interviewer"): 0.2,
    ("consultant", "high_value"): 0.3,
    ("consultant", "low_budget"): 0.2,
    ("high_value", "consultant"): 0.25,
    ("high_value", "interviewer"): 0.1,
    ("art_director", "high_value"): 0.1,
    ("art_director", "consultant"): 0.15,
    ("low_budget", "consultant"): 0.15,
    ("low_budget", "interviewer"): 0.1,
    ("interviewer", "consultant"): 0.35,
    ("chit_chat", "consultant"): 0.2,
}


@dataclass
class RouterScoreBreakdown:
    overall_score: float
    route_score: float
    expected_route: str
    actual_route: str
    failure_tag: str | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "route_score": round(self.route_score, 4),
            "expected_route": self.expected_route,
            "actual_route": self.actual_route,
            "failure_tag": self.failure_tag,
        }


def classify_failure(expected_route: str, actual_route: str) -> str | None:
    if expected_route == actual_route:
        return None
    if expected_route == "human_handoff":
        return "漏转人工"
    if expected_route == "consultant" and actual_route == "interviewer":
        return "成交信号被资料补全拦截"
    if expected_route == "interviewer" and actual_route != "interviewer":
        return "资料补全漏拦截"
    if expected_route == "low_budget":
        return "低预算分流错误"
    if expected_route == "high_value":
        return "高价值分流错误"
    if expected_route == "art_director":
        return "艺术赛道分流错误"
    if expected_route == "chit_chat":
        return "闲聊误入业务流"
    return "路由优先级错误"


def score_router_result(expected_route: str, actual_route: str) -> RouterScoreBreakdown:
    route_score = 1.0 if expected_route == actual_route else PAIRWISE_ROUTE_SCORES.get((expected_route, actual_route), 0.0)
    overall_score = route_score * 100
    return RouterScoreBreakdown(
        overall_score=overall_score,
        route_score=route_score,
        expected_route=expected_route,
        actual_route=actual_route,
        failure_tag=classify_failure(expected_route, actual_route),
    )
