from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from nodes_eval.common import dump_messages, has_tool_call, join_message_text
from utils.llm_factory import get_backend_llm


class RubricJudgement(BaseModel):
    task_completion: float = Field(ge=0.0, le=1.0)
    policy_compliance: float = Field(ge=0.0, le=1.0)
    handoff_timing: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    tone_fit: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="")
    failure_tags: List[str] = Field(default_factory=list)


FAILURE_TAG_ALIASES = {
    "failed_to_handoff": "该拉群未拉群",
    "ignored_user_request": "关键要点缺失",
    "violated_rubric_instruction": "回复质量偏低",
    "insufficientanalysis": "回复质量偏低",
    "lackspersonalization": "关键要点缺失",
}


@dataclass
class ExecutionScoreBreakdown:
    overall_score: float
    tool_score: Optional[float]
    format_score: Optional[float]
    keyword_score: Optional[float]
    status_score: Optional[float]
    message_type_score: Optional[float]
    rubric_score: Optional[float]
    active_weight_total: int
    failure_tags: List[str]
    judge_reason: str
    output_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 2),
            "tool_score": None if self.tool_score is None else round(self.tool_score, 4),
            "format_score": None if self.format_score is None else round(self.format_score, 4),
            "keyword_score": None if self.keyword_score is None else round(self.keyword_score, 4),
            "status_score": None if self.status_score is None else round(self.status_score, 4),
            "message_type_score": None if self.message_type_score is None else round(self.message_type_score, 4),
            "rubric_score": None if self.rubric_score is None else round(self.rubric_score, 4),
            "active_weight_total": self.active_weight_total,
            "failure_tags": self.failure_tags,
            "judge_reason": self.judge_reason,
            "output_text": self.output_text,
        }


class BackendRubricJudge:
    def __init__(self) -> None:
        self._judge = None
        llm = get_backend_llm(temperature=0)
        if llm is None:
            return
        self._judge = llm.with_structured_output(RubricJudgement)

    @property
    def enabled(self) -> bool:
        return self._judge is not None

    def evaluate(
        self,
        *,
        node_name: str,
        contract: Dict[str, Any],
        case_input: Dict[str, Any],
        output_text: str,
    ) -> Tuple[float, str, List[str]]:
        if self._judge is None:
            return self._fallback(contract=contract, output_text=output_text)

        prompt = (
            "你是执行层节点 eval 审稿器。请根据原始对话、节点职责和期望合同，"
            "判断回复是否完成任务、是否合规、拉群时机是否合理、是否贴合上下文。\n"
            "评分范围 0 到 1，每个维度单独给分。failure_tags 只能从以下标签中选择："
            "关键要点缺失、该拉群未拉群、误拉群/过早拉群、回复质量偏低、状态流转错误、格式违规。\n"
            f"节点: {node_name}\n"
            f"节点目标: {contract.get('node_goal', '')}\n"
            f"Rubric 备注: {contract.get('rubric_notes', '')}\n"
            f"Fatal errors: {contract.get('fatal_errors', [])}\n"
            f"输入状态: {case_input}\n"
            f"实际输出: {output_text}\n"
        )
        try:
            result = self._judge.invoke(prompt)
        except Exception:
            return self._fallback(contract=contract, output_text=output_text)

        dimension_score = (
            float(result.task_completion) * 0.3
            + float(result.policy_compliance) * 0.2
            + float(result.handoff_timing) * 0.2
            + float(result.grounding) * 0.2
            + float(result.tone_fit) * 0.1
        )
        return dimension_score, result.reason, _normalize_failure_tags(result.failure_tags)

    @staticmethod
    def _fallback(*, contract: Dict[str, Any], output_text: str) -> Tuple[float, str, List[str]]:
        if not output_text.strip():
            return 0.0, "fallback_empty_response", ["空回复"]
        required_groups = contract.get("required_keyword_groups") or []
        if required_groups:
            matched = 0
            lowered = output_text.lower()
            for group in required_groups:
                group_tokens = [token.lower() for token in group]
                if any(token in lowered for token in group_tokens):
                    matched += 1
            score = matched / len(required_groups)
        else:
            score = 0.75
        if len(output_text.strip()) < 6:
            score = min(score, 0.4)
        return score, "fallback_keyword_coverage", []


def _keyword_score(text: str, required_groups: Sequence[Sequence[str]], forbidden_keywords: Sequence[str]) -> Tuple[float, List[str]]:
    if not required_groups and not forbidden_keywords:
        return 1.0, []

    lowered = text.lower()
    matched = 0
    failure_tags: List[str] = []
    for group in required_groups:
        candidates = [candidate.lower() for candidate in group]
        if any(candidate in lowered for candidate in candidates):
            matched += 1
    score = matched / len(required_groups) if required_groups else 1.0
    if required_groups and score < 1.0:
        failure_tags.append("关键要点缺失")

    forbidden_hits = [token for token in forbidden_keywords if token.lower() in lowered]
    if forbidden_hits:
        score = max(0.0, score - 0.5)
        failure_tags.append("命中禁用表达")
    return score, failure_tags


def _format_score(
    output_messages: Sequence[Any],
    *,
    min_segments: int,
    max_segments: int,
    max_chars_per_segment: Optional[int],
) -> Tuple[float, List[str]]:
    if not output_messages:
        return 0.0, ["空回复"]

    score = 1.0
    failure_tags: List[str] = []
    segment_count = len(output_messages)
    if segment_count < min_segments or segment_count > max_segments:
        score -= 0.35
        failure_tags.append("格式违规")

    if max_chars_per_segment:
        too_long_segments = [
            str(getattr(message, "content", "") or "")
            for message in output_messages
            if len(str(getattr(message, "content", "") or "").strip()) > max_chars_per_segment
        ]
        if too_long_segments:
            score -= 0.35
            if "格式违规" not in failure_tags:
                failure_tags.append("格式违规")

    output_text = join_message_text(output_messages)
    if "**" in output_text:
        score -= 0.2
        if "格式违规" not in failure_tags:
            failure_tags.append("格式违规")
    return max(0.0, score), failure_tags


def _message_type_score(output_messages: Sequence[Any], required_types: Sequence[str]) -> Tuple[float, List[str]]:
    if not required_types:
        return 1.0, []
    actual_types = [item["type"] for item in dump_messages(output_messages)]
    score = 1.0 if actual_types == list(required_types) else 0.0
    return score, ([] if score == 1.0 else ["输出消息类型错误"])


def _normalize_failure_tags(tags: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for tag in tags:
        token = str(tag).strip()
        key = token.lower().replace(" ", "_")
        normalized.append(FAILURE_TAG_ALIASES.get(key, token))
    return list(dict.fromkeys(normalized))


def score_execution_output(
    *,
    node_name: str,
    contract: Dict[str, Any],
    case_input: Dict[str, Any],
    output_messages: Sequence[Any],
    actual_status: Optional[str],
    judge: Optional[BackendRubricJudge],
) -> ExecutionScoreBreakdown:
    failure_tags: List[str] = []
    output_text = join_message_text(output_messages)

    component_points = 0.0
    active_weight_total = 0

    tool_score: Optional[float] = None
    if contract.get("must_call_tool") is not None:
        active_weight_total += 25
        tool_score = 1.0 if has_tool_call(output_messages) == bool(contract["must_call_tool"]) else 0.0
        component_points += tool_score * 25
        if tool_score < 1.0:
            failure_tags.append("该拉群未拉群" if contract["must_call_tool"] else "误拉群/过早拉群")

    format_score, format_failures = _format_score(
        output_messages,
        min_segments=int(contract.get("min_segments", 1)),
        max_segments=int(contract.get("max_segments", 3)),
        max_chars_per_segment=contract.get("max_chars_per_segment"),
    )
    active_weight_total += 15
    component_points += format_score * 15
    failure_tags.extend(format_failures)

    keyword_score: Optional[float] = None
    if contract.get("required_keyword_groups") or contract.get("forbidden_keywords"):
        keyword_score, keyword_failures = _keyword_score(
            output_text,
            contract.get("required_keyword_groups") or [],
            contract.get("forbidden_keywords") or [],
        )
        active_weight_total += 15
        component_points += keyword_score * 15
        failure_tags.extend(keyword_failures)

    status_score: Optional[float] = None
    expected_status = contract.get("expected_status")
    if expected_status is not None:
        status_score = 1.0 if actual_status == expected_status else 0.0
        active_weight_total += 10
        component_points += status_score * 10
        if status_score < 1.0:
            failure_tags.append("状态流转错误")

    message_type_score: Optional[float] = None
    if contract.get("required_message_types"):
        message_type_score, type_failures = _message_type_score(output_messages, contract["required_message_types"])
        active_weight_total += 10
        component_points += message_type_score * 10
        failure_tags.extend(type_failures)

    rubric_score: Optional[float] = None
    judge_reason = ""
    if not contract.get("skip_rubric"):
        if judge is None:
            rubric_score, judge_reason, judge_failures = BackendRubricJudge._fallback(
                contract=contract,
                output_text=output_text,
            )
        else:
            rubric_score, judge_reason, judge_failures = judge.evaluate(
                node_name=node_name,
                contract=contract,
                case_input=case_input,
                output_text=output_text,
            )
        active_weight_total += 25
        component_points += rubric_score * 25
        failure_tags.extend(judge_failures)
        if rubric_score < 0.65:
            failure_tags.append("回复质量偏低")

    deduped_failure_tags = _normalize_failure_tags(tag for tag in failure_tags if tag)
    overall_score = 100.0 if active_weight_total == 0 else component_points / active_weight_total * 100
    return ExecutionScoreBreakdown(
        overall_score=overall_score,
        tool_score=tool_score,
        format_score=format_score,
        keyword_score=keyword_score,
        status_score=status_score,
        message_type_score=message_type_score,
        rubric_score=rubric_score,
        active_weight_total=active_weight_total,
        failure_tags=deduped_failure_tags,
        judge_reason=judge_reason,
        output_text=output_text,
    )
