import asyncio
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from nodes_eval.execution_eval.benchmark import score_execution_output
from nodes_eval.execution_eval.failure_analysis import generate_failure_analysis
from nodes_eval.execution_eval.run_eval import load_cases, run_cases_async, summarize_case_results


DATASET_PATH = Path(__file__).resolve().parents[1] / "nodes_eval/execution_eval/golden_dataset.json"


class StubJudge:
    def __init__(self, score: float = 0.8, reason: str = "ok", failure_tags=None):
        self.score = score
        self.reason = reason
        self.failure_tags = failure_tags or []

    def evaluate(self, **_kwargs):
        return self.score, self.reason, self.failure_tags


class RecordingJudge:
    def evaluate(self, **_kwargs):
        return 0.8, "ok", []


def test_execution_golden_dataset_has_14_cases():
    cases = load_cases(DATASET_PATH)
    assert len(cases) == 14


def test_score_execution_output_detects_missing_required_tool_call():
    breakdown = score_execution_output(
        node_name="consultant",
        contract={
            "must_call_tool": True,
            "min_segments": 1,
            "max_segments": 3,
            "max_chars_per_segment": 40,
            "required_keyword_groups": [["顾问"], ["群"]],
            "expected_status": "PERSUADING",
        },
        case_input={"messages": [{"type": "human", "content": "安排老师"}]},
        output_messages=[AIMessage(content="我先说下方案")],
        actual_status="PERSUADING",
        judge=StubJudge(score=0.7),
    )

    assert breakdown.tool_score == 0.0
    assert "该拉群未拉群" in breakdown.failure_tags


def test_score_execution_output_handles_tool_message_protocol_case():
    breakdown = score_execution_output(
        node_name="human_handoff",
        contract={
            "required_message_types": ["tool"],
            "min_segments": 1,
            "max_segments": 1,
            "expected_status": "FINISHED",
            "skip_rubric": True,
        },
        case_input={},
        output_messages=[ToolMessage(content="ok", tool_call_id="tool_1")],
        actual_status="FINISHED",
        judge=StubJudge(),
    )

    assert breakdown.message_type_score == 1.0
    assert breakdown.overall_score == 100.0


def test_score_execution_output_normalizes_judge_failure_tags():
    breakdown = score_execution_output(
        node_name="consultant",
        contract={
            "must_call_tool": False,
            "min_segments": 1,
            "max_segments": 3,
        },
        case_input={},
        output_messages=[AIMessage(content="先说方案")],
        actual_status=None,
        judge=StubJudge(score=0.4, failure_tags=["failed_to_handoff", "LacksPersonalization"]),
    )

    assert "该拉群未拉群" in breakdown.failure_tags
    assert "关键要点缺失" in breakdown.failure_tags


def test_summarize_case_results_aggregates_execution_metrics():
    summary = summarize_case_results(
        [
            {
                "case_id": "exe_001",
                "node_name": "first_greeting",
                "tags": ["deterministic"],
                "error": None,
                "score": {
                    "overall_score": 100.0,
                    "tool_score": None,
                    "format_score": 1.0,
                    "keyword_score": 1.0,
                    "rubric_score": None,
                    "failure_tags": [],
                },
            },
            {
                "case_id": "exe_002",
                "node_name": "consultant",
                "tags": ["handoff"],
                "error": "RuntimeError: boom",
                "score": {
                    "overall_score": 60.0,
                    "tool_score": 0.0,
                    "format_score": 0.8,
                    "keyword_score": 0.5,
                    "rubric_score": 0.7,
                    "failure_tags": ["该拉群未拉群"],
                },
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["overall_score"] == 80.0
    assert summary["tool_success_rate"] == 0.0
    assert summary["format_score"] == 0.9
    assert summary["keyword_score"] == 0.75
    assert summary["rubric_score"] == 0.7
    assert summary["error_count"] == 1


def test_run_cases_async_preserves_case_order(monkeypatch):
    cases = load_cases(DATASET_PATH)[:3]

    def fake_run_single_case(case, *, judge):
        return {
            "case_id": case.case_id,
            "node_name": case.node_name,
            "tags": case.tags,
            "input": {},
            "expected": {},
            "actual": {},
            "error": None,
            "score": {"overall_score": float(ord(case.case_id[-1]))},
        }

    monkeypatch.setattr("nodes_eval.execution_eval.run_eval.run_single_case", fake_run_single_case)

    results = asyncio.run(run_cases_async(cases, judge=RecordingJudge(), concurrency=2))

    assert [item["case_id"] for item in results] == sorted(case.case_id for case in cases)


def test_execution_failure_analysis_prioritizes_missing_tool_calls(tmp_path: Path):
    payload = {
        "summary": {
            "case_count": 2,
            "overall_score": 75.0,
            "tool_success_rate": 0.5,
            "format_score": 0.9,
            "keyword_score": 0.8,
            "rubric_score": 0.7,
            "pass_rate": 0.5,
            "error_count": 0,
        },
        "results": [
            {
                "case_id": "exe_005",
                "node_name": "consultant",
                "tags": ["handoff"],
                "input": {"messages": [{"type": "human", "content": "安排老师"}]},
                "expected": {"must_call_tool": True},
                "actual": {"messages": [{"type": "ai", "content": "先说方案"}]},
                "score": {"overall_score": 55.0, "failure_tags": ["该拉群未拉群"]},
            },
            {
                "case_id": "exe_014",
                "node_name": "chit_chat",
                "tags": ["casual"],
                "input": {"messages": [{"type": "human", "content": "你挺有意思"}]},
                "expected": {"must_call_tool": False},
                "actual": {"messages": [{"type": "ai", "content": "哈哈"}]},
                "score": {"overall_score": 85.0, "failure_tags": ["格式违规"]},
            },
        ],
    }

    output_dir = generate_failure_analysis(payload, output_root=tmp_path)
    abnormal_cases = json.loads((output_dir / "abnormal_cases.json").read_text(encoding="utf-8"))
    assert abnormal_cases[0]["case_id"] == "exe_005"
