import json
from pathlib import Path

from nodes_eval.router_eval.benchmark import score_router_result
from nodes_eval.router_eval.failure_analysis import generate_failure_analysis
from nodes_eval.router_eval.run_eval import load_cases, summarize_case_results


DATASET_PATH = Path(__file__).resolve().parents[1] / "nodes_eval/router_eval/golden_dataset.json"


def test_router_golden_dataset_has_16_cases():
    cases = load_cases(DATASET_PATH)
    assert len(cases) == 16


def test_score_router_result_penalizes_sales_routed_to_interviewer():
    breakdown = score_router_result("consultant", "interviewer")

    assert breakdown.route_score == 0.2
    assert breakdown.overall_score == 20.0
    assert breakdown.failure_tag == "成交信号被资料补全拦截"


def test_summarize_case_results_aggregates_router_metrics():
    summary = summarize_case_results(
        [
            {
                "case_id": "rtr_001",
                "tags": ["ok"],
                "score": {"overall_score": 100.0, "route_score": 1.0, "failure_tag": None},
            },
            {
                "case_id": "rtr_002",
                "tags": ["priority"],
                "score": {"overall_score": 20.0, "route_score": 0.2, "failure_tag": "成交信号被资料补全拦截"},
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["overall_score"] == 60.0
    assert summary["route_accuracy"] == 0.6
    assert summary["pass_rate"] == 0.5


def test_router_failure_analysis_prioritizes_handoff_errors(tmp_path: Path):
    payload = {
        "summary": {
            "case_count": 2,
            "overall_score": 50.0,
            "route_accuracy": 0.5,
            "pass_rate": 0.0,
        },
        "results": [
            {
                "case_id": "rtr_001",
                "tags": ["handoff"],
                "input": {"last_intent": "TRANSFER_TO_HUMAN"},
                "expected": {"route": "human_handoff"},
                "actual": {"route": "consultant"},
                "score": {"overall_score": 0.0, "failure_tag": "漏转人工"},
            },
            {
                "case_id": "rtr_002",
                "tags": ["profiling"],
                "input": {"last_intent": "NEED_CONSULTING"},
                "expected": {"route": "interviewer"},
                "actual": {"route": "consultant"},
                "score": {"overall_score": 35.0, "failure_tag": "资料补全漏拦截"},
            },
        ],
    }

    output_dir = generate_failure_analysis(payload, output_root=tmp_path)
    abnormal_cases = json.loads((output_dir / "abnormal_cases.json").read_text(encoding="utf-8"))
    assert abnormal_cases[0]["case_id"] == "rtr_001"
