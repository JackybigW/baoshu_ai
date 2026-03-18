import json
from pathlib import Path

from nodes_eval.classifier_eval.benchmark import score_classifier_result
from nodes_eval.classifier_eval.failure_analysis import generate_failure_analysis
from nodes_eval.classifier_eval.run_eval import load_cases, summarize_case_results


DATASET_PATH = Path(__file__).resolve().parents[1] / "nodes_eval/classifier_eval/golden_dataset.json"


def test_classifier_golden_dataset_has_20_cases():
    cases = load_cases(DATASET_PATH)
    assert len(cases) == 20


def test_score_classifier_result_penalizes_missed_handoff():
    breakdown = score_classifier_result(
        expected_intent="TRANSFER_TO_HUMAN",
        actual_intent="NEED_CONSULTING",
        expected_status=None,
        actual_status="CONSULTING",
    )

    assert breakdown.intent_score == 0.0
    assert breakdown.overall_score == 15.0
    assert breakdown.failure_tag == "漏转人工"


def test_summarize_case_results_aggregates_classifier_metrics():
    summary = summarize_case_results(
        [
            {
                "case_id": "clf_001",
                "tags": ["handoff"],
                "error": None,
                "score": {
                    "overall_score": 100.0,
                    "intent_score": 1.0,
                    "status_score": 1.0,
                    "failure_tag": None,
                },
            },
            {
                "case_id": "clf_002",
                "tags": ["sticky"],
                "error": "RuntimeError: boom",
                "score": {
                    "overall_score": 42.5,
                    "intent_score": 0.5,
                    "status_score": 0.0,
                    "failure_tag": "赛道漏判",
                },
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["overall_score"] == 71.25
    assert summary["intent_accuracy"] == 0.75
    assert summary["status_accuracy"] == 0.5
    assert summary["error_count"] == 1


def test_classifier_failure_analysis_orders_missed_handoff_first(tmp_path: Path):
    payload = {
        "summary": {
            "case_count": 2,
            "overall_score": 70.0,
            "intent_accuracy": 0.6,
            "status_accuracy": 0.7,
            "pass_rate": 0.5,
            "error_count": 0,
        },
        "results": [
            {
                "case_id": "clf_001",
                "tags": ["handoff"],
                "input": {"messages": [{"type": "human", "content": "给我电话"}]},
                "expected": {"intent": "TRANSFER_TO_HUMAN"},
                "actual": {"intent": "NEED_CONSULTING"},
                "score": {"overall_score": 15.0, "failure_tag": "漏转人工"},
            },
            {
                "case_id": "clf_002",
                "tags": ["casual"],
                "input": {"messages": [{"type": "human", "content": "你好"}]},
                "expected": {"intent": "GREETING"},
                "actual": {"intent": "CHIT_CHAT"},
                "score": {"overall_score": 90.0, "failure_tag": "类别误判"},
            },
        ],
    }

    output_dir = generate_failure_analysis(payload, output_root=tmp_path)
    abnormal_cases = json.loads((output_dir / "abnormal_cases.json").read_text(encoding="utf-8"))
    assert abnormal_cases[0]["case_id"] == "clf_001"
