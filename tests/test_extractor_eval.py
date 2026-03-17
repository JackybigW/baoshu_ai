from pathlib import Path

from nodes_eval.extractor_eval.benchmark import score_profiles
from nodes_eval.extractor_eval.run_eval import load_cases, summarize_case_results
from state import BudgetInfo, BudgetPeriod, CustomerProfile


DATASET_PATH = Path("/Users/jackywang/Documents/baoshu_ai/nodes_eval/extractor_eval/golden_dataset.json")


def test_golden_dataset_has_100_cases():
    cases = load_cases(DATASET_PATH)
    assert len(cases) == 100


def test_score_profiles_penalizes_hallucination():
    expected = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        budget=BudgetInfo(amount=-1, period=BudgetPeriod.UNKNOWN),
        target_major=None,
    )
    actual = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        target_major="金融",
    )
    actual.budget.amount = 50
    actual.budget.period = BudgetPeriod.TOTAL

    breakdown = score_profiles(expected, actual, semantic_matcher=lambda *_: (0.0, "stub"))

    assert breakdown.exact_recall == 1.0
    assert breakdown.hallucinated_units == 3
    assert breakdown.hallucination_rate == 0.6
    assert breakdown.overall_score == 80.0


def test_score_profiles_rescales_sparse_case_to_full_score():
    expected = CustomerProfile()
    actual = CustomerProfile()

    breakdown = score_profiles(expected, actual, semantic_matcher=lambda *_: (0.0, "stub"))

    assert breakdown.expected_units == 0
    assert breakdown.actual_units == 0
    assert breakdown.active_weight_total == 30
    assert breakdown.overall_score == 100.0


def test_summarize_case_results_aggregates_mean_score():
    summary = summarize_case_results(
        [
            {
                "case_id": "case_001",
                "tags": ["unit_confusion"],
                "error": None,
                "score": {
                    "overall_score": 80.0,
                    "exact_recall": 0.8,
                    "hallucination_rate": 0.1,
                    "fuzzy_semantic": 0.5,
                },
            },
            {
                "case_id": "case_002",
                "tags": ["all_missing"],
                "error": "APIConnectionError: boom",
                "score": {
                    "overall_score": 60.0,
                    "exact_recall": 0.6,
                    "hallucination_rate": 0.3,
                    "fuzzy_semantic": 0.2,
                },
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["overall_score"] == 70.0
    assert summary["exact_recall"] == 0.7
    assert summary["hallucination_rate"] == 0.2
    assert summary["fuzzy_semantic"] == 0.35
    assert summary["error_count"] == 1
