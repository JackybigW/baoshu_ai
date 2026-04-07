import asyncio
import json
from pathlib import Path

from nodes_eval.extractor_eval.benchmark import score_profiles
from nodes_eval.extractor_eval.failure_analysis import generate_failure_analysis
from nodes_eval.extractor_eval.run_eval import (
    load_cases,
    run_cases_async,
    summarize_case_results,
    summarize_model_runs,
)
from state import BudgetInfo, BudgetPeriod, CustomerProfile


DATASET_PATH = Path(__file__).resolve().parents[1] / "nodes_eval/extractor_eval/golden_dataset.json"


def test_golden_dataset_has_100_cases():
    cases = load_cases(DATASET_PATH)
    assert len(cases) == 100


def test_score_profiles_penalizes_hallucination():
    expected = CustomerProfile(
        user_role="学生",
        educationStage="本科",
        budget=BudgetInfo(amount=None, period=BudgetPeriod.UNKNOWN),
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


def test_summarize_model_runs_orders_best_model_first():
    leaderboard = summarize_model_runs(
        [
            {
                "llm": {
                    "label": "qwen",
                    "canonical_id": "qwen",
                    "provider": "openai",
                    "resolved_model": "qwen-plus",
                },
                "summary": {
                    "overall_score": 84.2,
                    "exact_recall": 0.82,
                    "hallucination_rate": 0.11,
                    "fuzzy_semantic": 0.8,
                    "pass_rate": 0.74,
                    "error_count": 0,
                },
            },
            {
                "llm": {
                    "label": "deepseek",
                    "canonical_id": "deepseek",
                    "provider": "openai",
                    "resolved_model": "deepseek-v3",
                },
                "summary": {
                    "overall_score": 88.4,
                    "exact_recall": 0.86,
                    "hallucination_rate": 0.09,
                    "fuzzy_semantic": 0.83,
                    "pass_rate": 0.81,
                    "error_count": 0,
                },
            },
        ]
    )

    assert [item["llm_label"] for item in leaderboard] == ["deepseek", "qwen"]


def test_score_profiles_allows_grounded_extra_semantic_text():
    expected = CustomerProfile()
    actual = CustomerProfile(target_major="传媒")

    breakdown = score_profiles(
        expected,
        actual,
        semantic_matcher=lambda *_: (0.0, "stub"),
        case_context={
            "last_ai_msg": "直接说你想申什么方向。",
            "last_user_msg": "我其实更喜欢传媒。",
        },
    )

    assert breakdown.hallucinated_units == 0
    assert breakdown.overall_score == 100.0
    assert breakdown.field_details[-1]["mode"] == "grounded_extra"


def test_score_profiles_accepts_contextual_target_conflict_resolution():
    expected = CustomerProfile(target_major="传媒")
    actual = CustomerProfile(target_major="金融（家长意向），传媒（孩子意向）")

    breakdown = score_profiles(
        expected,
        actual,
        semantic_matcher=lambda *_: (0.0, "stub"),
        case_context={
            "last_ai_msg": "主要想读什么方向？我需要分清楚孩子自己和家长的想法。",
            "last_user_msg": "我想让他读金融更稳，但他自己一直念叨传媒。",
        },
    )

    assert breakdown.exact_recall == 1.0
    assert breakdown.fuzzy_semantic == 1.0
    assert breakdown.overall_score == 100.0
    assert breakdown.field_details[-1]["reason"] == "contextual_conflict_resolution"


def test_failure_analysis_prioritizes_major_failures(tmp_path: Path):
    payload = {
        "summary": {
            "case_count": 2,
            "overall_score": 92.5,
            "exact_recall": 0.95,
            "hallucination_rate": 0.02,
            "fuzzy_semantic": 0.9,
            "pass_rate": 1.0,
            "error_count": 0,
        },
        "results": [
            {
                "case_id": "case_001",
                "tags": ["semantic_only"],
                "input": {
                    "last_ai_msg": "说一下目标方向。",
                    "last_user_msg": "我想读传媒。",
                },
                "expected": {
                    "user_role": None,
                    "educationStage": None,
                    "budget": {"amount": None, "period": "UNKNOWN"},
                    "destination_preference": None,
                    "abroad_readiness": None,
                    "target_school": None,
                    "target_major": "传媒",
                    "academic_background": None,
                    "language_level": None,
                },
                "actual": {
                    "user_role": None,
                    "educationStage": None,
                    "budget": {"amount": None, "period": "UNKNOWN"},
                    "destination_preference": None,
                    "abroad_readiness": None,
                    "target_school": None,
                    "target_major": "数字媒体",
                    "academic_background": None,
                    "language_level": None,
                },
                "error": None,
                "score": {"overall_score": 95.0},
            },
            {
                "case_id": "case_002",
                "tags": ["currency_conversion"],
                "input": {
                    "last_ai_msg": "预算如果不是人民币也直接说。",
                    "last_user_msg": "我预算5万美金。",
                },
                "expected": {
                    "user_role": None,
                    "educationStage": None,
                    "budget": {"amount": 36, "period": "TOTAL"},
                    "destination_preference": None,
                    "abroad_readiness": None,
                    "target_school": None,
                    "target_major": None,
                    "academic_background": None,
                    "language_level": None,
                },
                "actual": {
                    "user_role": None,
                    "educationStage": None,
                    "budget": {"amount": 5, "period": "TOTAL"},
                    "destination_preference": None,
                    "abroad_readiness": None,
                    "target_school": None,
                    "target_major": None,
                    "academic_background": None,
                    "language_level": None,
                },
                "error": None,
                "score": {"overall_score": 70.0},
            },
        ],
    }

    output_dir = generate_failure_analysis(payload, output_root=tmp_path)
    abnormal_cases = json.loads((output_dir / "abnormal_cases.json").read_text(encoding="utf-8"))

    assert abnormal_cases[0]["case_id"] == "case_002"
    assert abnormal_cases[0]["failure_tags"][0] == "单位/汇率/预算周期崩塌"


def test_failure_analysis_groups_results_under_model_label(tmp_path: Path):
    payload = {
        "llm": {
            "label": "qwen",
            "canonical_id": "qwen",
            "provider": "openai",
            "resolved_model": "qwen-plus",
        },
        "summary": {
            "case_count": 1,
            "overall_score": 70.0,
            "exact_recall": 0.7,
            "hallucination_rate": 0.1,
            "fuzzy_semantic": 0.6,
            "pass_rate": 0.0,
            "error_count": 0,
        },
        "results": [
            {
                "case_id": "case_001",
                "tags": ["currency_conversion"],
                "input": {
                    "last_ai_msg": "预算直接说。",
                    "last_user_msg": "5万美金。",
                },
                "expected": {
                    "user_role": None,
                    "educationStage": None,
                    "budget": {"amount": 36, "period": "TOTAL"},
                    "destination_preference": None,
                    "abroad_readiness": None,
                    "target_school": None,
                    "target_major": None,
                    "academic_background": None,
                    "language_level": None,
                },
                "actual": {
                    "user_role": None,
                    "educationStage": None,
                    "budget": {"amount": 5, "period": "TOTAL"},
                    "destination_preference": None,
                    "abroad_readiness": None,
                    "target_school": None,
                    "target_major": None,
                    "academic_background": None,
                    "language_level": None,
                },
                "error": None,
                "score": {"overall_score": 70.0},
            }
        ],
    }

    output_dir = generate_failure_analysis(
        payload,
        output_root=tmp_path,
        model_label="qwen",
    )

    assert output_dir.name == "qwen"
    assert (output_dir / "summary.md").exists()
    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "llm_label: `qwen`" in summary_text


def test_run_cases_async_preserves_case_order_for_single_model(monkeypatch):
    cases = load_cases(DATASET_PATH)[:3]

    class StubModelConfig:
        label = "deepseek"

    def fake_run_single_case(case, model_config, semantic_matcher):
        return {
            "llm": {"label": model_config.label},
            "case_id": case.case_id,
            "tags": case.tags,
            "input": {},
            "expected": {},
            "actual": {},
            "error": None,
            "score": {"overall_score": float(ord(case.case_id[-1]))},
        }

    monkeypatch.setattr("nodes_eval.extractor_eval.run_eval.run_single_case", fake_run_single_case)

    results = asyncio.run(run_cases_async(cases, StubModelConfig(), semantic_matcher=None, concurrency=2))

    assert [item["case_id"] for item in results] == sorted(case.case_id for case in cases)
