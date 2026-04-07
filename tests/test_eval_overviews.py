from pathlib import Path

from nodes_eval.classifier_eval.run_eval import write_run_overview as write_classifier_run_overview
from nodes_eval.execution_eval.run_eval import write_run_overview as write_execution_run_overview
from nodes_eval.extractor_eval.run_eval import write_run_overview as write_extractor_run_overview


def _sample_model_runs():
    return [
        {
            "llm": {
                "label": "deepseek",
                "canonical_id": "deepseek",
                "provider": "openai",
                "resolved_model": "deepseek-v3-2-251201",
            },
            "summary": {
                "overall_score": 91.2,
                "exact_recall": 0.9,
                "hallucination_rate": 0.05,
                "fuzzy_semantic": 0.88,
                "intent_accuracy": 0.95,
                "status_accuracy": 0.9,
                "tool_success_rate": 1.0,
                "format_score": 0.98,
                "keyword_score": 0.9,
                "rubric_score": 0.85,
                "pass_rate": 0.8,
                "error_count": 0,
            },
            "failure_analysis_dir": "/tmp/failure_analyses/deepseek",
        }
    ]


def test_extractor_run_overview_includes_resolved_model(tmp_path: Path):
    run_root = tmp_path / "extractor_run"
    run_root.mkdir()

    write_extractor_run_overview(
        run_root=run_root,
        dataset_path=tmp_path / "golden_dataset.json",
        model_runs=_sample_model_runs(),
    )

    readme_text = (run_root / "README.md").read_text(encoding="utf-8")
    assert "deepseek-v3-2-251201" in readme_text


def test_classifier_run_overview_includes_resolved_model(tmp_path: Path):
    run_root = tmp_path / "classifier_run"
    run_root.mkdir()

    write_classifier_run_overview(
        run_root=run_root,
        dataset_path=tmp_path / "golden_dataset.json",
        model_runs=_sample_model_runs(),
    )

    readme_text = (run_root / "README.md").read_text(encoding="utf-8")
    assert "deepseek-v3-2-251201" in readme_text


def test_execution_run_overview_includes_resolved_model(tmp_path: Path):
    run_root = tmp_path / "execution_run"
    run_root.mkdir()

    write_execution_run_overview(
        run_root=run_root,
        dataset_path=tmp_path / "golden_dataset.json",
        model_runs=_sample_model_runs(),
    )

    readme_text = (run_root / "README.md").read_text(encoding="utf-8")
    assert "deepseek-v3-2-251201" in readme_text
