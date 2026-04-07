from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.perception import classifier_node
from nodes_eval.classifier_eval.benchmark import score_classifier_result
from nodes_eval.classifier_eval.failure_analysis import generate_failure_analysis
from nodes_eval.common import (
    append_key_value_log,
    build_backend_model_configs,
    load_json,
    messages_from_dicts,
)
from state import CustomerProfile

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_MASTER_LOG_PATH = Path(__file__).resolve().parent / "eval_classifier.log"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"


class EvalInput(BaseModel):
    messages: List[Dict[str, Any]]
    profile: CustomerProfile = Field(default_factory=CustomerProfile)
    last_intent: Optional[str] = Field(default=None)
    dialog_status: Optional[str] = Field(default=None)


class EvalExpected(BaseModel):
    intent: str
    dialog_status: Optional[str] = Field(default=None)


class EvalCase(BaseModel):
    case_id: str
    tags: List[str] = Field(default_factory=list)
    input: EvalInput
    expected: EvalExpected


def load_cases(dataset_path: Path) -> List[EvalCase]:
    return [EvalCase.model_validate(item) for item in load_json(dataset_path)]


def run_single_case(case: EvalCase, model_config: Any) -> Dict[str, Any]:
    state = {
        "messages": messages_from_dicts(case.input.messages),
        "profile": case.input.profile,
        "last_intent": case.input.last_intent,
        "dialog_status": case.input.dialog_status,
        "runtime_config": {
            "backend_llm": model_config.llm,
            "backend_model": model_config.canonical_id,
            "eval_llm_label": model_config.label,
        },
    }

    error = None
    try:
        result = classifier_node(state)
        actual_intent = result.get("last_intent")
        actual_status = result.get("dialog_status")
    except Exception as exc:
        actual_intent = None
        actual_status = None
        error = f"{type(exc).__name__}: {exc}"

    breakdown = score_classifier_result(
        expected_intent=case.expected.intent,
        actual_intent=actual_intent,
        expected_status=case.expected.dialog_status,
        actual_status=actual_status,
    )
    return {
        "llm": model_config.to_dict(),
        "case_id": case.case_id,
        "tags": case.tags,
        "input": case.input.model_dump(mode="json", exclude_none=True),
        "expected": case.expected.model_dump(mode="json", exclude_none=True),
        "actual": {
            "intent": actual_intent,
            "dialog_status": actual_status,
        },
        "error": error,
        "score": breakdown.to_dict(),
    }


def summarize_case_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "case_count": 0,
            "overall_score": 0.0,
            "intent_accuracy": 0.0,
            "status_accuracy": 0.0,
            "pass_rate": 0.0,
            "error_count": 0,
            "lowest_cases": [],
        }

    overall_scores = [item["score"]["overall_score"] for item in results]
    intent_scores = [item["score"]["intent_score"] for item in results]
    status_scores = [item["score"]["status_score"] for item in results]
    passed_cases = sum(1 for item in results if item["score"]["overall_score"] >= 85)
    error_count = sum(1 for item in results if item.get("error"))
    lowest_cases = sorted(results, key=lambda item: item["score"]["overall_score"])[:10]

    return {
        "case_count": len(results),
        "overall_score": round(mean(overall_scores), 2),
        "intent_accuracy": round(mean(intent_scores), 4),
        "status_accuracy": round(mean(status_scores), 4),
        "pass_rate": round(passed_cases / len(results), 4),
        "error_count": error_count,
        "lowest_cases": [
            {
                "case_id": item["case_id"],
                "overall_score": item["score"]["overall_score"],
                "failure_tag": item["score"].get("failure_tag"),
                "tags": item["tags"],
            }
            for item in lowest_cases
        ],
    }


def summarize_model_runs(model_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    leaderboard = []
    for item in model_runs:
        summary = item["summary"]
        leaderboard.append(
            {
                "llm_label": item["llm"]["label"],
                "canonical_id": item["llm"]["canonical_id"],
                "provider": item["llm"]["provider"],
                "resolved_model": item["llm"]["resolved_model"],
                "overall_score": summary["overall_score"],
                "intent_accuracy": summary["intent_accuracy"],
                "status_accuracy": summary["status_accuracy"],
                "pass_rate": summary["pass_rate"],
                "error_count": summary["error_count"],
            }
        )
    leaderboard.sort(
        key=lambda item: (
            -item["overall_score"],
            -item["intent_accuracy"],
            -item["pass_rate"],
            item["error_count"],
            item["llm_label"],
        )
    )
    return leaderboard


def _per_model_log_path(label: str) -> Path:
    return DEFAULT_LOG_DIR / f"classifier_eval_{label}.log"


def append_log(
    summary: Dict[str, Any],
    *,
    dataset_path: Path,
    llm_meta: Dict[str, Any],
    log_paths: List[Path],
    failure_analysis_dir: Path,
) -> None:
    append_key_value_log(
        title="classifier_eval",
        kv_pairs=[
            ("dataset", str(dataset_path)),
            ("llm_label", llm_meta["label"]),
            ("llm_canonical", llm_meta["canonical_id"]),
            ("llm_provider", llm_meta["provider"]),
            ("llm_model", llm_meta["resolved_model"]),
            ("case_count", summary["case_count"]),
            ("overall_score", summary["overall_score"]),
            ("intent_accuracy", summary["intent_accuracy"]),
            ("status_accuracy", summary["status_accuracy"]),
            ("pass_rate", summary["pass_rate"]),
            ("error_count", summary["error_count"]),
            ("failure_analysis_dir", str(failure_analysis_dir)),
            ("lowest_cases", summary["lowest_cases"]),
        ],
        log_paths=log_paths,
    )


async def run_cases_async(cases: List[EvalCase], model_config: Any, concurrency: int) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_with_limit(case: EvalCase) -> Dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(run_single_case, case, model_config)

    results = await asyncio.gather(*(run_with_limit(case) for case in cases))
    results.sort(key=lambda item: item["case_id"])
    return results


def write_run_overview(*, run_root: Path, dataset_path: Path, model_runs: List[Dict[str, Any]]) -> None:
    leaderboard = summarize_model_runs(model_runs)
    lines = [
        f"# Classifier Eval Run {run_root.name}",
        "",
        f"- dataset: `{dataset_path}`",
        f"- llm_count: `{len(model_runs)}`",
        "",
        "## Leaderboard",
        "",
    ]
    for row in leaderboard:
        lines.append(
            "- "
            f"`{row['llm_label']}` | "
            f"model `{row['resolved_model']}` | "
            f"score `{row['overall_score']}` | "
            f"intent `{row['intent_accuracy']}` | "
            f"status `{row['status_accuracy']}` | "
            f"pass_rate `{row['pass_rate']}` | "
            f"errors `{row['error_count']}`"
        )
    (run_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (run_root / "leaderboard.json").write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classifier eval pipeline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset json.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases.")
    parser.add_argument("--case-id", default="", help="Only run one specific case id.")
    parser.add_argument("--tag", action="append", default=[], help="Only run cases that contain the given tag.")
    parser.add_argument(
        "--llm",
        action="append",
        default=[],
        help="Run classifier with a specific LLM alias. Repeatable. If omitted, uses backend_default.",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Max number of case executions to run concurrently within one requested LLM.")
    parser.add_argument("--output-json", default="", help="Optional path to write full result json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    cases = load_cases(dataset_path)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.tag:
        tags = set(args.tag)
        cases = [case for case in cases if tags.intersection(case.tags)]
    if args.limit > 0:
        cases = cases[: args.limit]

    model_configs = build_backend_model_configs(args.llm, temperature=0)
    run_timestamp = datetime.now()
    run_root = Path(__file__).resolve().parent / "failure_analyses" / run_timestamp.strftime("%Y%m%d_%H%M%S")

    model_runs: List[Dict[str, Any]] = []
    for model_config in model_configs:
        results = asyncio.run(run_cases_async(cases, model_config, args.concurrency))
        summary = summarize_case_results(results)
        payload = {
            "llm": model_config.to_dict(),
            "summary": summary,
            "results": results,
        }
        failure_analysis_dir = generate_failure_analysis(
            payload,
            output_root=Path(__file__).resolve().parent,
            run_timestamp=run_timestamp,
            model_label=model_config.label,
        )
        append_log(
            summary,
            dataset_path=dataset_path,
            llm_meta=model_config.to_dict(),
            log_paths=[DEFAULT_MASTER_LOG_PATH, _per_model_log_path(model_config.label)],
            failure_analysis_dir=failure_analysis_dir,
        )
        payload["failure_analysis_dir"] = str(failure_analysis_dir)
        model_runs.append(payload)

    write_run_overview(run_root=run_root, dataset_path=dataset_path, model_runs=model_runs)
    leaderboard = summarize_model_runs(model_runs)
    output_payload = {
        "run_timestamp": run_timestamp.isoformat(),
        "dataset": str(dataset_path),
        "case_count": len(cases),
        "llm_count": len(model_runs),
        "leaderboard": leaderboard,
        "master_log_path": str(DEFAULT_MASTER_LOG_PATH),
        "failure_analysis_root": str(run_root),
        "model_runs": model_runs,
    }

    if len(model_runs) == 1:
        single = model_runs[0]
        print(json.dumps(single["summary"], ensure_ascii=False, indent=2))
        print(f"llm={json.dumps(single['llm'], ensure_ascii=False)}")
        print(f"log_path={_per_model_log_path(single['llm']['label'])}")
        print(f"failure_analysis_dir={single['failure_analysis_dir']}")
    else:
        print(
            json.dumps(
                {
                    "case_count": output_payload["case_count"],
                    "llm_count": output_payload["llm_count"],
                    "leaderboard": leaderboard,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"master_log_path={DEFAULT_MASTER_LOG_PATH}")
        print(f"failure_analysis_root={run_root}")

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
