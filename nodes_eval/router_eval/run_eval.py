from __future__ import annotations

import argparse
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

from nodes_eval.common import append_key_value_log, load_json
from nodes_eval.router_eval.benchmark import score_router_result
from nodes_eval.router_eval.failure_analysis import generate_failure_analysis
from router import core_router
from state import CustomerProfile

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_MASTER_LOG_PATH = Path(__file__).resolve().parent / "eval_router.log"


class EvalInput(BaseModel):
    last_intent: Optional[str] = Field(default=None)
    dialog_status: Optional[str] = Field(default=None)
    profile: CustomerProfile = Field(default_factory=CustomerProfile)


class EvalExpected(BaseModel):
    route: str


class EvalCase(BaseModel):
    case_id: str
    tags: List[str] = Field(default_factory=list)
    input: EvalInput
    expected: EvalExpected


def load_cases(dataset_path: Path) -> List[EvalCase]:
    return [EvalCase.model_validate(item) for item in load_json(dataset_path)]


def run_single_case(case: EvalCase) -> Dict[str, Any]:
    state = {
        "last_intent": case.input.last_intent,
        "dialog_status": case.input.dialog_status,
        "profile": case.input.profile,
    }
    route = core_router(state)
    breakdown = score_router_result(case.expected.route, route)
    return {
        "case_id": case.case_id,
        "tags": case.tags,
        "input": case.input.model_dump(mode="json", exclude_none=True),
        "expected": case.expected.model_dump(mode="json", exclude_none=True),
        "actual": {"route": route},
        "score": breakdown.to_dict(),
    }


def summarize_case_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "case_count": 0,
            "overall_score": 0.0,
            "route_accuracy": 0.0,
            "pass_rate": 0.0,
            "lowest_cases": [],
        }

    overall_scores = [item["score"]["overall_score"] for item in results]
    route_scores = [item["score"]["route_score"] for item in results]
    passed_cases = sum(1 for item in results if item["score"]["overall_score"] == 100)
    lowest_cases = sorted(results, key=lambda item: item["score"]["overall_score"])[:10]
    return {
        "case_count": len(results),
        "overall_score": round(mean(overall_scores), 2),
        "route_accuracy": round(mean(route_scores), 4),
        "pass_rate": round(passed_cases / len(results), 4),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run router eval pipeline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset json.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases.")
    parser.add_argument("--case-id", default="", help="Only run one specific case id.")
    parser.add_argument("--tag", action="append", default=[], help="Only run cases that contain the given tag.")
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

    results = [run_single_case(case) for case in cases]
    summary = summarize_case_results(results)
    run_timestamp = datetime.now()
    payload = {
        "summary": summary,
        "results": results,
    }
    failure_analysis_dir = generate_failure_analysis(
        payload,
        output_root=Path(__file__).resolve().parent,
        run_timestamp=run_timestamp,
    )
    append_key_value_log(
        title="router_eval",
        kv_pairs=[
            ("dataset", str(dataset_path)),
            ("case_count", summary["case_count"]),
            ("overall_score", summary["overall_score"]),
            ("route_accuracy", summary["route_accuracy"]),
            ("pass_rate", summary["pass_rate"]),
            ("failure_analysis_dir", str(failure_analysis_dir)),
            ("lowest_cases", summary["lowest_cases"]),
        ],
        log_paths=[DEFAULT_MASTER_LOG_PATH],
    )

    output_payload = {
        "run_timestamp": run_timestamp.isoformat(),
        "dataset": str(dataset_path),
        "summary": summary,
        "failure_analysis_dir": str(failure_analysis_dir),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"log_path={DEFAULT_MASTER_LOG_PATH}")
    print(f"failure_analysis_dir={failure_analysis_dir}")

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
