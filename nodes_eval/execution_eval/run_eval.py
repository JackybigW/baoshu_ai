from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes import consultants
from nodes_eval.common import (
    append_key_value_log,
    build_frontend_model_configs,
    dump_messages,
    load_json,
    messages_from_dicts,
)
from nodes_eval.execution_eval.benchmark import BackendRubricJudge, score_execution_output
from nodes_eval.execution_eval.failure_analysis import generate_failure_analysis
from state import CustomerProfile

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_MASTER_LOG_PATH = Path(__file__).resolve().parent / "eval_execution.log"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"

NODE_RUNNERS = {
    "first_greeting": consultants.first_greeting_node,
    "interviewer": consultants.interviewer_node,
    "consultant": consultants.consultant_node,
    "high_value": consultants.high_value_node,
    "low_budget": consultants.low_budget_node,
    "art": consultants.art_node,
    "human_handoff": consultants.human_handoff_node,
    "chit_chat": consultants.chit_chat_node,
}


class EvalInput(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    profile: CustomerProfile = Field(default_factory=CustomerProfile)
    last_intent: Optional[str] = Field(default=None)
    dialog_status: Optional[str] = Field(default=None)
    has_proposed_solution: bool = Field(default=False)


class EvalExpected(BaseModel):
    must_call_tool: Optional[bool] = Field(default=None)
    required_message_types: List[str] = Field(default_factory=list)
    min_segments: int = Field(default=1)
    max_segments: int = Field(default=3)
    max_chars_per_segment: Optional[int] = Field(default=None)
    required_keyword_groups: List[List[str]] = Field(default_factory=list)
    forbidden_keywords: List[str] = Field(default_factory=list)
    expected_status: Optional[str] = Field(default=None)
    skip_rubric: bool = Field(default=False)
    node_goal: str = Field(default="")
    rubric_notes: str = Field(default="")
    fatal_errors: List[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    case_id: str
    tags: List[str] = Field(default_factory=list)
    node_name: str
    input: EvalInput
    expected: EvalExpected


def load_cases(dataset_path: Path) -> List[EvalCase]:
    return [EvalCase.model_validate(item) for item in load_json(dataset_path)]


@contextmanager
def temporary_frontend_llm(llm: Any) -> Iterator[None]:
    original_llm_chat = consultants.llm_chat
    original_llm = consultants.llm
    consultants.llm_chat = llm
    consultants.llm = llm
    try:
        yield
    finally:
        consultants.llm_chat = original_llm_chat
        consultants.llm = original_llm


def _per_model_log_path(label: str) -> Path:
    return DEFAULT_LOG_DIR / f"execution_eval_{label}.log"


def run_single_case(case: EvalCase, *, judge: Optional[BackendRubricJudge]) -> Dict[str, Any]:
    runner = NODE_RUNNERS[case.node_name]
    state = {
        "messages": messages_from_dicts(case.input.messages),
        "profile": case.input.profile,
        "last_intent": case.input.last_intent,
        "dialog_status": case.input.dialog_status,
        "has_proposed_solution": case.input.has_proposed_solution,
    }

    error = None
    try:
        result = runner(state)
        output_messages = result.get("messages", [])
        actual_status = result.get("dialog_status", case.input.dialog_status)
    except Exception as exc:
        output_messages = []
        actual_status = None
        error = f"{type(exc).__name__}: {exc}"

    breakdown = score_execution_output(
        node_name=case.node_name,
        contract=case.expected.model_dump(mode="json", exclude_none=True),
        case_input=case.input.model_dump(mode="json", exclude_none=True),
        output_messages=output_messages,
        actual_status=actual_status,
        judge=judge,
    )
    return {
        "case_id": case.case_id,
        "node_name": case.node_name,
        "tags": case.tags,
        "input": case.input.model_dump(mode="json", exclude_none=True),
        "expected": case.expected.model_dump(mode="json", exclude_none=True),
        "actual": {
            "messages": dump_messages(output_messages),
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
            "tool_success_rate": 0.0,
            "format_score": 0.0,
            "keyword_score": 0.0,
            "rubric_score": 0.0,
            "pass_rate": 0.0,
            "error_count": 0,
            "lowest_cases": [],
        }

    overall_scores = [item["score"]["overall_score"] for item in results]
    tool_scores = [item["score"]["tool_score"] for item in results if item["score"]["tool_score"] is not None]
    format_scores = [item["score"]["format_score"] for item in results if item["score"]["format_score"] is not None]
    keyword_scores = [item["score"]["keyword_score"] for item in results if item["score"]["keyword_score"] is not None]
    rubric_scores = [item["score"]["rubric_score"] for item in results if item["score"]["rubric_score"] is not None]
    passed_cases = sum(1 for item in results if item["score"]["overall_score"] >= 80)
    error_count = sum(1 for item in results if item.get("error"))
    lowest_cases = sorted(results, key=lambda item: item["score"]["overall_score"])[:10]
    return {
        "case_count": len(results),
        "overall_score": round(mean(overall_scores), 2),
        "tool_success_rate": round(mean(tool_scores), 4) if tool_scores else 1.0,
        "format_score": round(mean(format_scores), 4) if format_scores else 1.0,
        "keyword_score": round(mean(keyword_scores), 4) if keyword_scores else 1.0,
        "rubric_score": round(mean(rubric_scores), 4) if rubric_scores else 1.0,
        "pass_rate": round(passed_cases / len(results), 4),
        "error_count": error_count,
        "lowest_cases": [
            {
                "case_id": item["case_id"],
                "node_name": item["node_name"],
                "overall_score": item["score"]["overall_score"],
                "failure_tags": item["score"].get("failure_tags"),
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
                "tool_success_rate": summary["tool_success_rate"],
                "format_score": summary["format_score"],
                "keyword_score": summary["keyword_score"],
                "rubric_score": summary["rubric_score"],
                "pass_rate": summary["pass_rate"],
                "error_count": summary["error_count"],
            }
        )
    leaderboard.sort(
        key=lambda item: (
            -item["overall_score"],
            -item["tool_success_rate"],
            -item["rubric_score"],
            -item["pass_rate"],
            item["error_count"],
            item["llm_label"],
        )
    )
    return leaderboard


def append_log(
    summary: Dict[str, Any],
    *,
    dataset_path: Path,
    llm_meta: Dict[str, Any],
    log_paths: List[Path],
    failure_analysis_dir: Path,
) -> None:
    append_key_value_log(
        title="execution_eval",
        kv_pairs=[
            ("dataset", str(dataset_path)),
            ("llm_label", llm_meta["label"]),
            ("llm_canonical", llm_meta["canonical_id"]),
            ("llm_provider", llm_meta["provider"]),
            ("llm_model", llm_meta["resolved_model"]),
            ("case_count", summary["case_count"]),
            ("overall_score", summary["overall_score"]),
            ("tool_success_rate", summary["tool_success_rate"]),
            ("format_score", summary["format_score"]),
            ("keyword_score", summary["keyword_score"]),
            ("rubric_score", summary["rubric_score"]),
            ("pass_rate", summary["pass_rate"]),
            ("error_count", summary["error_count"]),
            ("failure_analysis_dir", str(failure_analysis_dir)),
            ("lowest_cases", summary["lowest_cases"]),
        ],
        log_paths=log_paths,
    )


def write_run_overview(*, run_root: Path, dataset_path: Path, model_runs: List[Dict[str, Any]]) -> None:
    leaderboard = summarize_model_runs(model_runs)
    lines = [
        f"# Execution Eval Run {run_root.name}",
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
            f"score `{row['overall_score']}` | "
            f"tool `{row['tool_success_rate']}` | "
            f"rubric `{row['rubric_score']}` | "
            f"pass_rate `{row['pass_rate']}` | "
            f"errors `{row['error_count']}`"
        )
    (run_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (run_root / "leaderboard.json").write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution-node eval pipeline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset json.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases.")
    parser.add_argument("--case-id", default="", help="Only run one specific case id.")
    parser.add_argument("--tag", action="append", default=[], help="Only run cases that contain the given tag.")
    parser.add_argument(
        "--llm",
        action="append",
        default=[],
        help="Run execution nodes with a specific LLM alias. Repeatable. If omitted, uses frontend_default.",
    )
    parser.add_argument("--no-judge", action="store_true", help="Disable rubric judge and only use deterministic/fallback scoring.")
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

    model_configs = build_frontend_model_configs(args.llm, temperature=0.7)
    run_timestamp = datetime.now()
    run_root = Path(__file__).resolve().parent / "failure_analyses" / run_timestamp.strftime("%Y%m%d_%H%M%S")

    model_runs: List[Dict[str, Any]] = []
    for model_config in model_configs:
        judge = None if args.no_judge else BackendRubricJudge()
        with temporary_frontend_llm(model_config.llm):
            results = [run_single_case(case, judge=judge) for case in cases]

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
