from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes_eval.extractor_eval.benchmark import DeepSeekSemanticMatcher, score_profiles
from state import CustomerProfile

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "eval_extracot.log"


class EvalInput(BaseModel):
    last_ai_msg: str = Field(default="")
    last_user_msg: str
    current_profile: Optional[CustomerProfile] = Field(default=None)


class EvalCase(BaseModel):
    case_id: str
    tags: List[str] = Field(default_factory=list)
    input: EvalInput
    expected: CustomerProfile


def load_cases(dataset_path: Path) -> List[EvalCase]:
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in raw_cases]


def run_single_case(case: EvalCase, semantic_matcher: Optional[DeepSeekSemanticMatcher]) -> Dict[str, Any]:
    from nodes.perception import extractor_node

    current_profile = case.input.current_profile or CustomerProfile()
    messages = []
    if case.input.last_ai_msg.strip():
        messages.append(AIMessage(content=case.input.last_ai_msg))
    messages.append(HumanMessage(content=case.input.last_user_msg))

    state = {
        "messages": messages,
        "profile": current_profile,
    }
    error = None
    try:
        result = extractor_node(state)
        actual_profile = result["profile"]
    except Exception as exc:
        actual_profile = CustomerProfile()
        error = f"{type(exc).__name__}: {exc}"

    breakdown = score_profiles(case.expected, actual_profile, semantic_matcher=semantic_matcher)
    return {
        "case_id": case.case_id,
        "tags": case.tags,
        "input": case.input.model_dump(mode="json", exclude_none=True),
        "expected": case.expected.model_dump(mode="json", exclude_none=False),
        "actual": actual_profile.model_dump(mode="json", exclude_none=False),
        "error": error,
        "score": breakdown.to_dict(),
    }


def summarize_case_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "case_count": 0,
            "overall_score": 0.0,
            "exact_recall": 0.0,
            "hallucination_rate": 0.0,
            "fuzzy_semantic": 0.0,
            "passed_cases": 0,
            "pass_rate": 0.0,
            "error_count": 0,
            "lowest_cases": [],
        }

    overall_scores = [item["score"]["overall_score"] for item in results]
    exact_scores = [item["score"]["exact_recall"] for item in results]
    hallucination_rates = [item["score"]["hallucination_rate"] for item in results]
    fuzzy_scores = [item["score"]["fuzzy_semantic"] for item in results]
    passed_cases = sum(1 for item in results if item["score"]["overall_score"] >= 75)
    error_count = sum(1 for item in results if item.get("error"))
    lowest_cases = sorted(results, key=lambda item: item["score"]["overall_score"])[:10]

    return {
        "case_count": len(results),
        "overall_score": round(mean(overall_scores), 2),
        "exact_recall": round(mean(exact_scores), 4),
        "hallucination_rate": round(mean(hallucination_rates), 4),
        "fuzzy_semantic": round(mean(fuzzy_scores), 4),
        "passed_cases": passed_cases,
        "pass_rate": round(passed_cases / len(results), 4),
        "error_count": error_count,
        "lowest_cases": [
            {
                "case_id": item["case_id"],
                "overall_score": item["score"]["overall_score"],
                "tags": item["tags"],
                "error": item.get("error"),
            }
            for item in lowest_cases
        ],
    }


def append_log(summary: Dict[str, Any], dataset_path: Path, log_path: Path) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[{timestamp}] extractor_eval",
        f"dataset={dataset_path}",
        f"case_count={summary['case_count']}",
        f"overall_score={summary['overall_score']}",
        f"exact_recall={summary['exact_recall']}",
        f"hallucination_rate={summary['hallucination_rate']}",
        f"fuzzy_semantic={summary['fuzzy_semantic']}",
        f"pass_rate={summary['pass_rate']}",
        f"error_count={summary['error_count']}",
        f"lowest_cases={json.dumps(summary['lowest_cases'], ensure_ascii=False)}",
        "",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extractor eval pipeline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset json.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases.")
    parser.add_argument("--case-id", default="", help="Only run one specific case id.")
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable DeepSeek semantic judge and fallback to token overlap only.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write full result json.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max number of eval cases to run concurrently.",
    )
    return parser.parse_args()


def ensure_backend_llm_ready() -> None:
    from utils.llm_factory import get_backend_llm

    if get_backend_llm() is None:
        raise SystemExit(
            "Extractor eval requires a backend LLM. Please configure DEEPSEEK_API_KEY "
            "or GOOGLE_API_KEY in the agent conda environment before running."
        )


async def run_cases_async(
    cases: List[EvalCase],
    semantic_matcher: Optional[DeepSeekSemanticMatcher],
    concurrency: int,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_with_limit(case: EvalCase) -> Dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(run_single_case, case, semantic_matcher)

    tasks = [run_with_limit(case) for case in cases]
    return await asyncio.gather(*tasks)


def main() -> None:
    args = parse_args()
    ensure_backend_llm_ready()
    dataset_path = Path(args.dataset).resolve()
    cases = load_cases(dataset_path)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.limit > 0:
        cases = cases[: args.limit]

    semantic_matcher = None if args.no_semantic else DeepSeekSemanticMatcher()
    if args.no_semantic and semantic_matcher is None:
        semantic_matcher = None

    results = asyncio.run(run_cases_async(cases, semantic_matcher, args.concurrency))
    summary = summarize_case_results(results)
    append_log(summary, dataset_path=dataset_path, log_path=DEFAULT_LOG_PATH)

    payload = {
        "summary": summary,
        "results": results,
        "log_path": str(DEFAULT_LOG_PATH),
    }
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"log_path={DEFAULT_LOG_PATH}")

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
