from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
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
from nodes_eval.extractor_eval.failure_analysis import generate_failure_analysis
from state import CustomerProfile
from utils.llm_factory import get_backend_llm, get_llm, get_llm_descriptor, list_supported_llms, resolve_llm_key

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
DEFAULT_MASTER_LOG_PATH = Path(__file__).resolve().parent / "eval_extracot.log"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"


class EvalInput(BaseModel):
    last_ai_msg: str = Field(default="")
    last_user_msg: str
    current_profile: Optional[CustomerProfile] = Field(default=None)


class EvalCase(BaseModel):
    case_id: str
    tags: List[str] = Field(default_factory=list)
    input: EvalInput
    expected: CustomerProfile


@dataclass(frozen=True)
class EvalModelConfig:
    requested_id: str
    canonical_id: str
    label: str
    provider: str
    resolved_model: str
    llm: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_id": self.requested_id,
            "canonical_id": self.canonical_id,
            "label": self.label,
            "provider": self.provider,
            "resolved_model": self.resolved_model,
        }


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    text = text.strip("._").lower()
    return text or "unknown_llm"


def load_cases(dataset_path: Path) -> List[EvalCase]:
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(item) for item in raw_cases]


def _resolved_chain_model_names(model_ids: List[str]) -> str:
    parts: List[str] = []
    for model_id in model_ids:
        descriptor = get_llm_descriptor(model_id)
        parts.append(descriptor["resolved_model"])
    return " -> ".join(parts)


def build_model_configs(model_ids: List[str]) -> List[EvalModelConfig]:
    requested_models = model_ids or ["backend_default"]
    configs: List[EvalModelConfig] = []
    seen_labels: set[str] = set()

    for requested_id in requested_models:
        normalized = requested_id.strip()
        lowered = normalized.lower().replace("-", "_")

        if lowered in {"backend", "default", "backend_default"}:
            llm = get_backend_llm()
            if llm is None:
                raise SystemExit(
                    "Extractor eval requires a backend LLM. Please configure the backend chain "
                    "(e.g. DEEPSEEK / GOOGLE / DOUBAO related env vars) before running."
                )
            label = "backend_default"
            if label in seen_labels:
                continue
            seen_labels.add(label)
            configs.append(
                EvalModelConfig(
                    requested_id=normalized,
                    canonical_id="backend_default",
                    label=label,
                    provider="fallback_chain",
                    resolved_model=_resolved_chain_model_names(["deepseek", "gemini_flash", "doubao"]),
                    llm=llm,
                )
            )
            continue

        try:
            descriptor = get_llm_descriptor(normalized)
        except ValueError as exc:
            raise SystemExit(
                f"{exc}. 当前支持: {', '.join(list_supported_llms())}。"
            ) from exc
        llm = get_llm(normalized, allow_missing=True)
        if llm is None:
            canonical = descriptor["canonical_id"]
            raise SystemExit(
                f"LLM `{normalized}` 无法初始化。请检查相关环境变量是否完整。"
                f" 当前支持: {', '.join(list_supported_llms())}。"
                f" 需要的 key/base_url 可参考 `{canonical}` 对应配置。"
            )

        label = _slugify(descriptor["canonical_id"])
        if label in seen_labels:
            continue
        seen_labels.add(label)
        configs.append(
            EvalModelConfig(
                requested_id=normalized,
                canonical_id=resolve_llm_key(normalized),
                label=label,
                provider=descriptor["provider"],
                resolved_model=descriptor["resolved_model"],
                llm=llm,
            )
        )

    return configs


def run_single_case(
    case: EvalCase,
    model_config: EvalModelConfig,
    semantic_matcher: Optional[DeepSeekSemanticMatcher],
) -> Dict[str, Any]:
    from nodes.perception import extractor_node

    current_profile = case.input.current_profile or CustomerProfile()
    messages = []
    if case.input.last_ai_msg.strip():
        messages.append(AIMessage(content=case.input.last_ai_msg))
    messages.append(HumanMessage(content=case.input.last_user_msg))

    state = {
        "messages": messages,
        "profile": current_profile,
        "runtime_config": {
            "backend_llm": model_config.llm,
            "backend_model": model_config.canonical_id,
            "eval_llm_label": model_config.label,
        },
    }

    error = None
    try:
        result = extractor_node(state)
        actual_profile = result["profile"]
    except Exception as exc:
        actual_profile = CustomerProfile()
        error = f"{type(exc).__name__}: {exc}"

    breakdown = score_profiles(
        case.expected,
        actual_profile,
        semantic_matcher=semantic_matcher,
        case_context=case.input.model_dump(mode="json", exclude_none=True),
    )
    return {
        "llm": model_config.to_dict(),
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
                "exact_recall": summary["exact_recall"],
                "hallucination_rate": summary["hallucination_rate"],
                "fuzzy_semantic": summary["fuzzy_semantic"],
                "pass_rate": summary["pass_rate"],
                "error_count": summary["error_count"],
            }
        )
    leaderboard.sort(
        key=lambda item: (
            -item["overall_score"],
            -item["pass_rate"],
            item["hallucination_rate"],
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[{timestamp}] extractor_eval",
        f"dataset={dataset_path}",
        f"llm_label={llm_meta['label']}",
        f"llm_canonical={llm_meta['canonical_id']}",
        f"llm_provider={llm_meta['provider']}",
        f"llm_model={llm_meta['resolved_model']}",
        f"case_count={summary['case_count']}",
        f"overall_score={summary['overall_score']}",
        f"exact_recall={summary['exact_recall']}",
        f"hallucination_rate={summary['hallucination_rate']}",
        f"fuzzy_semantic={summary['fuzzy_semantic']}",
        f"pass_rate={summary['pass_rate']}",
        f"error_count={summary['error_count']}",
        f"failure_analysis_dir={failure_analysis_dir}",
        f"lowest_cases={json.dumps(summary['lowest_cases'], ensure_ascii=False)}",
        "",
    ]
    for log_path in log_paths:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extractor eval pipeline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset json.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N cases.")
    parser.add_argument("--case-id", default="", help="Only run one specific case id.")
    parser.add_argument("--tag", action="append", default=[], help="Only run cases that contain the given tag. Repeatable.")
    parser.add_argument(
        "--llm",
        action="append",
        default=[],
        help=(
            "Run extractor with a specific LLM alias. Repeatable. "
            "Examples: --llm deepseek --llm qwen --llm glm --llm doubao --llm gemini_flash. "
            "If omitted, uses backend_default."
        ),
    )
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
        help="Max number of case executions to run concurrently within one requested LLM.",
    )
    return parser.parse_args()


async def run_cases_async(
    cases: List[EvalCase],
    model_config: EvalModelConfig,
    semantic_matcher: Optional[DeepSeekSemanticMatcher],
    concurrency: int,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_with_limit(case: EvalCase) -> Dict[str, Any]:
        async with semaphore:
            return await asyncio.to_thread(run_single_case, case, model_config, semantic_matcher)

    results = await asyncio.gather(*(run_with_limit(case) for case in cases))
    results.sort(key=lambda item: item["case_id"])
    return results


def _per_model_log_path(label: str) -> Path:
    return DEFAULT_LOG_DIR / f"extractor_eval_{label}.log"


def write_run_overview(
    *,
    run_root: Path,
    dataset_path: Path,
    model_runs: List[Dict[str, Any]],
) -> None:
    leaderboard = summarize_model_runs(model_runs)
    lines = [
        f"# Extractor Eval Run {run_root.name}",
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
            f"pass_rate `{row['pass_rate']}` | "
            f"hallucination `{row['hallucination_rate']}` | "
            f"errors `{row['error_count']}`"
        )
    lines.extend(
        [
            "",
            "## Model Dirs",
            "",
        ]
    )
    for item in model_runs:
        lines.append(
            f"- `{item['llm']['label']}`: `{item['failure_analysis_dir']}`"
        )
    (run_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (run_root / "leaderboard.json").write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    cases = load_cases(dataset_path)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.tag:
        tag_filter = set(args.tag)
        cases = [case for case in cases if tag_filter.intersection(case.tags)]
    if args.limit > 0:
        cases = cases[: args.limit]

    model_configs = build_model_configs(args.llm)
    semantic_matcher = None if args.no_semantic else DeepSeekSemanticMatcher()
    run_timestamp = datetime.now()
    run_root = Path(__file__).resolve().parent / "failure_analyses" / run_timestamp.strftime("%Y%m%d_%H%M%S")

    model_runs: List[Dict[str, Any]] = []
    for model_config in model_configs:
        results = asyncio.run(
            run_cases_async(cases, model_config, semantic_matcher, args.concurrency)
        )
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
        log_paths = [DEFAULT_MASTER_LOG_PATH, _per_model_log_path(model_config.label)]
        append_log(
            summary,
            dataset_path=dataset_path,
            llm_meta=model_config.to_dict(),
            log_paths=log_paths,
            failure_analysis_dir=failure_analysis_dir,
        )

        payload["failure_analysis_dir"] = str(failure_analysis_dir)
        payload["log_paths"] = {
            "master": str(DEFAULT_MASTER_LOG_PATH),
            "per_model": str(_per_model_log_path(model_config.label)),
        }
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
        print(f"log_path={single['log_paths']['per_model']}")
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
