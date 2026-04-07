from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SHARD_REGISTRY: Mapping[str, str] = {
    "consultant": "consultant_cases.json",
    "interviewer": "interviewer_cases.json",
    "high_value": "high_value_cases.json",
    "low_budget": "low_budget_cases.json",
    "art": "art_cases.json",
    "chit_chat": "chit_chat_cases.json",
}

MIN_CASES_PER_NODE = 20
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_case(case: Any, *, shard_name: str, shard_path: Path) -> Dict[str, Any]:
    from nodes_eval.execution_eval.run_eval import EvalCase

    if not isinstance(case, dict):
        raise ValueError(f"{shard_path}: each case must be a JSON object")

    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{shard_path}: case_id must be a non-empty string")

    node_name = case.get("node_name")
    if node_name != shard_name:
        raise ValueError(
            f"{shard_path}: case_id {case_id} declares node_name={node_name!r}, expected {shard_name!r}"
        )

    expected = case.get("expected") or {}
    for pattern in expected.get("forbidden_regexes") or []:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"{shard_path}: case_id {case_id} has invalid forbidden_regexes pattern {pattern!r}: {exc}"
            ) from exc

    try:
        EvalCase.model_validate(case)
    except Exception as exc:
        raise ValueError(f"{shard_path}: case_id {case_id} failed schema validation: {exc}") from exc

    return case


def load_shard_cases(*, node_name: str, datasets_dir: Path, strict: bool) -> List[Dict[str, Any]]:
    shard_filename = SHARD_REGISTRY[node_name]
    shard_path = datasets_dir / shard_filename
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing execution eval shard: {shard_path}")

    payload = _load_json(shard_path)
    if not isinstance(payload, list):
        raise ValueError(f"{shard_path}: shard must be a JSON array")

    cases = [_validate_case(case, shard_name=node_name, shard_path=shard_path) for case in payload]
    if strict and len(cases) < MIN_CASES_PER_NODE:
        raise ValueError(
            f"{shard_path}: expected at least {MIN_CASES_PER_NODE} cases, found {len(cases)}"
        )
    return cases


def merge_shards(*, datasets_dir: Path | str = DEFAULT_DATASETS_DIR, strict: bool = False) -> List[Dict[str, Any]]:
    base_dir = Path(datasets_dir)
    merged: List[Dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for node_name in SHARD_REGISTRY:
        for case in load_shard_cases(node_name=node_name, datasets_dir=base_dir, strict=strict):
            case_id = case["case_id"]
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id detected: {case_id}")
            seen_case_ids.add(case_id)
            merged.append(case)

    merged.sort(key=lambda item: item["case_id"])
    return merged


def build_execution_dataset(
    *,
    datasets_dir: Path | str = DEFAULT_DATASETS_DIR,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    strict: bool = True,
) -> Path:
    merged = merge_shards(datasets_dir=datasets_dir, strict=strict)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def build_dataset(
    *,
    datasets_dir: Path | str = DEFAULT_DATASETS_DIR,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    strict: bool = True,
) -> Path:
    return build_execution_dataset(datasets_dir=datasets_dir, output_path=output_path, strict=strict)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Merge execution-eval dataset shards.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="accepted for backward compatibility; strict mode is already the default",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow shard counts below the 20-case minimum and still write output",
    )
    args = parser.parse_args()
    output_path = build_execution_dataset(
        datasets_dir=args.datasets_dir,
        output_path=args.output,
        strict=not args.allow_partial,
    )
    print(output_path)


if __name__ == "__main__":
    main()
