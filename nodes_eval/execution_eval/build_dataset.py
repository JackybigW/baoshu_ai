from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


SHARD_REGISTRY: Mapping[str, str] = {
    "consultant": "consultant.json",
    "interviewer": "interviewer.json",
    "high_value": "high_value.json",
    "low_budget": "low_budget.json",
    "art": "art.json",
    "chit_chat": "chit_chat.json",
}

MIN_CASES_PER_NODE = 20
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_case(case: Any, *, shard_name: str, shard_path: Path) -> Dict[str, Any]:
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

    return case


def load_shard_cases(*, node_name: str, datasets_dir: Path) -> List[Dict[str, Any]]:
    shard_filename = SHARD_REGISTRY[node_name]
    shard_path = datasets_dir / shard_filename
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing execution eval shard: {shard_path}")

    payload = _load_json(shard_path)
    if not isinstance(payload, list):
        raise ValueError(f"{shard_path}: shard must be a JSON array")

    cases = [_validate_case(case, shard_name=node_name, shard_path=shard_path) for case in payload]
    if len(cases) < MIN_CASES_PER_NODE:
        raise ValueError(
            f"{shard_path}: expected at least {MIN_CASES_PER_NODE} cases, found {len(cases)}"
        )
    return cases


def merge_shards(*, datasets_dir: Path | str = DEFAULT_DATASETS_DIR) -> List[Dict[str, Any]]:
    base_dir = Path(datasets_dir)
    merged: List[Dict[str, Any]] = []
    seen_case_ids: set[str] = set()

    for node_name in SHARD_REGISTRY:
        for case in load_shard_cases(node_name=node_name, datasets_dir=base_dir):
            case_id = case["case_id"]
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate case_id detected: {case_id}")
            seen_case_ids.add(case_id)
            merged.append(case)

    merged.sort(key=lambda item: item["case_id"])
    return merged


def build_dataset(
    *,
    datasets_dir: Path | str = DEFAULT_DATASETS_DIR,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    merged = merge_shards(datasets_dir=datasets_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Merge execution-eval dataset shards.")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    output_path = build_dataset(datasets_dir=args.datasets_dir, output_path=args.output)
    print(output_path)


if __name__ == "__main__":
    main()
