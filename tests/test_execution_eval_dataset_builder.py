from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes_eval.execution_eval.build_dataset import build_execution_dataset


NODES = ["consultant", "interviewer", "high_value", "low_budget", "art", "chit_chat"]
SHARD_FILENAMES = [f"{node}_cases.json" for node in NODES]
REPO_DATASETS_DIR = Path(__file__).resolve().parents[1] / "nodes_eval/execution_eval/datasets"
REPO_GOLDEN_PATH = Path(__file__).resolve().parents[1] / "nodes_eval/execution_eval/golden_dataset.json"


def _write_shard(path: Path, node_name: str, case_ids: list[str]) -> None:
    payload = [
        {
            "case_id": case_id,
            "node_name": node_name,
            "tags": [node_name],
            "input": {"messages": []},
            "expected": {"node_goal": f"{node_name} goal"},
        }
        for case_id in case_ids
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_empty_shard(path: Path) -> None:
    path.write_text("[]\n", encoding="utf-8")


def test_build_execution_dataset_merges_scaffold_shards(tmp_path: Path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    for shard_filename in SHARD_FILENAMES:
        _write_empty_shard(datasets_dir / shard_filename)

    output_path = tmp_path / "golden_dataset.json"
    build_execution_dataset(datasets_dir=datasets_dir, output_path=output_path, strict=False)

    merged = json.loads(output_path.read_text(encoding="utf-8"))
    assert merged == []


def test_build_execution_dataset_enforces_minimums_in_strict_mode(tmp_path: Path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    for node_name in NODES:
        case_count = 19 if node_name == "consultant" else 20
        _write_shard(
            datasets_dir / f"{node_name}_cases.json",
            node_name,
            [f"{node_name[:3]}_{item:03d}" for item in range(case_count)],
        )

    with pytest.raises(ValueError, match="consultant.*19"):
        build_execution_dataset(datasets_dir=datasets_dir, output_path=tmp_path / "golden_dataset.json", strict=True)


def test_build_execution_dataset_rejects_duplicate_case_ids(tmp_path: Path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    for node_name in NODES:
        case_ids = [f"{node_name[:3]}_{item:03d}" for item in range(20)]
        if node_name == "interviewer":
            case_ids[0] = "dup_001"
        if node_name == "consultant":
            case_ids[0] = "dup_001"
        _write_shard(datasets_dir / f"{node_name}_cases.json", node_name, case_ids)

    with pytest.raises(ValueError, match="duplicate case_id"):
        build_execution_dataset(datasets_dir=datasets_dir, output_path=tmp_path / "golden_dataset.json", strict=False)


def test_build_execution_dataset_rejects_invalid_regex_patterns(tmp_path: Path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    for shard_filename, node_name in zip(SHARD_FILENAMES, NODES, strict=True):
        payload = [
            {
                "case_id": f"{node_name[:3]}_001",
                "node_name": node_name,
                "tags": [node_name],
                "input": {"messages": []},
                "expected": {"forbidden_regexes": ["["]},
            }
        ]
        (datasets_dir / shard_filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid forbidden_regexes"):
        build_execution_dataset(datasets_dir=datasets_dir, output_path=tmp_path / "golden_dataset.json", strict=False)


def test_build_execution_dataset_matches_checked_in_golden_dataset():
    output_path = REPO_GOLDEN_PATH.parent / "_tmp_built_execution_dataset.json"
    try:
        build_execution_dataset(datasets_dir=REPO_DATASETS_DIR, output_path=output_path, strict=True)
        committed = json.loads(REPO_GOLDEN_PATH.read_text(encoding="utf-8"))
        rebuilt = json.loads(output_path.read_text(encoding="utf-8"))
        assert rebuilt == committed
    finally:
        if output_path.exists():
            output_path.unlink()
