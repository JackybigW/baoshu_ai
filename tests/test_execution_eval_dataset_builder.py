from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes_eval.execution_eval.build_dataset import build_dataset


NODES = ["consultant", "interviewer", "high_value", "low_budget", "art", "chit_chat"]


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


def test_build_dataset_merges_shards_and_sorts_by_case_id(tmp_path: Path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    for index, node_name in enumerate(NODES):
        case_ids = [f"{node_name[:3]}_{20 - item:03d}" for item in range(20)]
        _write_shard(datasets_dir / f"{node_name}.json", node_name, case_ids)

    output_path = tmp_path / "golden_dataset.json"
    build_dataset(datasets_dir=datasets_dir, output_path=output_path)

    merged = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(merged) == 120
    assert [item["case_id"] for item in merged] == sorted(item["case_id"] for item in merged)


def test_build_dataset_rejects_shards_with_too_few_cases(tmp_path: Path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    for node_name in NODES:
        case_count = 19 if node_name == "consultant" else 20
        _write_shard(
            datasets_dir / f"{node_name}.json",
            node_name,
            [f"{node_name[:3]}_{item:03d}" for item in range(case_count)],
        )

    with pytest.raises(ValueError, match="consultant.*20"):
        build_dataset(datasets_dir=datasets_dir, output_path=tmp_path / "golden_dataset.json")


def test_build_dataset_rejects_duplicate_case_ids(tmp_path: Path):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    for node_name in NODES:
        case_ids = [f"{node_name[:3]}_{item:03d}" for item in range(20)]
        if node_name == "interviewer":
            case_ids[0] = "dup_001"
        if node_name == "consultant":
            case_ids[0] = "dup_001"
        _write_shard(datasets_dir / f"{node_name}.json", node_name, case_ids)

    with pytest.raises(ValueError, match="duplicate case_id"):
        build_dataset(datasets_dir=datasets_dir, output_path=tmp_path / "golden_dataset.json")

