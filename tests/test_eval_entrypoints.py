import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_path: str) -> str:
    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_classifier_eval_entrypoint_exposes_cli_help():
    output = _run_help("nodes_eval/classifier_eval/run_eval.py")
    assert "Run classifier eval pipeline." in output


def test_router_eval_entrypoint_exposes_cli_help():
    output = _run_help("nodes_eval/router_eval/run_eval.py")
    assert "Run router eval pipeline." in output


def test_execution_eval_entrypoint_exposes_cli_help():
    output = _run_help("nodes_eval/execution_eval/run_eval.py")
    assert "Run execution-node eval pipeline." in output
