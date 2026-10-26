"""Shared metadata preserves evidence scope and the standalone HTTP client."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import common


def test_canonical_hash_preserves_existing_unicode_manifest_format():
    expected = hashlib.sha256('{"max_tokens":128,"prompt":"你好"}'.encode("utf-8")).hexdigest()
    assert common.sha256_json({"prompt": "你好", "max_tokens": 128}) == expected
    assert common.sha256_json({"max_tokens": 128, "prompt": "你好"}) == expected


def test_fingerprints_include_shared_helpers_and_each_executed_entrypoint(monkeypatch):
    from benchmarks import benchmark_ragged_prefill, diagnose_ragged_prefill

    monkeypatch.setattr(common, "run_metadata_command", lambda command: {
        "available": True, "stdout": "commit" if "rev-parse" in command else "", "error": None})
    chat_path = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_chat_graph.py"
    chat = common.source_metadata(extra_files=(chat_path,))
    ragged = benchmark_ragged_prefill.source_fingerprint()
    diagnostic = diagnose_ragged_prefill.source_fingerprint()
    for report, expected in (
        (chat, {"benchmarks/benchmark_chat_graph.py"}),
        (ragged, {"benchmarks/benchmark_ragged_prefill.py"}),
        (diagnostic, {"benchmarks/diagnose_ragged_prefill.py", "benchmarks/benchmark_ragged_prefill.py"}),
    ):
        files = report["source_files_sha256"]
        assert {"benchmarks/common.py", "mini_engine/kernels/csrc/prefill.cu", *expected} <= files.keys()
        assert report["source_tree_sha256"] == common.sha256_json(files)
        assert not report["git_dirty"]


@pytest.mark.parametrize("entry_mode", ["direct", "module"])
def test_http_help_runs_without_torch_or_engine_from_another_directory(tmp_path, entry_mode):
    root = Path(__file__).resolve().parents[1]
    target = str(root / "benchmarks/benchmark_http.py") if entry_mode == "direct" else "benchmarks.benchmark_http"
    guard = """
import importlib.abc
import runpy
import sys

class RejectComputeDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mini_engine'}:
            raise AssertionError('HTTP client imported compute dependency: ' + fullname)

sys.meta_path.insert(0, RejectComputeDependencies())
target, mode = sys.argv[1:]
sys.argv = [target, '--help']
try:
    if mode == 'direct':
        runpy.run_path(target, run_name='__main__')
    else:
        runpy.run_module(target, run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not ({'torch', 'mini_engine'} & sys.modules.keys())
"""
    result = subprocess.run([sys.executable, "-B", "-c", guard, target, entry_mode], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(root)}, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=15)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "--base-url" in result.stdout
