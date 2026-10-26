"""Matched-input component comparison guards; no Triton dependency or GPU."""

import copy
import subprocess
import sys

import torch

from benchmarks import compare_paged_backends as entry


def test_dataset_is_deterministic_without_changing_global_rng_and_randomizes_pages():
    state = torch.random.get_rng_state().clone()
    first, a = entry.make_case(3, 11, heads=4, kv_heads=2, head_dim=16, block_size=4)
    entry.make_case(1, 3, heads=4, kv_heads=2, head_dim=16, block_size=4)
    second, b = entry.make_case(3, 11, heads=4, kv_heads=2, head_dim=16, block_size=4)
    assert a == b
    assert torch.equal(state, torch.random.get_rng_state())
    for name in entry.TENSOR_NAMES:
        assert torch.equal(first[name], second[name])
        assert len(a["tensors"][name]["sha256"]) == 64
    table = first["block_tables"]
    assert table.unique().numel() == table.numel()
    assert not torch.equal(table.flatten(), torch.arange(table.numel()))
    _, different = entry.make_case(3, 11, heads=4, kv_heads=2, head_dim=16, block_size=4, seed=4)
    assert a["case_sha256"] != different["case_sha256"]


def test_gqa_oracle_matches_independently_gathered_dense_attention_and_rejects_bad_outputs():
    tensors, _ = entry.make_case(2, 7, heads=4, kv_heads=2, head_dim=16, block_size=4)
    oracle = entry.fp32_oracle(tensors, 4)
    for row in range(2):
        ids = tensors["block_tables"][row]
        keys = tensors["key_cache"][ids].reshape(-1, 2, 16)[:7].repeat_interleave(2, 1)
        values = tensors["value_cache"][ids].reshape(-1, 2, 16)[:7].repeat_interleave(2, 1)
        q = tensors["query"][row].float().unsqueeze(1)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q, keys.transpose(0, 1).float(), values.transpose(0, 1).float()).squeeze(1)
        torch.testing.assert_close(oracle[row], expected)
    assert entry.validate_output(oracle.half(), oracle)["passed"]
    assert not entry.validate_output(oracle + 1, oracle)["passed"]
    poisoned = oracle.clone()
    poisoned[0, 0, 0] = float("nan")
    assert not entry.validate_output(poisoned, oracle)["passed"]


def successful_report():
    return {"status": "passed", "source_unchanged": True,
            "cases": [{"input": {"case_sha256": "identical"}, "correctness": {"passed": True}}],
            "measurement": {"calls": 50}, "environment": {"torch": "2.9"},
            "gpu_before": {"gpus": [{"uuid": "gpu1"}]},
            "gpu_after": {"gpus": [{"uuid": "gpu1"}]}}


def test_comparison_requires_actual_inputs_correctness_source_and_gpu_to_match():
    a, b = successful_report(), successful_report()
    b["environment"]["torch"] = "2.11"
    guard = entry.comparison_guard([a, b])
    assert guard["component_comparison_allowed"] and not guard["same_environment"]
    assert not guard["engine_throughput_comparison_allowed"]
    for field, value in (("case_sha256", "different"), ("passed", False)):
        broken = copy.deepcopy(b)
        location = broken["cases"][0]["input" if field == "case_sha256" else "correctness"]
        location[field] = value
        assert not entry.comparison_guard([a, broken])["component_comparison_allowed"]
    for key, value in (("status", "failed"), ("source_unchanged", False), ("gpu_after", {"gpus": []})):
        broken = copy.deepcopy(b)
        broken[key] = value
        assert not entry.comparison_guard([a, broken])["component_comparison_allowed"]


def test_help_requires_no_torch_triton_or_mini_import():
    code = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'myvllm', 'mini_engine'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['compare_paged_backends', '--help']
runpy.run_module('benchmarks.compare_paged_backends', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--mini-source" in result.stdout
