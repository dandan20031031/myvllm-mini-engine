"""Numerical gates catch hidden state errors even when greedy tokens agree."""

import pytest
import torch

from benchmarks.diagnose_ragged_prefill import stats, kv_comparison, compare_teacher_forced


def test_combined_budget_and_first_violation_are_explicit():
    reference = torch.tensor([0.0, 10.0, 0.0])
    result = stats(torch.tensor([0.1, 10.2, 0.126]), reference, 0.125, 0.01)
    assert result["mismatched"] == 1
    assert result["first_mismatch_index"] == [2]
    assert result["first_mismatch_budget"] == 0.125
    assert stats(torch.tensor([float('nan')]), torch.zeros(1), 1, 1)["mismatched"] == 1
    with pytest.raises(ValueError):
        stats(torch.ones(1), torch.ones(1), float('inf'), 0)


def test_kv_first_bad_layer_and_teacher_forced_step_reject_same_argmax():
    logits = torch.tensor([[0.0, 2.0, 1.0]])
    kv = torch.zeros(3, 1, 2, 4)
    bad = kv.clone()
    bad[2, 0, 1, 3] = 0.126
    assert kv_comparison(bad, kv, 0.125, 0.01)["first_failing_layer"] == 2
    ref = {"logits": logits, "kv": (kv, kv)}
    actual = {"logits": logits + 0.001, "kv": (bad, kv)}
    result = compare_teacher_forced([ref, actual], [ref, ref], 0.125, 0.01)
    assert result["first_argmax_difference"] is None
    assert result["first_failing_step"] == 1
    assert not result["all_steps_strict_passed"]
    assert result["steps"][1]["argmax"][0]["reference_top2_margin"] == 1.0
    with pytest.raises(ValueError):
        compare_teacher_forced([], [], 0.125, 0.01)


def test_teacher_forcing_consumes_reference_tokens_instead_of_candidate_argmax(monkeypatch):
    from types import SimpleNamespace
    import benchmarks.diagnose_ragged_prefill as entry
    calls = []
    logits = torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    def decode(tokens, cache, tables, positions):
        calls.append((tokens.tolist(), positions.tolist()))
        return logits
    cache = SimpleNamespace(shape=SimpleNamespace(block_size=4), device=torch.device("cpu"), clear=lambda pages: None)
    runner = SimpleNamespace(cache=cache, model=SimpleNamespace(forward_decode_batch=decode))
    monkeypatch.setattr(entry.bench, "execute_mode", lambda *args: logits)
    monkeypatch.setattr(entry.bench, "cache_snapshot", lambda *args: (torch.zeros(1, 2, 1, 1),) * 2)
    trace = entry.teacher_forced_trace(runner, {}, "unused", [[1, 2], [1, 2, 3]],
        [[0, 1], [2, 3]], torch.tensor([[0, 1], [2, 3]]), [[2, 3, 4], [4, 3, 2]])
    assert len(trace) == 3
    assert calls == [([2, 4], [2, 3]), ([3, 3], [3, 4])]


def test_benchmark_kernel_override_restores_on_failure(monkeypatch):
    import mini_engine.model.native_qwen3 as model_module
    from benchmarks.benchmark_decode_kernel import kernel_route
    def original(*args, **kwargs):
        return kwargs["backend"]
    monkeypatch.setattr(model_module, "paged_attention", original)
    with pytest.raises(RuntimeError, match="injected"):
        with kernel_route("cuda_vector128"):
            assert model_module.paged_attention(backend="cuda") == "cuda_vector128"
            assert model_module.paged_attention(backend="torch") == "torch"
            raise RuntimeError("injected")
    assert model_module.paged_attention is original
