"""Packed execution preserves chunk, Prefix, cancellation and ownership semantics."""

import pytest
import torch

from mini_engine.engine.request import GenerationRequest, RequestState
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config


def make_runner():
    torch.manual_seed(431)
    model = NativeQwen3(Qwen3Config(71, 32, 64, 2, 4, 2, 8,
                                  max_position_embeddings=128), "math").eval()
    return NativeQwen3Runner(model, None, num_blocks=40, block_size=4, eos_token_ids=[])


def cold_logits(model, tokens):
    cache = model.create_cache(40, 4)
    table = list(range((len(tokens) + 3) // 4))
    return model(torch.tensor(tokens), cache, table)


def finish(runtime):
    for _ in range(100):
        if not runtime.scheduler.waiting and not runtime.scheduler.running:
            return
        runtime.step()
    raise AssertionError("runtime made no progress")


def test_multiple_requests_share_projections_and_select_only_needed_logits():
    runner = make_runner()
    runtime = runner.create_batch_runtime(enable_prefix_cache=False)
    prompts = [[1, 2, 3, 4, 5, 6], [11, 12, 13], [21, 22, 23, 24, 25]]
    expected = [int(cold_logits(runner.model, ids)[-1].argmax()) for ids in prompts]
    requests = [GenerationRequest(str(i), ids, max_new_tokens=1) for i, ids in enumerate(prompts)]
    projection_rows, head_rows, batches = [], [], []
    handles = [
        runner.model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
            lambda module, args: projection_rows.append(args[0].shape[0])),
        runner.model.lm_head.register_forward_pre_hook(
            lambda module, args: head_rows.append(args[0].shape[0])),
        runner.model.register_forward_pre_hook(
            lambda module, args, kwargs: batches.append(kwargs.get("prefill_batch")), with_kwargs=True),
    ]
    try:
        for request in requests:
            runtime.scheduler.add_request(request)
        output = runtime.step()
    finally:
        for handle in handles:
            handle.remove()
        runtime.close()
    assert list(output.values()) == expected
    assert projection_rows == [14] and head_rows == [3]
    assert len(batches) == 1 and batches[0].query_lengths == (6, 3, 5)
    assert all(request.is_finished for request in requests)
    assert runner.block_manager.used_blocks == 0


def test_exact_prefix_and_ragged_suffix_share_batch_without_overwriting_prefix():
    runner = make_runner()
    runtime = runner.create_batch_runtime()
    prefix = list(range(1, 9))
    warm = GenerationRequest("warm", prefix, max_new_tokens=1)
    runtime.scheduler.add_request(warm)
    finish(runtime)
    entry = runtime.prefix_cache.lookup(prefix)
    saved_keys = runtime.cache.key[:, list(entry.block_ids)].clone()
    saved_values = runtime.cache.value[:, list(entry.block_ids)].clone()
    suffix = prefix + [9, 10, 11, 12, 13]
    other = [21, 22, 23, 24, 25, 26]
    expected = [int(cold_logits(runner.model, ids)[-1].argmax()) for ids in (prefix, suffix, other)]
    boundary = cold_logits(runner.model, suffix[:12])[-1:]
    requests = [GenerationRequest(name, ids, max_new_tokens=1)
                for name, ids in zip(("hit", "suffix", "other"), (prefix, suffix, other))]
    batches = []
    handle = runner.model.register_forward_pre_hook(
        lambda module, args, kwargs: batches.append(kwargs["prefill_batch"]), with_kwargs=True)
    try:
        for request in requests:
            runtime.scheduler.add_request(request)
        output = runtime.step()
    finally:
        handle.remove()
    assert [output[request.request_id] for request in requests] == expected
    assert len(batches) == 1 and batches[0].query_lengths == (5, 6)
    assert batches[0].starts == (8, 0)
    torch.testing.assert_close(runtime.cache.key[:, list(entry.block_ids)], saved_keys, atol=0, rtol=0)
    torch.testing.assert_close(runtime.cache.value[:, list(entry.block_ids)], saved_values, atol=0, rtol=0)
    cached = runtime.prefix_cache.lookup(suffix)
    assert len(cached.token_key) == 12
    torch.testing.assert_close(cached.next_logits, boundary, atol=2e-5, rtol=2e-5)
    runtime.close()
    assert runner.block_manager.used_blocks == 0


def test_intermediate_chunks_update_kv_without_lm_head_or_sampling():
    runner = make_runner()
    runtime = runner.create_batch_runtime(enable_prefix_cache=False,
        prefill_chunk_size=3, max_num_batched_tokens=6)
    requests = [GenerationRequest("long", list(range(1, 9)), max_new_tokens=1),
                GenerationRequest("short", [11, 12, 13, 14, 15], max_new_tokens=1)]
    head_rows = []
    handle = runner.model.lm_head.register_forward_pre_hook(
        lambda module, args: head_rows.append(args[0].shape[0]))
    try:
        for request in requests:
            runtime.scheduler.add_request(request)
        assert runtime.step() == {}
        assert head_rows == []
        assert [request.prefill_cursor for request in requests] == [3, 3]
        assert set(runtime.step()) == {"short"}
        assert head_rows == [1]
        assert set(runtime.step()) == {"long"}
        assert head_rows == [1, 1]
    finally:
        handle.remove()
        runtime.close()
    assert runner.block_manager.used_blocks == 0


def test_packed_execution_failure_aborts_all_reserved_prompt_rows(monkeypatch):
    runner = make_runner()
    runtime = runner.create_batch_runtime(enable_prefix_cache=False)
    requests = [GenerationRequest(str(i), [i + 1, i + 2, i + 3], max_new_tokens=2) for i in range(2)]
    def fail(*args, **kwargs):
        raise RuntimeError("injected packed prefill failure")
    monkeypatch.setattr(runner.model, "forward_prefill_batch", fail)
    for request in requests:
        runtime.scheduler.add_request(request)
    with pytest.raises(RuntimeError, match="packed prefill"):
        runtime.step()
    assert all(request.state is RequestState.ABORTED for request in requests)
    assert not runtime.scheduler.running and not runtime.scheduler.waiting
    assert runner.block_manager.used_blocks == 0
    runtime.close()


def test_cancelled_chunk_does_not_change_other_request_completion():
    runner = make_runner()
    runtime = runner.create_batch_runtime(enable_prefix_cache=False,
        prefill_chunk_size=3, max_num_batched_tokens=6)
    cancelled = GenerationRequest("cancel", [1, 2, 3, 4, 5, 6, 7], max_new_tokens=2)
    survivor = GenerationRequest("keep", [11, 12, 13, 14, 15, 16], max_new_tokens=1)
    expected = int(cold_logits(runner.model, survivor.prompt_token_ids)[-1].argmax())
    runtime.scheduler.add_request(cancelled)
    runtime.scheduler.add_request(survivor)
    assert runtime.step() == {}
    runtime.scheduler.abort("cancel")
    finish(runtime)
    assert cancelled.state is RequestState.ABORTED
    assert survivor.output_token_ids == [expected]
    runtime.close()
    assert runner.block_manager.used_blocks == 0
