"""Suffix math, real prefix reuse, bounded prefill and allocator progress."""

import os

import pytest
import torch

from mini_engine.cache import BlockManager, PrefixCache
from mini_engine.engine.request import GenerationRequest, RequestState
from mini_engine.engine.runtime import NativeBatchRuntime
from mini_engine.engine.scheduler import Scheduler
from mini_engine.kernels.prefill import prefill_attention
from mini_engine.model import NativeQwen3, Qwen3Config


def make_runtime(*, blocks=32, block_size=4, chunk=None, budget=64, prefix=True, cache=None):
    torch.manual_seed(231)
    model = NativeQwen3(Qwen3Config(41, 32, 64, 2, 4, 2, 8, max_position_embeddings=128), "math").eval()
    manager = BlockManager(blocks, block_size)
    scheduler = Scheduler(manager, budget)
    runtime = NativeBatchRuntime(model, scheduler, model.create_cache(blocks, block_size),
                                 prefill_chunk_size=chunk, enable_prefix_cache=prefix, prefix_cache=cache)
    return runtime


def finish(runtime, *requests, max_steps=100):
    for request in requests:
        runtime.scheduler.add_request(request)
    outputs = []
    for _ in range(max_steps):
        if not runtime.scheduler.waiting and not runtime.scheduler.running:
            return outputs
        outputs.append(runtime.step())
    raise AssertionError("scheduler failed to make progress")


def test_suffix_attention_uses_absolute_causal_positions():
    torch.manual_seed(232)
    q = torch.randn(1, 4, 3, 8)
    k, v = torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
    expanded_k, expanded_v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    scores = q @ expanded_k.transpose(-1, -2) / 8 ** 0.5
    visible = torch.arange(8)[None, :] <= torch.arange(5, 8)[:, None]
    expected = scores.masked_fill(~visible, float("-inf")).softmax(-1) @ expanded_v
    torch.testing.assert_close(prefill_attention(q, k, v, "math", query_start=5), expected)
    with pytest.raises(ValueError, match="query_start"):
        prefill_attention(q, k, v, "math", query_start=6)


def test_native_suffix_chunks_match_full_logits_and_keep_decode_paged(monkeypatch):
    runtime = make_runtime()
    model = runtime.model
    full_cache, suffix_cache = model.create_cache(8, 4), model.create_cache(8, 4)
    blocks = [6, 1, 4, 0]
    tokens = torch.tensor([1, 5, 9, 12, 6, 3, 8, 7, 2])
    expected = model(tokens, full_cache, blocks)
    first = model(tokens[:3], suffix_cache, blocks)
    second = model(tokens[3:5], suffix_cache, blocks, position=3)
    last = model(tokens[5:], suffix_cache, blocks, position=5)
    torch.testing.assert_close(torch.cat((first, second, last)), expected, atol=2e-5, rtol=2e-5)
    for layer in range(model.config.num_hidden_layers):
        for actual, reference in zip(suffix_cache.gather(layer, blocks, 9), full_cache.gather(layer, blocks, 9)):
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)

    def forbidden(*args, **kwargs):
        raise AssertionError("one-token decode must not gather the full KV sequence")
    monkeypatch.setattr(suffix_cache, "gather", forbidden)
    model(torch.tensor([4]), suffix_cache, blocks, position=9)


def test_exact_prefix_hit_skips_every_model_forward_and_preserves_shared_kv():
    runtime = make_runtime()
    calls = []
    handle = runtime.model.register_forward_pre_hook(lambda model, args, kwargs: calls.append(
        (args[0].numel(), kwargs.get("position", 0))), with_kwargs=True)
    try:
        first = GenerationRequest("first", list(range(1, 9)), max_new_tokens=1)
        finish(runtime, first)
        entry = runtime.prefix_cache.lookup(first.prompt_token_ids)
        saved = runtime.cache.key[:, list(entry.block_ids)].clone()
        second = GenerationRequest("second", first.prompt_token_ids.copy(), max_new_tokens=2)
        runtime.scheduler.add_request(second)
        assert runtime.step()["second"] == first.output_token_ids[0]
        assert calls == [(8, 0)]
        assert second.prefix_len == 8
        torch.testing.assert_close(runtime.cache.key[:, list(entry.block_ids)], saved, atol=0, rtol=0)
        runtime.step()
        assert second.is_finished
        assert all(runtime.scheduler.block_manager.refcount(block) == 1 for block in entry.block_ids)
    finally:
        handle.remove()


def test_unaligned_prompt_caches_boundary_logits_instead_of_last_prompt_logits():
    runtime = make_runtime()
    first = GenerationRequest("first", [1, 5, 9, 3, 7, 11], max_new_tokens=1)
    finish(runtime, first)
    entry = runtime.prefix_cache.lookup(first.prompt_token_ids)
    assert entry.token_key == (1, 5, 9, 3)
    independent = runtime.model.create_cache(4, 4)
    expected = runtime.model(torch.tensor(entry.token_key), independent, [1])[-1:]
    torch.testing.assert_close(entry.next_logits, expected, atol=2e-5, rtol=2e-5)
    exact = GenerationRequest("exact", list(entry.token_key), max_new_tokens=1)
    def forbidden(*args, **kwargs):
        raise AssertionError("boundary hit should use its matching cached logit")
    handle = runtime.model.register_forward_pre_hook(forbidden)
    try:
        finish(runtime, exact)
    finally:
        handle.remove()
    assert exact.output_token_ids == [int(expected.argmax())]


def test_cache_insert_discards_logits_that_belong_after_a_rounded_key():
    manager = BlockManager(4, 4)
    physical = manager.allocate("seed", 6)
    cache = PrefixCache(manager)
    entry = cache.insert([1, 2, 3, 4, 5, 6], physical, next_logits=torch.tensor([[1., 2.]]))
    assert entry.next_logits is None
    boundary = torch.tensor([[7., 9.]])
    entry = cache.insert([1, 2, 3, 4, 5, 6], physical,
                         next_logits=boundary, logits_position=4)
    torch.testing.assert_close(entry.next_logits, boundary)


def test_partial_prefix_computes_only_suffix_and_never_rewrites_shared_pages():
    runtime, cold = make_runtime(), make_runtime(prefix=False)
    warm = GenerationRequest("warm", list(range(1, 9)), max_new_tokens=1)
    finish(runtime, warm)
    entry = runtime.prefix_cache.lookup(warm.prompt_token_ids)
    saved = runtime.cache.key[:, list(entry.block_ids)].clone()
    calls = []
    handle = runtime.model.register_forward_pre_hook(lambda model, args, kwargs: calls.append(
        (args[0].numel(), kwargs.get("position", 0))), with_kwargs=True)
    request = GenerationRequest("suffix", list(range(1, 11)), max_new_tokens=3)
    expected = GenerationRequest("cold", request.prompt_token_ids.copy(), max_new_tokens=3)
    try:
        finish(runtime, request)
        finish(cold, expected)
    finally:
        handle.remove()
    assert calls == [(2, 8)]  # Subsequent steps use forward_decode_batch.
    assert request.output_token_ids == expected.output_token_ids
    torch.testing.assert_close(runtime.cache.key[:, list(entry.block_ids)], saved, atol=0, rtol=0)


def test_exact_prefix_without_logits_recomputes_private_pages():
    runtime = make_runtime()
    manager = runtime.scheduler.block_manager
    seed = manager.allocate("seed", 4)
    runtime.model(torch.tensor([1, 2, 3, 4]), runtime.cache, seed)
    runtime.prefix_cache.insert([1, 2, 3, 4], seed)
    saved = runtime.cache.key[:, seed].clone()
    request = GenerationRequest("other", [1, 2, 3, 4], max_new_tokens=1)
    finish(runtime, request)
    assert request.prefix_len == 0
    assert request.block_ids != seed
    torch.testing.assert_close(runtime.cache.key[:, seed], saved, atol=0, rtol=0)
    assert manager.refcount(seed[0]) == 1  # The external seed owner remains.


def test_long_prompt_chunks_obey_budget_and_match_unchunked_generation():
    runtime = make_runtime(chunk=3, budget=4)
    cold = make_runtime(prefix=False)
    request = GenerationRequest("long", list(range(1, 14)), max_new_tokens=4)
    reference = GenerationRequest("reference", request.prompt_token_ids.copy(), max_new_tokens=4)
    calls = []
    handle = runtime.model.register_forward_pre_hook(lambda model, args, kwargs: calls.append(
        (args[0].numel(), kwargs.get("position", 0))), with_kwargs=True)
    try:
        steps = finish(runtime, request)
    finally:
        handle.remove()
    finish(cold, reference)
    assert calls == [(3, 0), (3, 3), (3, 6), (3, 9), (1, 12)]
    assert steps[:4] == [{}, {}, {}, {}]
    assert request.output_token_ids == reference.output_token_ids


def test_decode_executes_before_new_prefill_and_consumes_budget():
    runtime = make_runtime(chunk=5, budget=4, prefix=False)
    short = GenerationRequest("short", [1, 2], max_new_tokens=3)
    runtime.scheduler.add_request(short)
    runtime.step()
    long = GenerationRequest("long", list(range(1, 11)), max_new_tokens=1)
    runtime.scheduler.add_request(long)
    events = []
    original_decode = runtime.model.forward_decode_batch
    original_forward = runtime.model.forward
    runtime.model.forward_decode_batch = lambda *args, **kwargs: (events.append("decode"), original_decode(*args, **kwargs))[1]
    runtime.model.forward = lambda *args, **kwargs: (events.append(("prefill", args[0].numel())), original_forward(*args, **kwargs))[1]
    runtime.step()
    assert events == ["decode", ("prefill", 3)]
    assert long.prefill_cursor == 3 and not long.output_token_ids
    finish(runtime)


def test_virtual_growth_reservation_prevents_concurrent_allocation_deadlock():
    runtime = make_runtime(blocks=6, block_size=2, chunk=2, budget=4)
    first = GenerationRequest("first", [1, 2, 3], max_new_tokens=6)
    second = GenerationRequest("second", [4, 5, 6], max_new_tokens=6)
    runtime.scheduler.add_request(first)
    runtime.scheduler.add_request(second)
    runtime.step()
    assert first.state is RequestState.RUNNING and second.state is RequestState.WAITING
    finish(runtime, max_steps=30)
    assert first.is_finished and second.is_finished
    runtime.prefix_cache.clear()
    assert runtime.scheduler.block_manager.free_blocks == 6


def test_final_token_never_grows_an_unneeded_kv_page():
    runtime = make_runtime(blocks=1, block_size=2, prefix=False)
    request = GenerationRequest("one", [1, 2], max_new_tokens=1)
    finish(runtime, request)
    assert request.generated_tokens == 1
    assert runtime.scheduler.block_manager.free_blocks == 1


def test_lru_eviction_drops_only_cache_pins_not_live_request_pages():
    manager = BlockManager(2, 4)
    first = manager.allocate("live", 4)
    cache = PrefixCache(manager)
    cache.insert([1, 2, 3, 4], first)
    manager.allocate("other", 4)
    assert not cache.evict_until_free(1)
    assert len(cache) == 0 and manager.refcount(first[0]) == 1
    manager.release("live")
    assert manager.free_blocks == 1


def test_waiting_prefix_is_refreshed_after_an_earlier_admission_evicts_it():
    runtime = make_runtime(blocks=2, block_size=4)
    warm = GenerationRequest("warm", [1, 2, 3, 4], max_new_tokens=1)
    finish(runtime, warm)
    larger = GenerationRequest("larger", [5, 6, 7, 8, 9], max_new_tokens=1)
    reused = GenerationRequest("reused", warm.prompt_token_ids.copy(), max_new_tokens=1)
    runtime.scheduler.add_request(larger)
    runtime.scheduler.add_request(reused)
    runtime.prepare_waiting()
    assert reused.prefix_len == 4
    finish(runtime)
    assert reused.prefix_len == 0
    assert reused.output_token_ids == warm.output_token_ids


def test_impossible_request_with_reserved_scratch_fails_instead_of_spinning():
    runtime = make_runtime(blocks=2, block_size=4)
    runtime.scheduler.block_manager.allocate("scratch", 4)
    request = GenerationRequest("impossible", [1, 2, 3, 4], max_new_tokens=5)
    runtime.scheduler.add_request(request)
    with pytest.raises(MemoryError, match="scratch"):
        runtime.step()
    assert request.state is RequestState.ABORTED
    assert not runtime.scheduler.waiting


def test_empty_supplied_cache_is_preserved_and_can_be_disabled():
    runtime = make_runtime(prefix=False)
    supplied = PrefixCache(runtime.scheduler.block_manager)
    other = NativeBatchRuntime(runtime.model, runtime.scheduler, runtime.cache,
                               prefix_cache=supplied, enable_prefix_cache=False)
    assert other.prefix_cache is supplied
    finish(other, GenerationRequest("a", [1, 2, 3, 4], max_new_tokens=1))
    assert len(supplied) == 0


def test_prefill_failure_aborts_reserved_work_and_releases_private_pages():
    runtime = make_runtime(prefix=False)
    request = GenerationRequest("fails", [1, 2, 3], max_new_tokens=2)
    runtime.scheduler.add_request(request)
    def fail(*args, **kwargs):
        raise RuntimeError("injected prefill failure")
    handle = runtime.model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            runtime.step()
    finally:
        handle.remove()
    assert request.state is RequestState.ABORTED
    assert runtime.scheduler.block_manager.free_blocks == 32


def test_mixed_request_sampling_parameters_apply_during_batch_decode(monkeypatch):
    import mini_engine.engine.runtime as runtime_module

    runtime = make_runtime(prefix=False)
    first = GenerationRequest("greedy", [1, 2, 3], max_new_tokens=2)
    second = GenerationRequest("sample", [4, 5, 6], max_new_tokens=2, temperature=0.7, top_k=1)
    calls = []
    original = runtime_module.sample_next_token
    def track(logits, temperature=0., top_k=0, top_p=1., **kwargs):
        calls.append((temperature, top_k, top_p))
        return original(logits, temperature, top_k, top_p, **kwargs)
    monkeypatch.setattr(runtime_module, "sample_next_token", track)
    finish(runtime, first, second)
    assert calls == [(0., 0, 1.), (0.7, 1, 1.), (0., 0, 1.), (0.7, 1, 1.)]


def test_truncate_releases_only_trailing_request_references():
    manager = BlockManager(5, 4)
    blocks = manager.allocate("draft", 12)
    manager.pin([blocks[-1]])
    assert manager.truncate("draft", 5) == blocks[:2]
    assert manager.refcount(blocks[-1]) == 1
    assert manager.table("draft").block_ids == tuple(blocks[:2])
    manager.release("draft")
    manager.unpin([blocks[-1]])
    assert manager.free_blocks == 5


@pytest.mark.skipif(os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_rectangular_gqa_suffix_attention_matches_offset_math(dtype):
    torch.manual_seed(233)
    q = torch.randn(1, 4, 3, 16, device="cuda", dtype=dtype)
    k, v = torch.randn(1, 2, 11, 16, device="cuda", dtype=dtype), torch.randn(1, 2, 11, 16, device="cuda", dtype=dtype)
    expected = prefill_attention(q, k, v, "math", query_start=8)
    actual = prefill_attention(q, k, v, "cuda", query_start=8)
    torch.testing.assert_close(actual, expected, atol=0.025 if dtype == torch.bfloat16 else 0.004, rtol=0.02)


@pytest.mark.skipif(os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_cuda_multi_token_suffix_matches_full_prefill(dtype):
    torch.manual_seed(234)
    config = Qwen3Config(41, 32, 64, 2, 4, 2, 16, max_position_embeddings=64)
    model = NativeQwen3(config, "cuda", "cuda", "cuda", "cuda", "cuda").cuda().to(dtype).eval()
    full, suffix = model.create_cache(8, 4), model.create_cache(8, 4)
    ids = torch.tensor([1, 3, 7, 5, 9, 2, 11, 4, 8], device="cuda")
    blocks = [6, 1, 4, 0]
    expected = model(ids, full, blocks)
    actual = torch.cat((model(ids[:5], suffix, blocks), model(ids[5:], suffix, blocks, position=5)))
    torch.testing.assert_close(actual, expected, atol=0.06 if dtype == torch.bfloat16 else 0.015, rtol=0.03)
