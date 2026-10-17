"""Tiny random weights validate math without downloading model weights."""

import importlib.util

import pytest
import torch
from torch.nn import functional as F

from mini_engine.kernels.fallback import paged_attention
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config


def test_paged_attention_noncontiguous_gqa():
    torch.manual_seed(7)
    key, value = torch.randn(3, 4, 2, 8), torch.randn(3, 4, 2, 8)
    query = torch.randn(1, 6, 8)
    actual = paged_attention(query, key, value, torch.tensor([[2, 0]]), torch.tensor([7]), 4)
    k = key[[2, 0]].reshape(8, 2, 8)[:7].repeat_interleave(3, dim=1).transpose(0, 1)
    v = value[[2, 0]].reshape(8, 2, 8)[:7].repeat_interleave(3, dim=1).transpose(0, 1)
    expected = F.scaled_dot_product_attention(query.unsqueeze(2), k.unsqueeze(0), v.unsqueeze(0)).squeeze(2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device,dtype,atol", [
    ("cpu", torch.float32, 2e-5),
    pytest.param("cuda", torch.bfloat16, 0.025, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    pytest.param("cuda", torch.float16, 0.005, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_native_matches_transformers_prefill_decode(device, dtype, atol):
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("install .[model] for HF numerical alignment")
    from transformers import Qwen3Config as HFConfig, Qwen3ForCausalLM

    torch.manual_seed(17)
    cfg = HFConfig(vocab_size=71, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                   max_position_embeddings=128, tie_word_embeddings=True,
                   attention_dropout=0.0, bos_token_id=1, eos_token_id=2)
    cfg._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    native = NativeQwen3.from_hf_model(reference, prefill_backend="math")
    cache = native.create_cache(8, 4)
    block_ids = [6, 1, 4, 0]
    tokens = torch.tensor([3, 8, 11, 5, 9], device=device)
    with torch.inference_mode():
        expected = reference(tokens[None], use_cache=False).logits[0]
        actual = native(tokens, cache, block_ids)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
        for _ in range(6):
            next_id = expected[-1].argmax().view(1)
            position = tokens.numel()
            tokens = torch.cat([tokens, next_id])
            expected = reference(tokens[None], use_cache=False).logits[0]
            actual = native(next_id, cache, block_ids, position=position)
            torch.testing.assert_close(actual[-1], expected[-1], atol=atol, rtol=atol)
            assert actual[-1].argmax() == expected[-1].argmax()


class TinyTokenizer:
    eos_token_id = None

    def apply_chat_template(self, *args, **kwargs):
        return torch.tensor([[1, 3, 5]])

    def decode(self, ids, **kwargs):
        return str(ids)


def make_runner():
    cfg = Qwen3Config(17, 16, 32, 1, 2, 1, 8, max_position_embeddings=32)
    return NativeQwen3Runner(NativeQwen3(cfg, "math"), TinyTokenizer(), num_blocks=4, block_size=4)


def test_generation_releases_blocks_and_counts_forward_calls():
    runner = make_runner()
    calls = []
    handle = runner.model.register_forward_hook(lambda *args: calls.append(1))
    try:
        result = runner.generate_incremental("hi", 5)
        assert len(calls) == 5  # one prefill plus four decode forwards
        assert len(result.token_ids) == 5 and result.finish_reason == "length"
        assert result.total_latency_ms >= result.ttft_ms >= 0
        assert runner.block_manager.free_blocks == 4
        single = runner.generate_incremental("hi", 1)
        assert single.tpot_ms == 0
    finally:
        handle.remove()


def test_generation_error_releases_cache():
    runner = make_runner()
    def fail(*args):
        raise RuntimeError("injected failure")
    handle = runner.model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            runner.generate_incremental("hi", 4)
        assert runner.block_manager.free_blocks == 4
        assert not runner._busy
    finally:
        handle.remove()
    assert len(runner.generate_incremental("hi", 1).token_ids) == 1


def test_generation_rejects_capacity_and_zero_output():
    runner = make_runner()
    with pytest.raises(MemoryError):
        runner.generate_incremental("hi", 20)
    with pytest.raises(ValueError):
        runner.generate_incremental("hi", 0)


def test_eos_stops_without_extra_decode():
    runner = make_runner()
    runner.eos_token_ids = list(range(runner.model.config.vocab_size))
    result = runner.generate_incremental("hi", 5)
    assert result.finish_reason == "stop" and len(result.token_ids) == 1
    assert result.tpot_ms == 0.0
    assert runner.block_manager.free_blocks == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_windows_sdpa_auto_and_flash_availability():
    from mini_engine.kernels.prefill import prefill_attention
    q = torch.randn(1, 4, 13, 16, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q[:, :2]), torch.randn_like(q[:, :2])
    expected = prefill_attention(q, k, v, "math")
    actual = prefill_attention(q, k, v, "auto")
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    if not torch.backends.cuda.is_flash_attention_available():
        with pytest.raises(RuntimeError, match="no CUDA Flash"):
            prefill_attention(q, k, v, "flash")


def test_cache_rejects_negative_physical_id():
    runner = make_runner()
    with pytest.raises(IndexError, match="physical block"):
        runner.cache.gather(0, [-1], 1)
