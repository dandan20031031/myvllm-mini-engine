"""Stage 08 CUDA data-movement correctness and native integration."""

import os

import pytest
import torch

from mini_engine.kernels.data import apply_rope, embedding_lookup, write_kv_cache

pytestmark = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_embedding_matches_torch_and_int32_ids(dtype):
    torch.manual_seed(61)
    weight = torch.randn(37, 33, device="cuda", dtype=dtype)
    ids = torch.tensor([0, 36, 3, 3, 17], device="cuda", dtype=torch.int32)
    expected = embedding_lookup(weight, ids, backend="torch")
    actual = embedding_lookup(weight, ids, backend="cuda")
    torch.testing.assert_close(actual, expected, atol=0.0 if dtype == torch.float32 else 0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_rope_gqa_matches_reference(dtype):
    torch.manual_seed(62)
    tokens, q_heads, kv_heads, dim = 7, 8, 2, 16
    q = torch.randn(tokens, q_heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(tokens, kv_heads, dim, device="cuda", dtype=dtype)
    angles = torch.randn(tokens, dim // 2, device="cuda", dtype=dtype)
    cos, sin = angles.cos(), angles.sin()
    expected_q, expected_k = apply_rope(q, k, cos.unsqueeze(1), sin.unsqueeze(1), backend="torch")
    actual_q, actual_k = apply_rope(q, k, cos, sin, backend="cuda")
    torch.testing.assert_close(actual_q, expected_q, atol=0.004 if dtype == torch.bfloat16 else 0.001, rtol=0.01)
    torch.testing.assert_close(actual_k, expected_k, atol=0.004 if dtype == torch.bfloat16 else 0.001, rtol=0.01)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_kv_write_crosses_pages_and_uses_stream(dtype):
    torch.manual_seed(63)
    block_size, blocks, kv_heads, dim = 4, 6, 2, 8
    key_cache = torch.zeros(blocks, block_size, kv_heads, dim, device="cuda", dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    ids = torch.tensor([4, 1, 5], device="cuda", dtype=torch.long)
    key = torch.randn(7, kv_heads, dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    expected_k, expected_v = key_cache.clone(), value_cache.clone()
    write_kv_cache(expected_k, expected_v, ids, 1, key, value, block_size, backend="torch")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        write_kv_cache(key_cache, value_cache, ids, 1, key, value, block_size, backend="cuda")
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(key_cache, expected_k)
    assert torch.equal(value_cache, expected_v)


def test_cuda_data_rejects_bad_ids_and_shapes():
    weight = torch.randn(8, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="embedding id"):
        embedding_lookup(weight, torch.tensor([8], device="cuda"), backend="cuda")
    q = torch.randn(3, 4, 16, device="cuda", dtype=torch.float16)
    k = torch.randn(3, 2, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="RoPE cos"):
        apply_rope(q, k, torch.ones(2, 8, device="cuda", dtype=torch.float16),
                   torch.ones(3, 8, device="cuda", dtype=torch.float16), backend="cuda")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_qwen3_data_cuda_matches_torch(dtype):
    from mini_engine.model import NativeQwen3, Qwen3Config

    torch.manual_seed(64)
    config = Qwen3Config(53, 32, 64, 2, 4, 2, 16, max_position_embeddings=128)
    cuda_model = NativeQwen3(config, "math", "cuda", "cuda", "cuda").to(device="cuda", dtype=dtype).eval()
    torch_model = NativeQwen3(config, "math", "cuda", "cuda", "torch").to(device="cuda", dtype=dtype).eval()
    torch_model.load_state_dict(cuda_model.state_dict(), strict=True)
    cache_cuda, cache_torch = cuda_model.create_cache(8, 4), torch_model.create_cache(8, 4)
    ids = torch.tensor([3, 7, 11, 19, 23], device="cuda", dtype=torch.long)
    blocks = [6, 1, 4, 0]
    with torch.inference_mode():
        a = cuda_model(ids, cache_cuda, blocks)
        b = torch_model(ids, cache_torch, blocks)
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.012
        torch.testing.assert_close(a, b, atol=tolerance, rtol=0.02)
        for position in range(ids.numel(), ids.numel() + 4):
            token = b[-1].argmax().view(1)
            a = cuda_model(token, cache_cuda, blocks, position)
            b = torch_model(token, cache_torch, blocks, position)
            torch.testing.assert_close(a, b, atol=tolerance, rtol=0.02)
            assert a[-1].argmax() == b[-1].argmax()
