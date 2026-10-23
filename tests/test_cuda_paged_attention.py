"""Opt-in GPU toolchain tests: set MINI_ENGINE_TEST_CUDA=1 on Windows."""

import os

import pytest
import torch
from torch.nn import functional as F

from mini_engine.kernels.paged import paged_attention

pytestmark = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with MSVC/nvcc/Ninja available",
)


def inputs(dtype, head_dim=128, block_size=16, lengths=(17, 257, 521), heads=16, kv_heads=8):
    torch.manual_seed(23)
    width = (max(lengths) + block_size - 1) // block_size
    blocks = len(lengths) * width + 3
    key = torch.randn(blocks, block_size, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    table = torch.randperm(blocks, device="cuda")[:len(lengths) * width].view(len(lengths), width)
    lens = torch.tensor(lengths, device="cuda", dtype=torch.long)
    query = torch.randn(len(lengths), heads, head_dim, device="cuda", dtype=dtype)
    return query, key, value, table, lens, block_size


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [16, 32, 64, 80, 96, 128, 192, 256])
def test_cuda_gqa_splitkv_matches_dense(dtype, dim):
    args = inputs(dtype, head_dim=dim)
    actual = paged_attention(*args, backend="cuda")
    q, k, v, tables, lens, block = args
    expected = []
    for b, length in enumerate(lens.cpu().tolist()):
        ids = tables[b, :(length + block - 1) // block]
        key = k[ids].reshape(-1, k.shape[2], dim)[:length].repeat_interleave(q.shape[1] // k.shape[2], 1)
        value = v[ids].reshape(-1, v.shape[2], dim)[:length].repeat_interleave(q.shape[1] // v.shape[2], 1)
        out = F.scaled_dot_product_attention(q[b].float().unsqueeze(1), key.float().transpose(0, 1), value.float().transpose(0, 1))
        expected.append(out.squeeze(1))
    expected = torch.stack(expected)
    atol = 0.004 if dtype == torch.bfloat16 else 0.0006
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=0.015)


@pytest.mark.parametrize("block_size,lengths,heads,kv_heads", [
    (4, (1,), 4, 4), (16, (256, 255), 8, 1),
    (32, (257, 1025, 9), 16, 8), (16, (4097,), 16, 8),
])
def test_cuda_block_and_partition_boundaries(block_size, lengths, heads, kv_heads):
    args = inputs(torch.bfloat16, block_size=block_size, lengths=lengths, heads=heads, kv_heads=kv_heads)
    actual = paged_attention(*args, backend="cuda")
    expected = paged_attention(*args, backend="torch")
    torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.025)


def test_cuda_rejects_bad_metadata_and_noncontiguous_query():
    args = list(inputs(torch.float16))
    args[4][0] = 0
    with pytest.raises(RuntimeError, match="context length"):
        paged_attention(*args, backend="cuda")
    args[4][0] = 17
    args[3][0, 0] = -1
    with pytest.raises(RuntimeError, match="physical block"):
        paged_attention(*args, backend="cuda")
    args[3][0, 0] = 0
    args[0] = args[0].transpose(0, 1).contiguous().transpose(0, 1)
    with pytest.raises(RuntimeError, match="contiguous"):
        paged_attention(*args, backend="cuda")


def test_cuda_current_stream_and_trusted_invalid_indices():
    args = list(inputs(torch.bfloat16, lengths=(65,), head_dim=128))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = paged_attention(*args, backend="cuda", validate=False)
        expected = paged_attention(*args, backend="torch")
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.025)
    # Trusted mode cannot read arbitrary memory even with a corrupt page id.
    args[3].fill_(-1)
    actual = paged_attention(*args, backend="cuda", validate=False)
    assert torch.isfinite(actual).all() and not actual.any()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16, 32, 64, 128, 256])
def test_vector128_matches_scalar_baseline_and_graph(dtype, block_size):
    args = inputs(dtype, block_size=block_size, lengths=(1, 255, 257, 519))
    expected = paged_attention(*args, backend="cuda_baseline", validate=False)
    actual = paged_attention(*args, backend="cuda", validate=False)
    torch.testing.assert_close(actual, expected, atol=0.006 if dtype == torch.bfloat16 else 0.001, rtol=0.025)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_attention(*args, backend="cuda", validate=False)
    args[0].mul_(0.5)
    graph.replay()
    expected = paged_attention(*args, backend="torch")
    torch.testing.assert_close(out, expected, atol=0.006 if dtype == torch.bfloat16 else 0.001, rtol=0.025)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_logits_use_cuda_decode(dtype):
    from mini_engine.model import NativeQwen3, Qwen3Config

    torch.manual_seed(8)
    cfg = Qwen3Config(71, 32, 64, 2, 4, 2, 16, max_position_embeddings=128)
    model = NativeQwen3(cfg, "math", "torch").to(device="cuda", dtype=dtype).eval()
    torch_cache = model.create_cache(8, 4)
    cuda_cache = model.create_cache(8, 4)
    ids = torch.tensor([1, 5, 6, 8, 7], device="cuda")
    blocks = [6, 1, 4, 0]
    a = model(ids, torch_cache, blocks)[-1:]
    b = model(ids, cuda_cache, blocks)[-1:]
    for pos in range(5, 11):
        token = a.argmax(-1)
        model.decode_backend = "torch"
        a = model(token, torch_cache, blocks, pos)
        model.decode_backend = "cuda"
        b = model(token, cuda_cache, blocks, pos)
        torch.testing.assert_close(a, b, atol=0.06 if dtype == torch.bfloat16 else 0.015, rtol=0.02)
        assert a.argmax() == b.argmax()
