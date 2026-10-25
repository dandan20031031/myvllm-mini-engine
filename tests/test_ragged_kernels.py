"""Packed prefill and flat KV scatter: reference, contracts and CUDA streams."""

import os

import pytest
import torch

from mini_engine.kernels.data import write_kv_cache_slots
from mini_engine.kernels.prefill import paged_prefill_attention


CUDA = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available")


def inputs(device="cpu", dtype=torch.float32, dim=16):
    torch.manual_seed(631)
    key = torch.randn(8, 4, 2, dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    q = torch.randn(8, 4, dim, device=device, dtype=dtype)
    metadata = [torch.tensor(x, device=device, dtype=torch.int64) for x in (
        [0, 3, 8], [5, 1], [[5, 1, -1], [2, 7, -1]], [8, 6])]
    return q, key, value, *metadata


def oracle(q, kc, vc, offsets, starts, tables, lengths, block):
    result = []
    for request, (begin, end) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:])):
        for token in range(begin, end):
            context = starts[request].item() + token - begin + 1
            logical = torch.arange(context, device=q.device)
            physical = tables[request, logical // block]
            key = kc[physical, logical % block].repeat_interleave(q.shape[1] // kc.shape[2], dim=1).float()
            value = vc[physical, logical % block].repeat_interleave(q.shape[1] // kc.shape[2], dim=1).float()
            score = torch.einsum("hd,thd->ht", q[token].float(), key) / q.shape[-1] ** 0.5
            result.append(torch.einsum("ht,thd->hd", score.softmax(-1), value))
    return torch.stack(result).to(q.dtype)


def test_reference_packed_prefill_reads_noncontiguous_pages_and_absolute_suffix():
    args = inputs()
    expected = oracle(*args, 4)
    actual = paged_prefill_attention(*args, 4, backend="torch")
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_reference_slots_scatter_preserves_other_cache_rows_and_rejects_races():
    _, key, value, *_ = inputs()
    original_k, original_v = key.clone(), value.clone()
    slots = torch.tensor([5, 6, 7, 9, 10, 11, 28, 29], dtype=torch.int64)
    # Split projection views need not be contiguous in the input token dimension.
    packed = torch.randn(8, 4, 16)
    source_k, source_v = packed[:, :2], packed[:, 2:]
    write_kv_cache_slots(key, value, slots, source_k, source_v)
    expected_k, expected_v = original_k.view(-1, 2, 16), original_v.view(-1, 2, 16)
    expected_k[slots], expected_v[slots] = source_k, source_v
    assert torch.equal(key.view_as(expected_k), expected_k)
    assert torch.equal(value.view_as(expected_v), expected_v)
    with pytest.raises(ValueError, match="duplicate"):
        write_kv_cache_slots(key, value, torch.zeros(8, dtype=torch.int64), source_k, source_v)
    with pytest.raises(ValueError, match="overlap"):
        write_kv_cache_slots(key, value, slots, key.view(-1, 2, 16)[:8], source_v)
    with pytest.raises(ValueError, match="overlap"):
        write_kv_cache_slots(key, key, slots, source_k, source_v)


@pytest.mark.parametrize("change,message", [
    (lambda a: a[3].fill_(0), "offset"),
    (lambda a: a[4].fill_(-1), "context"),
    (lambda a: a[6].fill_(99), "capacity"),
    (lambda a: a[5].fill_(-1), "physical"),
    (lambda a: a[5].fill_(8), "physical"),
    (lambda a: a.__setitem__(3, a[3].to(torch.int32)), "int64"),
])
def test_invalid_paged_metadata_rejected_before_dispatch(change, message):
    args = list(inputs())
    change(args)
    with pytest.raises(ValueError, match=message):
        paged_prefill_attention(*args, 4, backend="torch")


@pytest.mark.parametrize("slot", [-1, 32])
def test_slot_capacity_rejected_before_mutation(slot):
    _, key, value, *_ = inputs()
    before = key.clone()
    with pytest.raises(ValueError, match="capacity"):
        write_kv_cache_slots(key, value, torch.tensor([slot]), torch.ones(1, 2, 16), torch.ones(1, 2, 16))
    assert torch.equal(key, before)


def test_empty_packed_queries_and_slots_are_supported():
    q, key, value, *_ = inputs()
    offsets = torch.tensor([0], dtype=torch.int64)
    empty = torch.empty(0, dtype=torch.int64)
    result = paged_prefill_attention(q[:0], key, value, offsets, empty,
        torch.empty(0, 0, dtype=torch.int64), empty, 4, backend="torch")
    assert result.shape == q[:0].shape
    write_kv_cache_slots(key, value, empty, key.new_empty(0, 2, 16), value.new_empty(0, 2, 16))


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [16, 128])
def test_cuda_packed_attention_and_slot_scatter_match_reference_on_nondefault_stream(dtype, dim):
    q, key, value, offsets, starts, tables, lengths = inputs("cuda", dtype, dim)
    slots = torch.tensor([5, 6, 7, 9, 10, 11, 28, 29], device="cuda", dtype=torch.int64)
    source_k = torch.randn(8, 2, dim, device="cuda", dtype=dtype)
    source_v = torch.randn_like(source_k)
    expected_k, expected_v = key.clone(), value.clone()
    write_kv_cache_slots(expected_k, expected_v, slots, source_k, source_v, backend="torch")
    expected = oracle(q, expected_k, expected_v, offsets, starts, tables, lengths, 4)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        write_kv_cache_slots(key, value, slots, source_k, source_v, backend="cuda")
        result = paged_prefill_attention(q, key, value, offsets, starts, tables, lengths, 4)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(key, expected_k) and torch.equal(value, expected_v)
    tolerance = 0.02 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)


@CUDA
def test_cuda_paged_context_exceeds_dense_prefill_limit():
    block, context = 32, 8193
    pages = (context + block - 1) // block
    key = torch.randn(pages, block, 1, 16, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    q = torch.randn(1, 2, 16, device="cuda", dtype=torch.float16)
    tables = torch.randperm(pages, device="cuda").view(1, -1)
    offsets = torch.tensor([0, 1], device="cuda")
    starts, lengths = torch.tensor([context - 1], device="cuda"), torch.tensor([context], device="cuda")
    actual = paged_prefill_attention(q, key, value, offsets, starts, tables, lengths, block)
    expected = oracle(q, key, value, offsets, starts, tables, lengths, block)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)
