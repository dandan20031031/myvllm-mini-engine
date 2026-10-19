"""CUDA dispatch for embedding, rotary position encoding and KV writes."""

import torch
from torch.nn import functional as F

from .fallback import apply_rope as reference_rope


def _backend(backend):
    if backend not in {"torch", "cuda"}:
        raise ValueError("data backend must be torch or cuda")


def embedding_lookup(weight, input_ids, *, backend="torch", validate=True):
    _backend(backend)
    if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a 1D int32/int64 tensor")
    if backend == "torch":
        return F.embedding(input_ids, weight)
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("CUDA embedding supports FP32/FP16/BF16 weights")
    input_ids = input_ids.to(dtype=torch.long)
    from .cuda_extension import load_cuda_extension
    return load_cuda_extension().embedding_lookup(weight, input_ids, validate)


def apply_rope(q, k, cos, sin, *, backend="torch", validate=True):
    _backend(backend)
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("q and k must be [tokens, heads, head_dim] with matching token/head dimensions")
    if backend == "torch":
        return reference_rope(q, k, cos, sin)
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA RoPE supports FP16/BF16")
    from .cuda_extension import load_cuda_extension
    return load_cuda_extension().apply_rope(q, k, cos, sin, validate)


def write_kv_cache(key_cache, value_cache, block_ids, token_offset, key, value,
                   block_size, *, backend="torch", validate=True):
    _backend(backend)
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must be [tokens, kv_heads, head_dim]")
    if key.device != key_cache.device or value.device != key_cache.device:
        raise ValueError("KV tensors must be on the cache device")
    if key.dtype != key_cache.dtype or value.dtype != key_cache.dtype:
        raise ValueError("KV tensors must match cache dtype")
    ids = block_ids.reshape(-1).to(device=key_cache.device, dtype=torch.long)
    if token_offset < 0 or token_offset + key.shape[0] > ids.numel() * block_size:
        raise ValueError("write exceeds supplied block table capacity")
    if backend == "torch":
        remaining, source, position = key.shape[0], 0, token_offset
        while remaining:
            block_index, in_block = divmod(position, block_size)
            count = min(remaining, block_size - in_block)
            physical = ids[block_index]
            key_cache[physical, in_block:in_block + count].copy_(key[source:source + count])
            value_cache[physical, in_block:in_block + count].copy_(value[source:source + count])
            source += count
            position += count
            remaining -= count
        return
    from .cuda_extension import load_cuda_extension
    load_cuda_extension().write_kv_cache(key_cache, value_cache, ids, token_offset, key, value, block_size, validate)
