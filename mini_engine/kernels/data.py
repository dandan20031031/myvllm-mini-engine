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


def write_kv_cache_batch(key_cache, value_cache, block_tables, token_offsets, key, value,
                         block_size, *, backend="torch", validate=True):
    _backend(backend)
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("batch key/value must be [batch, kv_heads, head_dim]")
    if block_tables.ndim != 2 or token_offsets.ndim != 1 or block_tables.shape[0] != key.shape[0] or token_offsets.shape[0] != key.shape[0]:
        raise ValueError("batch block table or offset shape mismatch")
    tables = block_tables.to(device=key_cache.device, dtype=torch.long).contiguous()
    offsets = token_offsets.to(device=key_cache.device, dtype=torch.long).contiguous()
    if backend == "torch":
        for batch in range(key.shape[0]):
            write_kv_cache(key_cache, value_cache, tables[batch], int(offsets[batch]), key[batch].unsqueeze(0), value[batch].unsqueeze(0), block_size, backend="torch", validate=validate)
        return
    from .cuda_extension import load_cuda_extension
    load_cuda_extension().write_kv_cache_batch(key_cache, value_cache, tables, offsets, key, value, block_size, validate)


def write_kv_cache_slots(key_cache, value_cache, slot_mapping, key, value,
                         *, backend="torch", validate=True):
    """Scatter packed KV rows to unique physical ``page * block_size + offset`` slots.

    ``validate=False`` trusts slot contents checked by the batch factory; shape,
    dtype, device and cache layout checks remain active without a device sync.
    """
    _backend(backend)
    if (key_cache.ndim != 4 or value_cache.shape != key_cache.shape
            or any(size <= 0 for size in key_cache.shape)
            or not key_cache.is_contiguous() or not value_cache.is_contiguous()):
        raise ValueError("KV caches must be contiguous [pages, block_size, kv_heads, head_dim]")
    if (key.ndim != 3 or key.shape != value.shape
            or key.shape[1:] != key_cache.shape[2:]):
        raise ValueError("packed key/value must be [tokens, kv_heads, head_dim] matching the cache")
    if (slot_mapping.ndim != 1 or slot_mapping.numel() != key.shape[0]
            or slot_mapping.dtype != torch.int64 or not slot_mapping.is_contiguous()):
        raise ValueError("slot_mapping must be contiguous int64 [tokens]")
    for tensor in (value_cache, slot_mapping, key, value):
        if tensor.device != key_cache.device:
            raise ValueError("packed KV tensors and slots must share the cache device")
    if any(t.dtype != key_cache.dtype for t in (value_cache, key, value)):
        raise ValueError("packed KV tensors must share the cache dtype")
    if key_cache.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("packed KV requires FP32/FP16/BF16")
    if validate:
        slots = slot_mapping.tolist()
        capacity = key_cache.shape[0] * key_cache.shape[1]
        if any(slot < 0 or slot >= capacity for slot in slots):
            raise ValueError("KV slot is outside cache capacity")
        if len(set(slots)) != len(slots):
            raise ValueError("duplicate KV slots would race during scatter")
        if (torch._C._overlaps(key_cache, value_cache)
                or any(torch._C._overlaps(cache, source)
                       for cache in (key_cache, value_cache) for source in (key, value, slot_mapping))):
            raise ValueError("KV destinations and source tensors must not overlap")
    if backend == "torch":
        rows_k = key_cache.view(-1, *key_cache.shape[2:])
        rows_v = value_cache.view(-1, *value_cache.shape[2:])
        rows_k.index_copy_(0, slot_mapping, key)
        rows_v.index_copy_(0, slot_mapping, value)
        return
    if key_cache.device.type != "cuda" or key_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA packed KV writes require CUDA FP16/BF16")
    from .cuda_extension import load_cuda_extension
    load_cuda_extension().write_kv_cache_slots(
        key_cache, value_cache, slot_mapping, key.contiguous(), value.contiguous(), False)
