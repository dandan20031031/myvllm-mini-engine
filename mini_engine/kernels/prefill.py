"""Windows-compatible SDPA prefill; Flash is optional, never assumed."""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def prefill_attention(q, k, v, backend="auto", *, query_start=0):
    """Inputs [1, heads, tokens, dim]; use causal attention without a dense mask.

    auto lets PyTorch select its available kernels. flash is a strict request:
    a PyTorch build without Flash support fails instead of reporting false use.
    math is useful for reproducible correctness comparisons.
    """
    if backend not in {"auto", "flash", "math", "cuda", "dandan"}:
        raise ValueError("unknown prefill backend")
    if backend == "flash" and (q.device.type != "cuda" or not torch.backends.cuda.is_flash_attention_available()):
        raise RuntimeError("This PyTorch build has no CUDA Flash Attention backend; select auto or math")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[1] % k.shape[1]:
        raise ValueError("prefill expects [batch, heads, tokens, dim] and divisible GQA heads")
    if (q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]
            or q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device):
        raise ValueError("prefill tensors must share batch, head_dim, dtype and device")
    if (not isinstance(query_start, int) or isinstance(query_start, bool) or query_start < 0
            or q.shape[-2] <= 0 or query_start + q.shape[-2] > k.shape[-2]):
        raise ValueError("query_start and query length must fit the KV sequence")
    if backend == "dandan":
        from .dandan_backend import validate_dandan_runtime, require_dandan_capability
        validate_dandan_runtime(q.device, q.dtype)
        ops = require_dandan_capability("prefill")
        # Keep native GQA heads: expansion would waste memory and bandwidth.
        return ops.flash_attention(q, k, v, causal=True, query_start=query_start, backend="cuda")
    if backend == "cuda":
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("CUDA prefill requires CUDA FP16/BF16 tensors")
        if k.shape[-2] > 8192:
            raise ValueError("CUDA prefill supports at most 8192 tokens; use auto/math for longer prompts")
        from .cuda_extension import load_cuda_extension
        return load_cuda_extension().causal_prefill_suffix(
            q.contiguous(), k.contiguous(), v.contiguous(), query_start,
        )
    groups = q.shape[1] // k.shape[1]
    # Explicit GQA expansion also works with PyTorch builds without enable_gqa.
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    suffix = query_start != 0 or q.shape[-2] != k.shape[-2]
    mask = None
    if suffix:
        # SDPA's rectangular is_causal mask starts at row zero. A suffix row
        # must instead see every key up to its absolute position.
        q_positions = query_start + torch.arange(q.shape[-2], device=q.device)
        k_positions = torch.arange(k.shape[-2], device=q.device)
        mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    if backend == "auto":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=not suffix)
    selected = SDPBackend.FLASH_ATTENTION if backend == "flash" else SDPBackend.MATH
    with sdpa_kernel(selected):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=not suffix)


def paged_prefill_attention(q, key_cache, value_cache, cu_seqlens_q, query_starts,
                            block_tables, context_lens, block_size, *, backend="cuda", validate=True):
    """Packed causal queries read paged KV directly on the CUDA path.

    Q is [total_queries, query_heads, head_dim]. Each request describes a suffix
    starting at its absolute query_start; context_len equals start + query_count.
    The torch path materializes KV only as a correctness reference. Validation
    of metadata values synchronizes CUDA; trusted model batches can skip it.
    """
    if backend not in {"torch", "cuda"}:
        raise ValueError("paged prefill backend must be torch or cuda")
    if (type(block_size) is not int or block_size <= 0 or key_cache.ndim != 4
            or value_cache.shape != key_cache.shape or key_cache.shape[1] != block_size
            or any(size <= 0 for size in key_cache.shape)
            or not key_cache.is_contiguous() or not value_cache.is_contiguous()):
        raise ValueError("paged prefill caches must be contiguous [pages, block_size, kv_heads, head_dim]")
    if (q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] != key_cache.shape[3]
            or q.shape[1] % key_cache.shape[2]):
        raise ValueError("paged prefill Q must be [tokens, heads, head_dim] with divisible GQA heads")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("paged prefill requires FP32/FP16/BF16")
    if key_cache.dtype != q.dtype or value_cache.dtype != q.dtype:
        raise ValueError("paged prefill Q/K/V must share dtype")
    metadata = (cu_seqlens_q, query_starts, block_tables, context_lens)
    if (cu_seqlens_q.ndim != 1 or query_starts.ndim != 1 or context_lens.ndim != 1
            or block_tables.ndim != 2 or query_starts.numel() != block_tables.shape[0]
            or context_lens.numel() != block_tables.shape[0]
            or cu_seqlens_q.numel() != block_tables.shape[0] + 1):
        raise ValueError("paged prefill metadata shapes do not match request count")
    if any(t.dtype != torch.int64 or not t.is_contiguous() for t in metadata):
        raise ValueError("paged prefill metadata must be contiguous int64 tensors")
    if any(t.device != q.device for t in (key_cache, value_cache, *metadata)):
        raise ValueError("paged prefill tensors must share device")
    if validate:
        offsets, starts, tables, lengths = (t.tolist() for t in metadata)
        if offsets[0] != 0 or offsets[-1] != q.shape[0] or any(b < a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("query offsets must be monotone and cover all packed queries")
        for index, (start, length, table) in enumerate(zip(starts, lengths, tables)):
            count = offsets[index + 1] - offsets[index]
            if start < 0 or length != start + count or length > len(table) * block_size:
                raise ValueError("query start/context length exceeds block table capacity or query range")
            used = (length + block_size - 1) // block_size
            if any(page < 0 or page >= key_cache.shape[0] for page in table[:used]):
                raise ValueError("active block table contains an invalid physical page")
    if backend == "cuda":
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("CUDA paged prefill requires CUDA FP16/BF16")
        if q.shape[2] > 1024:
            raise ValueError("CUDA paged prefill supports head_dim up to 1024")
        from .cuda_extension import load_cuda_extension
        return load_cuda_extension().paged_prefill_attention(
            q.contiguous(), key_cache, value_cache, cu_seqlens_q, query_starts,
            block_tables, context_lens, block_size, False)
    output = torch.empty_like(q)
    offsets, starts, lengths = (t.tolist() for t in (cu_seqlens_q, query_starts, context_lens))
    for index, (start, length) in enumerate(zip(starts, lengths)):
        begin, end = offsets[index:index + 2]
        if begin == end:
            continue
        pages = block_tables[index, :(length + block_size - 1) // block_size]
        key = key_cache[pages].reshape(-1, *key_cache.shape[2:])[:length]
        value = value_cache[pages].reshape(-1, *value_cache.shape[2:])[:length]
        groups = q.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        value = value.repeat_interleave(groups, dim=1).transpose(0, 1).float()
        query = q[begin:end].transpose(0, 1).float()
        mask = torch.arange(length, device=q.device)[None, :] <= (
            start + torch.arange(end - begin, device=q.device)[:, None])
        rows = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        output[begin:end] = rows.transpose(0, 1).to(q.dtype)
    return output
