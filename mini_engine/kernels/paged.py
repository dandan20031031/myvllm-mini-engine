"""Explicit backend dispatch. CUDA compile/runtime failures are never hidden."""

import torch

from .fallback import paged_attention as reference_paged_attention


def paged_attention(query, key_cache, value_cache, block_tables, context_lens, block_size,
                    *, backend="torch", validate=True):
    if backend == "torch":
        return reference_paged_attention(query, key_cache, value_cache, block_tables, context_lens, block_size)
    if backend != "cuda":
        raise ValueError("decode backend must be torch or cuda")
    from .cuda_extension import load_cuda_extension
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("custom CUDA paged attention supports FP16/BF16")
    return load_cuda_extension().paged_attention(
        query, key_cache, value_cache, block_tables, context_lens, block_size, validate,
    )
