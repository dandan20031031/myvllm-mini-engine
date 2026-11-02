"""Explicit backend dispatch. CUDA compile/runtime failures are never hidden."""

import torch

from .fallback import paged_attention as reference_paged_attention


def paged_attention(query, key_cache, value_cache, block_tables, context_lens, block_size,
                    *, backend="torch", validate=True):
    if backend == "torch":
        return reference_paged_attention(query, key_cache, value_cache, block_tables, context_lens, block_size)
    if backend not in {"cuda", "cuda_baseline", "cuda_vector128"}:
        raise ValueError("decode backend must be torch, cuda, cuda_baseline or cuda_vector128")
    from .cuda_extension import load_cuda_extension
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("custom CUDA paged attention supports FP16/BF16")
    extension = load_cuda_extension()
    op = {"cuda": extension.paged_attention, "cuda_baseline": extension.paged_attention_baseline,
          "cuda_vector128": extension.paged_attention_vector128}[backend]
    return op(
        query, key_cache, value_cache, block_tables, context_lens, block_size, validate,
    )
