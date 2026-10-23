"""Windows-compatible SDPA prefill; Flash is optional, never assumed."""

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def prefill_attention(q, k, v, backend="auto"):
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
    if backend == "dandan":
        from .dandan_backend import validate_dandan_runtime, require_dandan_capability
        validate_dandan_runtime(q.device, q.dtype)
        ops = require_dandan_capability("prefill")
        # Keep native GQA heads: expansion would waste memory and bandwidth.
        return ops.flash_attention(q, k, v, causal=True, query_start=0, backend="cuda")
    groups = q.shape[1] // k.shape[1]
    # Explicit GQA expansion also works with PyTorch builds without enable_gqa.
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    if backend == "cuda":
        if q.device.type != "cuda" or q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("CUDA prefill requires CUDA FP16/BF16 tensors")
        from .cuda_extension import load_cuda_extension
        if q.shape[-2] > 8192:
            raise ValueError("CUDA prefill supports at most 8192 tokens; use auto/math for longer prompts")
        return load_cuda_extension().causal_prefill(q.contiguous(), k.contiguous(), v.contiguous())
    if backend == "auto":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    selected = SDPBackend.FLASH_ATTENTION if backend == "flash" else SDPBackend.MATH
    with sdpa_kernel(selected):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
