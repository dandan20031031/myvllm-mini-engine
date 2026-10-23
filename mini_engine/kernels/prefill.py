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
