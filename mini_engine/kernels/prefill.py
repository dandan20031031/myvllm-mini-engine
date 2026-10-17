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
    if backend not in {"auto", "flash", "math"}:
        raise ValueError("prefill backend must be auto, flash or math")
    if backend == "flash" and (q.device.type != "cuda" or not torch.backends.cuda.is_flash_attention_available()):
        raise RuntimeError("This PyTorch build has no CUDA Flash Attention backend; select auto or math")
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[1] % k.shape[1]:
        raise ValueError("prefill expects [batch, heads, tokens, dim] and divisible GQA heads")
    groups = q.shape[1] // k.shape[1]
    # Explicit GQA expansion also works with PyTorch builds without enable_gqa.
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    if backend == "auto":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    selected = SDPBackend.FLASH_ATTENTION if backend == "flash" else SDPBackend.MATH
    with sdpa_kernel(selected):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
