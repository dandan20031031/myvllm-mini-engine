"""Inference-only PyTorch/CUDA dispatch for Qwen3 norm, residual and activation.

CUDA preserves the low-precision rounding points of the eager reference.
residual_rms_norm returns (normalized, summed), leaving both inputs unchanged.
"""

from .fallback import rms_norm as reference_rms_norm
from .fallback import silu_and_mul as reference_silu_and_mul


def _check_backend(backend):
    if backend not in {"torch", "cuda"}:
        raise ValueError("elementwise backend must be torch or cuda")


def rms_norm(x, weight, eps=1e-6, *, backend="torch"):
    _check_backend(backend)
    if backend == "torch":
        return reference_rms_norm(x, weight, eps)
    from .cuda_extension import load_cuda_extension
    return load_cuda_extension().rms_norm(x, weight, float(eps))


def residual_rms_norm(x, residual, weight, eps=1e-6, *, backend="torch"):
    """Return (RMSNorm(sum), sum), where sum is rounded to the input dtype."""
    _check_backend(backend)
    if x.shape != residual.shape:
        raise ValueError("x and residual must have the same shape")
    if backend == "torch":
        summed = x + residual
        return reference_rms_norm(summed, weight, eps), summed
    from .cuda_extension import load_cuda_extension
    normalized, summed = load_cuda_extension().residual_rms_norm(x, residual, weight, float(eps))
    return normalized, summed


def residual_add(x, residual, *, backend="torch"):
    _check_backend(backend)
    if x.shape != residual.shape:
        raise ValueError("x and residual must have the same shape")
    if backend == "torch":
        return x + residual
    from .cuda_extension import load_cuda_extension
    return load_cuda_extension().residual_add(x, residual)


def silu_and_mul(gate, up, *, backend="torch"):
    _check_backend(backend)
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    if backend == "torch":
        return reference_silu_and_mul(gate, up)
    from .cuda_extension import load_cuda_extension
    return load_cuda_extension().silu_and_mul(gate, up)
