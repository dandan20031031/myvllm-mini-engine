"""Explicit Linear backend selection.

The CUDA path uses PyTorch's CUDA matmul dispatch (cuBLAS/cuBLASLt selected by
PyTorch). It is intentionally separate from the quantized reference backend;
this makes the performance backend visible without claiming a custom GEMM.
"""

import torch
from torch import nn

from .dandan_backend import DANDAN_LINEAR_BACKENDS, LINEAR_BACKENDS, require_dandan_ops


class CudaLinear(nn.Linear):
    """Inference Linear that makes the CUDA matmul boundary explicit."""

    backend = "cuda"

    def forward(self, x):
        if not x.is_cuda or not self.weight.is_cuda:
            raise ValueError("CudaLinear requires CUDA input and weights")
        original_shape = x.shape
        if x.shape[-1] != self.in_features:
            raise ValueError("input feature dimension does not match Linear")
        flat = x.reshape(-1, self.in_features)
        output = torch.mm(flat, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*original_shape[:-1], self.out_features)


def _dandan_linear(x, weight, bias, backend):
    if not x.is_cuda or not weight.is_cuda or x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("dandan Linear requires CUDA FP16 input and weights")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError("dandan Linear is inference-only; use torch.inference_mode()")
    ops = require_dandan_ops()
    output = ops.gemm(x, weight, weight_transposed=True,
                      backend="auto" if backend == "dandan" else "cuda")
    return output if bias is None else output + bias


class DandanLinear(nn.Linear):
    """Keep nn.Linear checkpoints and [N,K] weights; load the library on demand."""

    def __init__(self, in_features, out_features, *, bias=False, backend="dandan"):
        super().__init__(in_features, out_features, bias=bias)
        self.backend = backend

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("input feature dimension does not match Linear")
        output = _dandan_linear(x.reshape(-1, self.in_features), self.weight, self.bias, self.backend)
        return output.reshape(*x.shape[:-1], self.out_features)


class FusedLinear(nn.Module):
    """Concatenated output projection used to reduce small GEMM launches.

    The module keeps the same weight layout as ``nn.Linear`` but produces
    several adjacent output ranges with one matrix multiplication. It is
    inference-only in the current engine and supports both the reference
    PyTorch path and the explicit CUDA matmul path.
    """

    def __init__(self, in_features, out_features, *, bias=False, backend="torch", device=None, dtype=None):
        super().__init__()
        if backend not in LINEAR_BACKENDS:
            raise ValueError("unknown linear backend")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.backend = backend
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Fused modules are populated from existing Linear modules before use.
        # Keep a valid initializer for direct construction and tests.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            bound = self.in_features ** -0.5
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linears(cls, linears, *, backend="torch"):
        linears = tuple(linears)
        if not linears:
            raise ValueError("at least one Linear module is required")
        if any(not isinstance(linear, nn.Linear) for linear in linears):
            raise TypeError("fused projections require nn.Linear-compatible modules")
        in_features = linears[0].in_features
        if any(linear.in_features != in_features for linear in linears):
            raise ValueError("fused projections must share in_features")
        has_bias = linears[0].bias is not None
        if any((linear.bias is not None) != has_bias for linear in linears):
            raise ValueError("fused projections must agree on bias presence")
        weight = torch.cat([linear.weight.detach() for linear in linears], dim=0).contiguous()
        bias = None
        if has_bias:
            bias = torch.cat([linear.bias.detach() for linear in linears], dim=0).contiguous()
        fused = cls(
            in_features,
            weight.shape[0],
            bias=has_bias,
            backend=backend,
            device=weight.device,
            dtype=weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(weight)
            if bias is not None:
                fused.bias.copy_(bias)
        return fused

    def forward(self, x):
        if self.backend == "cuda" and (not x.is_cuda or not self.weight.is_cuda):
            raise ValueError("CUDA fused Linear requires CUDA input and weights")
        if x.shape[-1] != self.in_features:
            raise ValueError("input feature dimension does not match fused Linear")
        original_shape = x.shape
        flat = x.reshape(-1, self.in_features)
        if self.backend in DANDAN_LINEAR_BACKENDS:
            output = _dandan_linear(flat, self.weight, self.bias, self.backend)
        else:
            output = torch.mm(flat, self.weight.t())
            if self.bias is not None:
                output = output + self.bias
        return output.reshape(*original_shape[:-1], self.out_features)


def make_linear(in_features, out_features, *, bias, backend="torch"):
    if backend == "torch":
        return nn.Linear(in_features, out_features, bias=bias)
    if backend == "cuda":
        return CudaLinear(in_features, out_features, bias=bias)
    if backend in DANDAN_LINEAR_BACKENDS:
        return DandanLinear(in_features, out_features, bias=bias, backend=backend)
    raise ValueError("unknown linear backend")
