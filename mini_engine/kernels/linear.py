"""Explicit Linear backend selection.

The CUDA path uses PyTorch's CUDA matmul dispatch (cuBLAS/cuBLASLt selected by
PyTorch). It is intentionally separate from the quantized reference backend;
this makes the performance backend visible without claiming a custom GEMM.
"""

import torch
from torch import nn


class CudaLinear(nn.Linear):
    """Inference Linear that makes the CUDA matmul boundary explicit."""

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


def make_linear(in_features, out_features, *, bias, backend="torch"):
    if backend == "torch":
        return nn.Linear(in_features, out_features, bias=bias)
    if backend == "cuda":
        return CudaLinear(in_features, out_features, bias=bias)
    raise ValueError("linear backend must be torch or cuda")
