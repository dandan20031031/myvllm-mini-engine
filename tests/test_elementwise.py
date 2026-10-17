"""Torch fallback contract is available without building a CUDA extension."""

import pytest
import torch

from mini_engine.kernels.elementwise import residual_add, residual_rms_norm, rms_norm, silu_and_mul


def test_reference_fused_outputs_and_inputs_unchanged():
    x = torch.randn(3, 17)
    residual = torch.randn_like(x)
    weight = torch.randn(17)
    before = x.clone(), residual.clone()
    normalized, summed = residual_rms_norm(x, residual, weight)
    torch.testing.assert_close(summed, x + residual)
    torch.testing.assert_close(normalized, rms_norm(x + residual, weight))
    torch.testing.assert_close(residual_add(x, residual), summed)
    assert torch.equal(x, before[0]) and torch.equal(residual, before[1])


def test_unknown_backend_and_mismatched_pair_shapes():
    x = torch.ones(2, 3)
    with pytest.raises(ValueError, match="backend"):
        rms_norm(x, torch.ones(3), backend="invalid")
    for op in (residual_add, silu_and_mul):
        with pytest.raises(ValueError, match="same shape"):
            op(x, torch.ones(2, 4))
