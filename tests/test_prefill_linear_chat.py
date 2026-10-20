import pytest
import torch
from torch.nn import functional as F

from mini_engine.kernels.linear import CudaLinear, make_linear
from mini_engine.kernels.prefill import prefill_attention


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_cuda_matches_sdpa(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(71)
    q = torch.randn(1, 4, 17, 16, device="cuda", dtype=dtype)
    k = torch.randn(1, 2, 17, 16, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    expected = prefill_attention(q, k, v, "math")
    actual = prefill_attention(q, k, v, "cuda")
    torch.testing.assert_close(actual, expected, atol=0.012 if dtype == torch.float16 else 0.05, rtol=0.03)


def test_linear_backend_contract_cpu_and_cuda():
    torch.manual_seed(72)
    module = make_linear(7, 5, bias=True, backend="torch")
    x = torch.randn(2, 3, 7)
    expected = module(x)
    assert expected.shape == (2, 3, 5)
    with pytest.raises(ValueError, match="CUDA"):
        CudaLinear(7, 5)(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_linear_matches_torch_linear():
    torch.manual_seed(73)
    ref = torch.nn.Linear(7, 5, bias=True, device="cuda", dtype=torch.bfloat16).eval()
    actual = CudaLinear(7, 5, bias=True, device="cuda", dtype=torch.bfloat16).eval()
    actual.load_state_dict(ref.state_dict())
    x = torch.randn(2, 3, 7, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(actual(x), ref(x), atol=0.004, rtol=0.01)
