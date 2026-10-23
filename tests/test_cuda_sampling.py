import os

import pytest
import torch

from mini_engine.sampling import sample_next_token

pytestmark = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="CUDA toolchain opt-in required",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [1, 257, 4097, 151936])
def test_greedy_ties_nan_and_tail(dtype, vocab):
    torch.manual_seed(29)
    x = torch.randn(5, vocab, device="cuda", dtype=dtype)
    x[0].fill_(float("-inf"))
    x[1].fill_(1)
    x[2, -1] = float("inf")
    x[3, -1] = float("nan")
    x[3, 0] = float("nan")
    x[4, -1] = 99
    assert torch.equal(sample_next_token(x, backend="cuda"), x.argmax(-1))


def test_greedy_graph_replay_and_validation():
    x = torch.randn(3, 8193, device="cuda", dtype=torch.float16)
    sample_next_token(x, backend="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = sample_next_token(x, backend="cuda")
    for index in [0, 8192, 4096]:
        x.fill_(0)
        x[:, index] = 2
        graph.replay()
        assert torch.equal(y, x.argmax(-1))
    with pytest.raises(ValueError, match="greedy"):
        sample_next_token(x, temperature=1, backend="cuda")
    with pytest.raises(RuntimeError, match="contiguous CUDA"):
        sample_next_token(x.cpu(), backend="cuda")
    assert sample_next_token(x[:0], backend="cuda").numel() == 0
