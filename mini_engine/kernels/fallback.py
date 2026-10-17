"""Reference PyTorch kernels used before custom CUDA extensions are available."""

import math

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply precomputed rotary embeddings to tensors ending in head_dim."""
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_out = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
    k_out = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
    return q_out, k_out


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(gate) * up


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Correctness-first decode attention reference.

    Shapes: query [B, Hq, D], caches [num_blocks, block_size, Hkv, D],
    block_tables [B, max_blocks]. GQA/MQA are supported when Hq is divisible
    by Hkv.  A CUDA kernel will replace this function without changing its API.
    """
    outputs = []
    if query.ndim != 3 or key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("invalid query or cache shape")
    if key_cache.shape[1] != block_size or query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("block size or head dimension mismatch")
    if query.shape[1] % key_cache.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    scale = 1.0 / math.sqrt(query.shape[-1])
    for batch in range(query.shape[0]):
        length = int(context_lens[batch])
        if length <= 0 or length > block_tables.shape[1] * block_size:
            raise ValueError("invalid context length")
        ids = block_tables[batch, : math.ceil(length / block_size)]
        keys = key_cache[ids].reshape(-1, key_cache.shape[2], key_cache.shape[3])[:length]
        values = value_cache[ids].reshape(-1, value_cache.shape[2], value_cache.shape[3])[:length]
        group = query.shape[1] // keys.shape[1]
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", query[batch].float(), keys.float()) * scale
        # Keep probability and PV accumulation in FP32, matching the CUDA
        # kernel; rounding probabilities to BF16 before PV amplifies error.
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probs, values.float()).to(query.dtype))
    return torch.stack(outputs)
