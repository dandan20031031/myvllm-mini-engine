"""Physical paged KV cache storage.

The block manager owns *which* physical blocks belong to a request.  This
module owns the tensors stored in those blocks.  Keeping the two concerns
separate lets the same scheduler drive a PyTorch reference implementation or
an eventual CUDA paged-attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCacheShape:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int


class KVCachePool:
    """Preallocated key/value tensors indexed by physical block id.

    Tensor layout is ``[layer, block, token, kv_head, head_dim]``.  The layout
    is intentionally explicit and contiguous, making it straightforward to
    pass the tensors and block tables to a custom CUDA kernel later.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ):
        values = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        if any(value <= 0 for value in values):
            raise ValueError("all KV cache dimensions must be positive")
        self.shape = KVCacheShape(*values)
        self.dtype = dtype
        self.device = torch.device(device)
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.key = torch.empty(shape, dtype=dtype, device=self.device)
        self.value = torch.empty_like(self.key)

    @property
    def numel(self) -> int:
        return self.key.numel() * 2

    @property
    def memory_bytes(self) -> int:
        return self.numel * self.key.element_size()

    def _block_ids(self, block_ids):
        ids = torch.as_tensor(block_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() and bool(((ids < 0) | (ids >= self.shape.num_blocks)).any()):
            raise IndexError("physical block id out of range")
        return ids

    def clear(self, block_ids: list[int] | tuple[int, ...] | None = None) -> None:
        """Zero all blocks or only blocks returned to the allocator."""
        if block_ids is None:
            self.key.zero_()
            self.value.zero_()
            return
        if not block_ids:
            return
        ids = self._block_ids(block_ids)
        # ``tensor[:, ids]`` is advanced indexing and returns a copy.  Use
        # index_fill_ so the selected physical blocks are cleared in place.
        self.key.index_fill_(1, ids, 0)
        self.value.index_fill_(1, ids, 0)

    def write(
        self,
        layer: int,
        block_ids: torch.Tensor | list[int],
        token_offset: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write a contiguous token range into one or more physical blocks.

        ``key`` and ``value`` have shape ``[tokens, kv_heads, head_dim]``.
        The range may cross block boundaries but may not exceed the supplied
        block table.
        """
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("key and value must both have shape [tokens, kv_heads, head_dim]")
        if key.shape[1:] != (self.shape.num_kv_heads, self.shape.head_dim):
            raise ValueError("KV tensor shape does not match cache configuration")
        if layer < 0 or layer >= self.shape.num_layers:
            raise IndexError("layer out of range")
        if token_offset < 0:
            raise ValueError("token_offset cannot be negative")
        ids = self._block_ids(block_ids)
        capacity = ids.numel() * self.shape.block_size
        if token_offset + key.shape[0] > capacity:
            raise ValueError("write exceeds supplied block table capacity")
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        remaining = key.shape[0]
        source = 0
        position = token_offset
        while remaining:
            block_index, in_block = divmod(position, self.shape.block_size)
            count = min(remaining, self.shape.block_size - in_block)
            block_id = ids[block_index]
            self.key[layer, block_id, in_block : in_block + count].copy_(key[source : source + count])
            self.value[layer, block_id, in_block : in_block + count].copy_(value[source : source + count])
            source += count
            position += count
            remaining -= count

    def gather(
        self,
        layer: int,
        block_ids: torch.Tensor | list[int],
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the logical prefix represented by a block table."""
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        if layer < 0 or layer >= self.shape.num_layers:
            raise IndexError("layer out of range")
        ids = self._block_ids(block_ids)
        if token_count > ids.numel() * self.shape.block_size:
            raise ValueError("token_count exceeds supplied block table capacity")
        if token_count == 0:
            empty_shape = (0, self.shape.num_kv_heads, self.shape.head_dim)
            return (torch.empty(empty_shape, dtype=self.dtype, device=self.device),) * 2
        key = self.key[layer, ids].reshape(-1, self.shape.num_kv_heads, self.shape.head_dim)[:token_count]
        value = self.value[layer, ids].reshape(-1, self.shape.num_kv_heads, self.shape.head_dim)[:token_count]
        return key, value
