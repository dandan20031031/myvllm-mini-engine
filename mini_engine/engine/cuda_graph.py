"""Shape-bucketed CUDA Graph decode runner.

The scheduler stays on CPU. A bucket owns fixed-address GPU buffers and a
captured one-token batched decode graph. Padding slots require caller-provided
scratch KV pages so an inactive graph slot cannot overwrite a live request.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphShape:
    batch_size: int
    max_blocks: int


class DecodeGraphBucket:
    def __init__(self, model, cache, batch_size: int, max_blocks: int, *, padding_block_ids=None):
        if batch_size <= 0 or max_blocks <= 0:
            raise ValueError("batch_size and max_blocks must be positive")
        if cache.device.type != "cuda":
            raise ValueError("CUDA Graph buckets require a CUDA cache")
        self.model = model
        self.cache = cache
        self.shape = GraphShape(batch_size, max_blocks)
        self.device = cache.device
        self.input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.block_tables = torch.full((batch_size, max_blocks), -1, dtype=torch.long, device=self.device)
        self.padding_block_ids = None
        if padding_block_ids is not None:
            padding = torch.as_tensor(padding_block_ids, dtype=torch.long, device=self.device).flatten()
            if padding.numel() < max_blocks:
                raise ValueError("padding_block_ids must cover max_blocks")
            self.padding_block_ids = padding[:max_blocks].contiguous()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None

    @property
    def captured(self) -> bool:
        return self.graph is not None

    def _fill_padding(self, active_count: int) -> None:
        if active_count == self.shape.batch_size:
            return
        if self.padding_block_ids is None:
            raise ValueError("padded replay requires scratch padding_block_ids")
        self.input_ids[active_count:].zero_()
        self.positions[active_count:].zero_()
        self.block_tables[active_count:] = self.padding_block_ids

    @torch.inference_mode()
    def capture(self, *, warmup: int = 3) -> None:
        if self.graph is not None:
            return
        if warmup < 0:
            raise ValueError("warmup cannot be negative")
        if self.padding_block_ids is None and self.shape.batch_size > 1:
            # Capture is safe with full buckets. Padded slots need explicit
            # scratch pages; silently aliasing live pages would corrupt KV.
            raise ValueError("provide padding_block_ids before capturing a padded bucket")
        self.block_tables.zero_()
        for _ in range(warmup):
            self.logits = self.model.forward_decode_batch(
                self.input_ids, self.cache, self.block_tables, self.positions
            )
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.logits = self.model.forward_decode_batch(
                self.input_ids, self.cache, self.block_tables, self.positions
            )
        self.graph = graph

    @torch.inference_mode()
    def replay(self, input_ids, positions, block_tables, *, active_count: int | None = None):
        if self.graph is None or self.logits is None:
            raise RuntimeError("graph bucket has not been captured")
        active_count = self.shape.batch_size if active_count is None else active_count
        if not 0 < active_count <= self.shape.batch_size:
            raise ValueError("active_count must be in (0, batch_size]")
        if input_ids.shape != (active_count,) or positions.shape != (active_count,):
            raise ValueError("input_ids/positions do not match active_count")
        if block_tables.shape != (active_count, self.shape.max_blocks):
            raise ValueError("block_tables do not match bucket max_blocks")
        self.input_ids[:active_count].copy_(input_ids)
        self.positions[:active_count].copy_(positions)
        self.block_tables[:active_count].copy_(block_tables)
        self._fill_padding(active_count)
        self.graph.replay()
        return self.logits[:active_count]


class DecodeGraphManager:
    """Select the smallest captured bucket that can hold a decode batch."""

    def __init__(self, model, cache, *, batch_sizes=(1, 2, 4, 8, 16), block_buckets=(64, 128, 256, 512)):
        self.model = model
        self.cache = cache
        self.batch_sizes = tuple(sorted(set(int(size) for size in batch_sizes if size > 0)))
        self.block_buckets = tuple(sorted(set(int(size) for size in block_buckets if size > 0)))
        if not self.batch_sizes or not self.block_buckets:
            raise ValueError("at least one graph bucket is required")
        self.buckets: dict[GraphShape, DecodeGraphBucket] = {}

    def shape_for(self, batch_size: int, max_blocks: int) -> GraphShape:
        batch = next((size for size in self.batch_sizes if size >= batch_size), None)
        blocks = next((size for size in self.block_buckets if size >= max_blocks), None)
        if batch is None or blocks is None:
            raise ValueError("request exceeds configured decode graph buckets")
        return GraphShape(batch, blocks)

    def get(self, batch_size: int, max_blocks: int, *, padding_block_ids=None) -> DecodeGraphBucket:
        shape = self.shape_for(batch_size, max_blocks)
        key = shape
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = DecodeGraphBucket(
                self.model, self.cache, shape.batch_size, shape.max_blocks,
                padding_block_ids=padding_block_ids,
            )
            self.buckets[key] = bucket
        elif padding_block_ids is not None and bucket.padding_block_ids is None:
            bucket.padding_block_ids = torch.as_tensor(
                padding_block_ids, dtype=torch.long, device=self.cache.device
            )[: shape.max_blocks].contiguous()
        return bucket

    def capture(self, batch_size: int, max_blocks: int, *, padding_block_ids=None, warmup: int = 3):
        bucket = self.get(batch_size, max_blocks, padding_block_ids=padding_block_ids)
        bucket.capture(warmup=warmup)
        return bucket
