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
    def __init__(self, model, cache, batch_size: int, max_blocks: int, *, padding_block_ids=None,
                 capture_sampling=False, sampling_backend="torch"):
        if batch_size <= 0 or max_blocks <= 0:
            raise ValueError("batch_size and max_blocks must be positive")
        if cache.device.type != "cuda":
            raise ValueError("CUDA Graph buckets require a CUDA cache")
        if sampling_backend not in {"torch", "cuda"}:
            raise ValueError("sampling backend must be torch or cuda")
        self.model = model
        self.cache = cache
        self.shape = GraphShape(batch_size, max_blocks)
        self.device = cache.device
        self.capture_sampling = bool(capture_sampling)
        self.sampling_backend = sampling_backend
        self._token_metadata = torch.zeros((2, batch_size), dtype=torch.long, device=self.device)
        self.input_ids, self.positions = self._token_metadata.unbind(0)
        self.block_tables = torch.full((batch_size, max_blocks), -1, dtype=torch.long, device=self.device)
        # Addresses are stable across requests and bucket replays. H2D copies
        # read these pinned buffers asynchronously; never overwrite them until
        # the event recorded after the previous H2D has completed.
        self._host_token_metadata = torch.empty((2, batch_size), dtype=torch.long, device="cpu", pin_memory=True)
        self.host_input_ids, self.host_positions = self._host_token_metadata.unbind(0)
        self.host_block_tables = torch.empty((batch_size, max_blocks), dtype=torch.long, device="cpu", pin_memory=True)
        try:
            # NumPy views avoid one PyTorch dispatch per Python integer. NumPy
            # is optional; the CPU scalar-assignment path preserves correctness
            # on minimal PyTorch installations without it.
            self._host_arrays = tuple(buffer.numpy() for buffer in
                                      (self.host_input_ids, self.host_positions, self.host_block_tables))
        except (RuntimeError, ImportError):
            self._host_arrays = None
        self._metadata_h2d_done = torch.cuda.Event(enable_timing=False)
        self._metadata_h2d_pending = False
        self._last_replay_stream = None
        self._staged_tables = None
        self.padding_block_ids = None
        self.padding_block_tables = None
        self._padding_host_ids = None
        if padding_block_ids is not None:
            self.set_padding(padding_block_ids)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: torch.Tensor | None = None
        self.sampled_token_ids: torch.Tensor | None = None

    def set_padding(self, padding_block_ids):
        self._staged_tables = None
        padding = torch.as_tensor(padding_block_ids, dtype=torch.long, device=self.device).flatten()
        if padding.numel() < self.shape.batch_size:
            raise ValueError("padding_block_ids must cover one private page per batch slot")
        host_ids = padding[:self.shape.batch_size].cpu().tolist()
        if len(set(host_ids)) != self.shape.batch_size:
            raise ValueError("padding slots require distinct scratch pages")
        if any(page < 0 for page in host_ids):
            raise ValueError("padding slots require nonnegative physical pages")
        cache_shape = getattr(self.cache, "shape", None)
        num_blocks = getattr(cache_shape, "num_blocks", None)
        if num_blocks is not None and any(page >= num_blocks for page in host_ids):
            raise ValueError("padding scratch page is outside the cache")
        self.padding_block_ids = padding.contiguous()
        self._padding_host_ids = host_ids
        self.padding_block_tables = torch.full((self.shape.batch_size, self.shape.max_blocks), -1,
                                              device=self.device, dtype=torch.long)
        # Padding always decodes position zero. Only column zero is accessed;
        # give each slot a private write target, including during capture.
        self.padding_block_tables[:, 0] = padding[:self.shape.batch_size]

    @staticmethod
    def validate_metadata(tokens, positions, block_tables, *, batch_size, max_blocks):
        """Validate Python metadata without requiring CUDA; return active rows.

        Rows may have different widths. Missing columns are reset to -1 on
        every replay, so shrinking/reordering requests cannot reuse stale pages.
        Device-specific vocabulary/context/page limits are checked by the
        bucket separately before any buffer is modified or GPU work enqueued.
        """
        if not all(isinstance(values, (list, tuple)) for values in (tokens, positions, block_tables)):
            raise ValueError("tokens, positions and block_tables must be Python sequences")
        active = len(tokens)
        if not 0 < active <= batch_size or len(positions) != active or len(block_tables) != active:
            raise ValueError("metadata rows must fit batch_size and agree on active count")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("tokens must contain nonnegative Python integers")
        if any(type(position) is not int or position < 0 for position in positions):
            raise ValueError("positions must contain nonnegative Python integers")
        for row in block_tables:
            if not isinstance(row, (list, tuple)) or not 0 < len(row) <= max_blocks:
                raise ValueError("each block table must have 1 to max_blocks entries")
            if any(type(page) is not int or page < -1 for page in row):
                raise ValueError("block table pages must be Python integers >= -1")
        return active

    def _validate_metadata_limits(self, tokens, positions, block_tables):
        config = getattr(self.model, "config", None)
        vocab_size = getattr(config, "vocab_size", None)
        context = getattr(config, "max_position_embeddings", None)
        if vocab_size is not None and any(token >= vocab_size for token in tokens):
            raise ValueError("token is outside the model vocabulary")
        if context is not None and any(position >= context for position in positions):
            raise ValueError("decode position exceeds model context")
        cache_shape = getattr(self.cache, "shape", None)
        block_size = getattr(cache_shape, "block_size", None)
        num_blocks = getattr(cache_shape, "num_blocks", None)
        for position, row in zip(positions, block_tables):
            if num_blocks is not None and any(page >= num_blocks for page in row):
                raise ValueError("physical page is outside the cache")
            if block_size is not None:
                required = position // block_size + 1
                if required > len(row) or any(page < 0 for page in row[:required]):
                    raise ValueError("block table does not cover the decode position")

    def _execution_stream(self):
        stream = torch.cuda.current_stream(self.device)
        if (self._last_replay_stream is not None
                and self._last_replay_stream.cuda_stream != stream.cuda_stream):
            # Reusing a bucket on a different stream is uncommon. Finish all
            # previously queued consumers too, not merely the graph itself.
            # Same-stream replays rely on stream ordering and never sync here.
            self._last_replay_stream.synchronize()
        self._last_replay_stream = stream
        return stream

    def _sample(self):
        if self.capture_sampling:
            from ..sampling import sample_next_token
            self.sampled_token_ids = sample_next_token(self.logits, temperature=0.0,
                                                       backend=self.sampling_backend)

    def _stage_metadata(self, tokens, positions, block_tables):
        if self._metadata_h2d_pending:
            self._metadata_h2d_done.synchronize()
            self._metadata_h2d_pending = False
        active = len(tokens)
        table_key = tuple(tuple(row) for row in block_tables)
        tables_changed = table_key != self._staged_tables
        if self._host_arrays is not None:
            host_tokens, host_positions, host_tables = self._host_arrays
            host_tokens.fill(0)
            host_positions.fill(0)
            host_tokens[:active] = tokens
            host_positions[:active] = positions
            if tables_changed:
                host_tables.fill(-1)
                host_tables[:, 0] = self._padding_host_ids
                for row, table in enumerate(block_tables):
                    host_tables[row, :len(table)] = table
        else:
            self.host_input_ids.zero_()
            self.host_positions.zero_()
            if tables_changed:
                self.host_block_tables.fill_(-1)
                for row, page in enumerate(self._padding_host_ids):
                    self.host_block_tables[row, 0] = page
            for row, (token, position, table) in enumerate(zip(tokens, positions, block_tables)):
                self.host_input_ids[row] = token
                self.host_positions[row] = position
                if tables_changed:
                    for column, page in enumerate(table):
                        self.host_block_tables[row, column] = page
        self._staged_tables = table_key
        return tables_changed

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
        self.block_tables[active_count:] = self.padding_block_tables[active_count:]

    @torch.inference_mode()
    def capture(self, *, warmup: int = 3) -> None:
        if self.graph is not None:
            return
        if warmup < 0:
            raise ValueError("warmup cannot be negative")
        if self.padding_block_ids is None:
            # Capture itself performs model writes. Without scratch pages it
            # could overwrite a live request's KV page before the first replay.
            raise ValueError("provide reserved padding_block_ids before capture")
        self._staged_tables = None
        self.block_tables.copy_(self.padding_block_tables)
        self.input_ids.zero_()
        self.positions.zero_()
        for _ in range(warmup):
            self.logits = self.model.forward_decode_batch(
                self.input_ids, self.cache, self.block_tables, self.positions
            )
            self._sample()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.logits = self.model.forward_decode_batch(
                self.input_ids, self.cache, self.block_tables, self.positions
            )
            self._sample()
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
        self._execution_stream()
        self._staged_tables = None
        self.input_ids[:active_count].copy_(input_ids)
        self.positions[:active_count].copy_(positions)
        self.block_tables[:active_count].copy_(block_tables)
        self._fill_padding(active_count)
        self.graph.replay()
        return self.logits[:active_count]

    @torch.inference_mode()
    def replay_metadata(self, tokens, positions, block_tables, *, sample=False):
        """Replay using Python lists and preallocated pinned/device metadata.

        Returns a GPU view of logits [active,vocab], or int64 greedy token IDs
        [active] when sample=True and sampling was captured. No logits D2H or
        graph-external argmax is needed for the latter path. The caller may
        copy token IDs to CPU once for scheduler/EOS handling.

        The returned view is overwritten by the next replay. Consume or clone
        it on the current stream before replaying again. Calls are serialized
        by the owning worker; a bucket is not a thread-safe concurrent API.
        Host staging is reused only after the preceding H2D event completes.
        """
        if self.graph is None or self.logits is None:
            raise RuntimeError("graph bucket has not been captured")
        if sample and (not self.capture_sampling or self.sampled_token_ids is None):
            raise ValueError("sample=True requires capture_sampling=True before graph capture")
        active = self.validate_metadata(tokens, positions, block_tables,
                                        batch_size=self.shape.batch_size, max_blocks=self.shape.max_blocks)
        self._validate_metadata_limits(tokens, positions, block_tables)
        if self._padding_host_ids is None:
            raise ValueError("metadata replay requires reserved padding_block_ids")
        stream = self._execution_stream()
        tables_changed = self._stage_metadata(tokens, positions, block_tables)
        self._token_metadata.copy_(self._host_token_metadata, non_blocking=True)
        if tables_changed:
            self.block_tables.copy_(self.host_block_tables, non_blocking=True)
        self._metadata_h2d_pending = True
        self._metadata_h2d_done.record(stream)
        self.graph.replay()
        return self.sampled_token_ids[:active] if sample else self.logits[:active]


class DecodeGraphManager:
    """Select the smallest captured bucket that can hold a decode batch."""

    def __init__(self, model, cache, *, batch_sizes=(1, 2, 4, 8, 16), block_buckets=(64, 128, 256, 512),
                 capture_sampling=False, sampling_backend="torch"):
        self.model = model
        self.cache = cache
        if sampling_backend not in {"torch", "cuda"}:
            raise ValueError("sampling backend must be torch or cuda")
        self.capture_sampling = bool(capture_sampling)
        self.sampling_backend = sampling_backend
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
                capture_sampling=self.capture_sampling, sampling_backend=self.sampling_backend,
            )
            self.buckets[key] = bucket
        elif padding_block_ids is not None and bucket.padding_block_ids is None:
            bucket.set_padding(padding_block_ids)
        return bucket

    def capture(self, batch_size: int, max_blocks: int, *, padding_block_ids=None, warmup: int = 3):
        bucket = self.get(batch_size, max_blocks, padding_block_ids=padding_block_ids)
        bucket.capture(warmup=warmup)
        return bucket
