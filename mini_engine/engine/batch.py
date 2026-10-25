"""Packed query metadata with independent request positions and paged KV."""

from dataclasses import dataclass, field
from numbers import Integral

import torch


def _integers(values, name):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype != torch.int64:
            raise ValueError(f"{name} must be a 1D int64 tensor")
        values = values.detach().cpu().tolist()
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of integers") from error
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in result):
        raise ValueError(f"{name} must contain integers")
    if any(value < -(1 << 63) or value >= (1 << 63) for value in result):
        raise ValueError(f"{name} values must fit int64")
    return tuple(int(value) for value in result)


def _signature(tensor):
    return (tensor.data_ptr(), tensor._version, tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


@dataclass(frozen=True)
class PrefillBatch:
    """Own read-only metadata for nonempty query chunks, never padded tokens.

    Use ``from_sequences``; exact Prefix hits with no new queries stay outside
    the packed model call. Request tables may share read-only prefix pages,
    but no page written by this batch may belong to another request's table.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    cu_seqlens_q: torch.Tensor
    query_starts: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    query_lengths: tuple[int, ...]
    query_offsets: tuple[int, ...]
    starts: tuple[int, ...]
    block_ids: tuple[tuple[int, ...], ...]
    block_size: int
    num_blocks: int
    _token_values: tuple[int, ...] = field(repr=False)
    _tensor_signatures: tuple = field(repr=False, compare=False)

    _TENSOR_NAMES = ("input_ids", "positions", "slot_mapping", "cu_seqlens_q",
                     "query_starts", "context_lens", "block_tables")

    @property
    def batch_size(self):
        return len(self.query_lengths)

    @property
    def num_tokens(self):
        return self.query_offsets[-1]

    @classmethod
    def from_sequences(cls, token_sequences, block_tables, query_starts, *,
                       block_size, num_blocks, device):
        for name, value in (("block_size", block_size), ("num_blocks", num_blocks)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        block_size, num_blocks = int(block_size), int(num_blocks)
        try:
            sequences = tuple(_integers(values, "token sequence") for values in token_sequences)
        except TypeError as error:
            raise ValueError("token_sequences must contain query sequences") from error
        if not sequences or any(not values for values in sequences):
            raise ValueError("prefill batch requires nonempty query sequences")
        starts = _integers(query_starts, "query_starts")
        if isinstance(block_tables, torch.Tensor):
            if block_tables.ndim != 2 or block_tables.dtype != torch.int64:
                raise ValueError("block_tables must be a 2D int64 tensor")
            block_tables = block_tables.detach().cpu().tolist()
        tables = tuple(_integers(values, "block table") for values in block_tables)
        if len(starts) != len(sequences) or len(tables) != len(sequences):
            raise ValueError("queries, query_starts and block_tables must have equal batch size")

        lengths, offsets, token_ids, positions, slots, contexts = [], [0], [], [], [], []
        clean_tables = []
        page_readers, page_writers, written_slots = {}, {}, set()
        # Complete validation on host values before creating device metadata.
        for request, (tokens, table, start) in enumerate(zip(sequences, tables, starts)):
            if start < 0:
                raise ValueError("query_starts must be nonnegative")
            if any(token < 0 for token in tokens):
                raise ValueError("token IDs must be nonnegative")
            first_padding = next((index for index, page in enumerate(table) if page == -1), len(table))
            if any(page != -1 for page in table[first_padding:]):
                raise ValueError("block table padding -1 must be trailing")
            table = table[:first_padding]
            if any(page < 0 or page >= num_blocks for page in table):
                raise ValueError("physical block ID out of range")
            if len(set(table)) != len(table):
                raise ValueError("a request block table cannot contain duplicate physical pages")
            end = start + len(tokens)
            if end > len(table) * block_size:
                raise ValueError("query range exceeds supplied block table capacity")
            for page in table:
                page_readers.setdefault(page, set()).add(request)
            for position in range(start, end):
                page = table[position // block_size]
                slot = page * block_size + position % block_size
                if slot in written_slots:
                    raise ValueError("prefill batch contains duplicate writable KV slots")
                written_slots.add(slot)
                page_writers.setdefault(page, set()).add(request)
                positions.append(position)
                slots.append(slot)
            lengths.append(len(tokens))
            offsets.append(offsets[-1] + len(tokens))
            token_ids.extend(tokens)
            contexts.append(end)
            clean_tables.append(table)
        for page, writers in page_writers.items():
            if len(page_readers[page] | writers) > 1:
                raise ValueError("prefill cannot write a physical page shared by requests")

        width = max(map(len, clean_tables))
        padded_tables = [table + (-1,) * (width - len(table)) for table in clean_tables]
        # Metadata created under inference_mode(False) keeps version counters.
        # A public forward can reject in-place mutation cheaply without a
        # device-to-host metadata copy before every layer or replay.
        with torch.inference_mode(False):
            tensors = tuple(torch.tensor(values, dtype=torch.int64, device=device) for values in
                            (token_ids, positions, slots, offsets, starts, contexts, padded_tables))
        return cls(*tensors, tuple(lengths), tuple(offsets), starts, tuple(clean_tables),
                   block_size, num_blocks, tuple(token_ids), tuple(map(_signature, tensors)))

    def validate(self, *, block_size, num_blocks, device, max_position_embeddings, vocab_size):
        """Validate the public execution boundary; layer wrappers can trust it."""
        if self.block_size != block_size or self.num_blocks != num_blocks:
            raise ValueError("prefill metadata does not match KV cache geometry")
        device = torch.device(device)
        tensors = tuple(getattr(self, name) for name in self._TENSOR_NAMES)
        if len(self._tensor_signatures) != len(tensors):
            raise ValueError("construct prefill metadata with PrefillBatch.from_sequences")
        if any(not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int64
               or tensor.device != device or not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("prefill metadata must be contiguous int64 tensors on the cache device")
        if tuple(map(_signature, tensors)) != self._tensor_signatures:
            raise ValueError("prefill metadata was modified; rebuild it with from_sequences")
        if any(start + length > max_position_embeddings for start, length in zip(self.starts, self.query_lengths)):
            raise ValueError("prefill query range exceeds model context limit")
        if any(token >= vocab_size for token in self._token_values):
            raise ValueError("prefill token ID exceeds model vocabulary")
