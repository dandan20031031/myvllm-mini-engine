"""Complete-block prefix cache with reference-counted physical KV pages."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from .block_manager import BlockManager


@dataclass
class PrefixEntry:
    token_key: tuple[int, ...]
    block_ids: tuple[int, ...]
    next_logits: torch.Tensor | None = None


class PrefixCache:
    """LRU cache for full token blocks.

    Only complete blocks are shared.  The cache itself owns one reference to
    every physical page; each request gets another reference while running.
    """

    def __init__(self, block_manager: BlockManager, *, max_entries: int = 128):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.block_manager = block_manager
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], PrefixEntry] = OrderedDict()

    def lookup(self, tokens: list[int] | tuple[int, ...]) -> PrefixEntry | None:
        token_tuple = tuple(int(token) for token in tokens)
        best = None
        for key, entry in self._entries.items():
            if len(key) <= len(token_tuple) and token_tuple[: len(key)] == key:
                if best is None or len(key) > len(best.token_key):
                    best = entry
        if best is not None:
            self._entries.move_to_end(best.token_key)
        return best

    def attach(self, request_id: str, entry: PrefixEntry) -> list[int]:
        return self.block_manager.attach(request_id, entry.block_ids)

    def insert(
        self,
        tokens: list[int] | tuple[int, ...],
        block_ids: list[int] | tuple[int, ...],
        *,
        next_logits: torch.Tensor | None = None,
    ) -> PrefixEntry | None:
        complete_tokens = (len(tokens) // self.block_manager.block_size) * self.block_manager.block_size
        block_count = complete_tokens // self.block_manager.block_size
        if block_count <= 0:
            return None
        key = tuple(int(token) for token in tokens[:complete_tokens])
        physical = tuple(int(block) for block in block_ids[:block_count])
        old = self._entries.pop(key, None)
        if old is not None:
            self.block_manager.unpin(old.block_ids)
        self.block_manager.pin(physical)
        entry = PrefixEntry(
            token_key=key,
            block_ids=physical,
            next_logits=None if next_logits is None else next_logits.detach().clone(),
        )
        self._entries[key] = entry
        while len(self._entries) > self.max_entries:
            _, evicted = self._entries.popitem(last=False)
            self.block_manager.unpin(evicted.block_ids)
        return entry

    def find_longest(self, tokens: list[int] | tuple[int, ...]) -> PrefixEntry | None:
        return self.lookup(tokens)

    def clear(self) -> None:
        for entry in self._entries.values():
            self.block_manager.unpin(entry.block_ids)
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
