from dataclasses import dataclass


@dataclass(frozen=True)
class BlockTable:
    request_id: str
    block_ids: tuple[int, ...]


class BlockManager:
    """Fixed-size KV block allocator.

    The allocator is deliberately independent from tensors.  A later CUDA
    KV-cache pool can use the returned physical ids as rows in its cache
    storage without changing the scheduler API.
    """

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._owned: dict[str, list[int]] = {}
        self._refcounts: list[int] = [0] * num_blocks

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    def blocks_required(self, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count cannot be negative")
        return (token_count + self.block_size - 1) // self.block_size

    def can_allocate(self, token_count: int) -> bool:
        return self.blocks_required(token_count) <= self.free_blocks

    def allocate(self, request_id: str, token_count: int) -> list[int]:
        if request_id in self._owned:
            raise ValueError(f"request already owns blocks: {request_id}")
        count = self.blocks_required(token_count)
        if count > self.free_blocks:
            raise MemoryError(f"need {count} blocks, only {self.free_blocks} free")
        blocks = [self._free.pop() for _ in range(count)]
        for block in blocks:
            self._refcounts[block] = 1
        self._owned[request_id] = blocks
        return list(blocks)

    def allocate_with_prefix(
        self,
        request_id: str,
        token_count: int,
        prefix_block_ids: list[int] | tuple[int, ...],
    ) -> list[int]:
        """Attach cached prefix pages and allocate only the suffix pages."""
        if request_id in self._owned:
            raise ValueError(f"request already owns blocks: {request_id}")
        prefix = list(prefix_block_ids)
        required = self.blocks_required(token_count)
        if len(prefix) > required:
            raise ValueError("prefix has more blocks than the request needs")
        for block in prefix:
            if block < 0 or block >= self.num_blocks or self._refcounts[block] <= 0:
                raise ValueError(f"cannot attach unallocated block: {block}")
        suffix_count = required - len(prefix)
        if suffix_count > self.free_blocks:
            raise MemoryError(f"need {suffix_count} blocks, only {self.free_blocks} free")
        for block in prefix:
            self._refcounts[block] += 1
        suffix = [self._free.pop() for _ in range(suffix_count)]
        for block in suffix:
            self._refcounts[block] = 1
        self._owned[request_id] = prefix + suffix
        return list(self._owned[request_id])

    def attach(self, request_id: str, block_ids: list[int] | tuple[int, ...]) -> list[int]:
        """Attach retained shared blocks to a request."""
        if request_id in self._owned:
            raise ValueError(f"request already owns blocks: {request_id}")
        blocks = list(block_ids)
        for block in blocks:
            if block < 0 or block >= self.num_blocks or self._refcounts[block] <= 0:
                raise ValueError(f"cannot attach unallocated block: {block}")
        for block in blocks:
            self._refcounts[block] += 1
        self._owned[request_id] = blocks
        return list(blocks)

    def pin(self, block_ids: list[int] | tuple[int, ...]) -> None:
        """Hold one external reference, for example a prefix cache entry."""
        for block in block_ids:
            if block < 0 or block >= self.num_blocks or self._refcounts[block] <= 0:
                raise ValueError(f"cannot pin unallocated block: {block}")
        for block in block_ids:
            self._refcounts[block] += 1

    def unpin(self, block_ids: list[int] | tuple[int, ...]) -> None:
        for block in block_ids:
            self._decrement(block)

    def _decrement(self, block: int) -> None:
        if self._refcounts[block] <= 0:
            raise ValueError(f"block is not retained: {block}")
        self._refcounts[block] -= 1
        if self._refcounts[block] == 0:
            self._free.append(block)

    def grow(self, request_id: str, token_count: int) -> list[int]:
        if request_id not in self._owned:
            return self.allocate(request_id, token_count)
        current = self._owned[request_id]
        needed = self.blocks_required(token_count) - len(current)
        if needed <= 0:
            return list(current)
        if needed > self.free_blocks:
            raise MemoryError(f"need {needed} additional blocks, only {self.free_blocks} free")
        current.extend(self._free.pop() for _ in range(needed))
        for block in current[-needed:]:
            self._refcounts[block] = 1
        return list(current)

    def release(self, request_id: str) -> None:
        blocks = self._owned.pop(request_id, None)
        if blocks is None:
            return
        for block in blocks:
            self._decrement(block)

    def refcount(self, block_id: int) -> int:
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(block_id)
        return self._refcounts[block_id]

    def table(self, request_id: str) -> BlockTable:
        if request_id not in self._owned:
            raise KeyError(request_id)
        return BlockTable(request_id, tuple(self._owned[request_id]))
