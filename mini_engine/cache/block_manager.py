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
        self._owned[request_id] = blocks
        return list(blocks)

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
        return list(current)

    def release(self, request_id: str) -> None:
        blocks = self._owned.pop(request_id, None)
        if blocks is None:
            return
        self._free.extend(blocks)

    def table(self, request_id: str) -> BlockTable:
        if request_id not in self._owned:
            raise KeyError(request_id)
        return BlockTable(request_id, tuple(self._owned[request_id]))
