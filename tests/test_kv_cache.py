import torch

from mini_engine.cache.kv_cache import KVCachePool


def test_kv_cache_write_crosses_block_boundary():
    cache = KVCachePool(1, 3, 4, 2, 2, dtype=torch.float32)
    key = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    value = key + 100
    cache.write(0, [2, 0], 3, key, value)
    got_key, got_value = cache.gather(0, [2, 0], 6)
    assert torch.equal(got_key[3:6], key)
    assert torch.equal(got_value[3:6], value)


def test_kv_cache_memory_and_clear():
    cache = KVCachePool(2, 4, 8, 2, 16, dtype=torch.float16)
    assert cache.memory_bytes == 2 * 4 * 8 * 2 * 16 * 2 * 2
    cache.key.fill_(1)
    cache.value.fill_(1)
    cache.clear([1])
    assert torch.all(cache.key[:, 1] == 0)
    assert torch.all(cache.value[:, 0] == 1)
