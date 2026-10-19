# 阶段 09+10：Continuous Batching Runtime 与 Prefix Block Cache

## 1. 完成内容

本阶段把 Scheduler 接到 NativeQwen3 model execution，并加入完整物理 block 的 Prefix Cache：

- decode 阶段一次处理多个请求；
- batch block table、positions 和 token 输入；
- Prefix Cache 只共享完整物理 block；
- BlockManager 增加引用计数、pin/unpin 和共享 attach；
- LRU 淘汰时保护仍被请求使用的 KV page。

prefill 当前仍逐请求执行，Prefix Cache 当前完成存储复用和生命周期保护，但尚未凭 next logits 跳过完整 prompt 计算。

## 2. 主要调用链

```text
NativeBatchRuntime.step
  -> prefix lookup
  -> Scheduler.schedule
  -> prefill_step(new requests)
  -> decode_step(active requests)
       -> input_ids [batch]
       -> positions [batch]
       -> block_tables [batch, max_blocks]
       -> NativeQwen3.forward_decode_batch
       -> batch logits / greedy sampling
       -> Scheduler.on_token / grow / finish / release
```

`NativeQwen3.forward_decode_batch` 在每层执行 Q/K/V、RoPE、batch KV scatter、Paged Attention、elementwise CUDA 和 logits，多个请求共享一次模型 forward。

## 3. Prefix Cache

`PrefixCache` 只缓存完整 block。一个长度不是 block size 整数倍的 prompt，最后一个不完整 block 不会被共享。

```text
cache entry = token_key + physical block ids + optional next_logits
```

查找选择最长匹配 token 前缀，并将 entry 移到 LRU 尾部。

引用计数规则：

```text
allocate          -> refcount 1
PrefixCache.pin   -> +1
request attach    -> +1
request release   -> -1
LRU evict         -> -1
refcount == 0     -> free list
```

这样请求释放时不会误回收仍被 Prefix Cache 使用的物理页。

## 4. Batch KV 写入

`write_kv_cache_batch` 支持：

```text
key/value     [batch, kv_heads, head_dim]
block_tables  [batch, max_blocks]
offsets       [batch]
```

每个 CUDA CTA 处理一个 request，按该 request 的 logical offset 直接定位 physical page。torch fallback 和 CUDA backend 保持同一调用接口。

## 5. 测试结果

新增测试覆盖：

- Prefix Cache page 共享；
- request release 不会回收 cache pin page；
- LRU 淘汰回收 page；
- 两请求 prefill admission；
- 两请求 decode batch；
- Prefix hit block table 复用。

全量回归：

```text
139 passed, 2 warnings
```

## 6. 当前限制

- prefill 仍逐请求；
- batch sampler 当前统一 greedy，尚未融合每请求 temperature/top-k/top-p；
- Prefix Cache 尚未跳过 prompt 计算；
- 服务级取消和客户端断连尚未实现；
- 尚未完成大并发 TTFT/TPOT benchmark。

## 7. 下一步

阶段 11 实现 CUDA Graph decode：固定 batch shape bucket、预分配 metadata buffer、capture/replay，并测量对 TPOT 的收益。随后完善 prefix-aware prefill，形成真实 Prefix Cache 加速。
