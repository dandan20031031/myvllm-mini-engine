# 阶段 23：真正复用 Prefix KV 与有预算的 Chunked Prefill

本阶段将 Prefix Cache 从共享物理页提升为跳过命中前缀的模型计算，并让长 prompt 在 token 预算内分块执行。每轮先运行已有请求的 decode，再执行剩余预算允许的 prefill。Windows 原生路径与已有 Torch、CUDA、dandan 后端保持兼容，没有修改独立算子库。

## 1. 实现了什么

- 完全命中且具有正确边界 logits 的 prompt：不调用模型，直接采样首个输出 token。
- 命中完整 block 前缀、后面还有 suffix：从 `position=prefix_len` 开始计算新 token，不重写共享页。
- 非 block 对齐 prompt：缓存完整 block 的 KV，并保存**对应 block 末尾行**的 logits，不把整个 prompt 末尾 logits 放到较短 key 上。
- Chunked Prefill：`prefill_chunk_size` 限制单请求单轮 prefill token 数；`max_num_batched_tokens` 限制本轮 decode token 数和 prefill token 数之和。
- 单请求的任意非空多 token suffix forward：为分块 prefill 和后续投机验证提供同一个模型接口。
- decode 先于 prefill 实际执行，并在 decode 请求数超过预算时轮转。
- 按请求最大实际 KV 需求做虚拟预留，实际页仍按需分配，避免并发请求互相占满池后都无法继续 grow。
- LRU eviction 只释放缓存引用，不释放其他请求持有的页；入队前得到的 prefix 元数据会在 admission 时刷新，避免 stale 页号。
- 最终输出 token 和 EOS 后不再 grow 无用页；abort、forward/sampling 异常释放请求引用。
- `enable_prefix_cache=False` 正式开关，以及各请求独立采样参数。
- `BlockManager.truncate(request_id, keep_tokens)`，供投机拒绝后释放完整尾页。

## 2. API

```python
scheduler = Scheduler(
    block_manager,
    max_num_batched_tokens=128,
    prefill_chunk_size=32,
)
runtime = NativeBatchRuntime(
    model,
    scheduler,
    kv_cache,
    prefill_chunk_size=32,
    enable_prefix_cache=True,
    sampling_backend="torch",  # 可选 CUDA greedy；由采样后端校验支持范围
)

scheduler.add_request(GenerationRequest("r0", prompt_ids, max_new_tokens=64))
while scheduler.waiting or scheduler.running:
    newly_emitted = runtime.step()  # {request_id: token_id}
```

`prefill_chunk_size=None` 对预算内的短 prompt 继续一次完整 prefill。超过全局预算的 prompt 会分轮推进，避免原先一直等待。没有输出 token 的 prefill 轮次可返回空字典；此时 `request.prefill_cursor` 会前进。

```python
logits = model(suffix_ids, cache, block_ids, position=existing_prefix_tokens)
```

返回 `[suffix_length, vocab_size]`，第 i 行对应绝对位置 `position+i`。调用者须保证前缀 KV 已物化、块表容量足够。

## 3. 每个代码文件及调用逻辑

### `mini_engine/engine/request.py`

`GenerationRequest` 增加本轮 `prefill_start/end` 与累计 `prefill_cursor`，记录只计算了 prompt 的哪一段。`prefix_next_logits` 持有已命中边界的最后一行 logits。

`max_cache_tokens = prompt_len + max_new_tokens - 1`。生成的最后一个 token 只需返回给用户，不需要再送入模型，因此没有对应的 KV 写入。

cursor 在 schedule 预留该轮范围时前进。执行器必须先完成返回的工作再调下一轮；若执行失败，runtime 会 abort 该轮未成功执行的相关请求，避免将未计算范围当成有效 KV。

### `mini_engine/cache/prefix_cache.py`

`insert` 将 key 截到完整 block 边界，验证块表覆盖且页号不重复。默认传入的 `next_logits` 被认为属于 `len(tokens)`，只有该位置与完整 key 末尾一致时才保留。显式传 `logits_position` 可说明 logits 属于哪个边界。

runtime 的常规做法是直接传边界 token prefix 和 forward 返回的对应行，例如 chunk 从 3 算到 7、block size=4，就缓存 token 0..3 与本轮 row 0 的 logits，而不是 row 3。

新 entry 先准备 logits、pin 新页，再替换旧 entry 并 unpin 旧页；避免同页替换时先降到零引用导致错误。

`evict_until_free(minimum_free, protected_keys=...)` 从 LRU 头部逐个移除 cache pin。页仍有请求引用时不会进入 free list。当前准备 admission 的命中 entry 会受到保护。

### `mini_engine/cache/block_manager.py`

已有 allocate、attach、grow、pin/unpin 继续管理独立引用。本阶段为 prefix/attach 增加重复物理页检查，并补充 `truncate`：只释放逻辑尾部的完整页，保留最后一页中被拒绝 token 的旧内容，下次 forward 在原位置覆盖。

`truncate` 不清零被保留页，也不撤销 Prefix Cache 或其他请求持有的引用。

### `mini_engine/engine/scheduler.py`

每轮执行顺序：

1. 为 decode-ready 请求保留一 token 预算，先选择已有流；必要时轮转。
2. 用剩余预算推进已经运行但 prompt 尚未完成的请求。
3. 检查 waiting 请求的最新 prefix 命中、最大 KV 虚拟预留，再 admission 新请求。

虚拟预留计算每个运行请求的“最大总页数减当前持有页数”。新请求必须在物理 free 页加缓存按需 eviction 后，仍能容纳所有已有预留及自己的未来增长，才允许 admission。该策略比超额接入更保守，但保证已接入流有继续生成的空间。

单请求预算本身超过物理池，或扣除 graph scratch 等保留页后永久不可能放入时，会明确报 `MemoryError`，不会无限返回空轮次。遇暂时资源不足则等待已有运行流释放预留。若实际 EOS 提前发生，虚拟预留随请求结束立即释放。

`on_token` 先判断 EOS/长度结束，结束则释放请求页；只有将来还会送回模型的 token 才 grow。

### `mini_engine/model/native_qwen3.py`

`forward(..., position=n)` 接受非空任意长度 suffix。每层先在绝对位置 n 写入新 K/V：

- Q length=1 且 position>0：仍然调用 PagedAttention，直接读分页 KV。
- Q length>1 且 position>0：gather 该层已有前缀加本轮 suffix，调用带 `query_start=n` 的 dense attention。
- position=0：普通完整/首块 prefill。

完整 prefix 页由 runtime 跳过，forward 只从未缓存位置写入。suffix 的 gather 是 correctness/复用接口，具有额外内存与复制开销；没有把常规 decode 改成全量 gather。

### `mini_engine/kernels/prefill.py`

`prefill_attention(..., query_start=0)` 描述 query 第一行的绝对 KV 位置。math/auto 的矩形 attention 使用显式 offset causal mask，避免 SDPA 的 `is_causal=True` 将 suffix 第一行错误地当成绝对位置零。

- `dandan` 将原生 GQA 和 `query_start` 交给算子库，不扩展 KV heads。
- `cuda` 调用项目的 `causal_prefill_suffix`，保留原生 GQA，不回退 Torch。
- Torch math/auto 分支才显式扩展 GQA。

项目自有 CUDA suffix kernel 与扩展绑定由本阶段并行任务补齐，保留旧的三参数 square prefill 入口。

### `mini_engine/engine/runtime.py`

`step` 调用 prepare → schedule → decode → prefill。admission 时再次查询 PrefixCache，防止同一轮更早的 cache eviction 使 waiting 请求持有旧页号。

`prefill_step` 对计划范围 `[start:end]` 调用模型；保存该 chunk 内最后一个完整 block 的 KV 和正确边界 logits。尚未完成整个 prompt 时不采样。完全命中时范围为空，使用缓存 logits。完整共享页绝不写回。

`decode_step` 仍支持 eager 或现有 DecodeGraphBucket。采样按每个请求的 temperature/top_k/top_p 执行，统一参数请求可一次 batch sample。结果用一次 `.cpu().tolist()` 搬回 CPU，避免逐行 `.item()` 同步。

传入一个空 PrefixCache 对象也会被保留；修复原先 `prefix_cache or PrefixCache(...)` 因 `__len__` 为零而替换对象的问题。

## 4. 验证

本任务执行的 CPU 命令（所有缓存位于项目 D 盘）：

```powershell
. .\scripts\enter_project.ps1
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MINI_ENGINE_TEST_CUDA = '0'
.\.venv\Scripts\python -m pytest -q `
  tests/test_chunked_prefix.py tests/test_core.py tests/test_runtime_prefix.py `
  -p no:cacheprovider --basetemp build/tmp/pytest_stage23_cpu
```

初次CPU结果：**24 passed，4 skipped**。其中新文件包含18个CPU回归与4个opt-in CUDA参数化用例。随后全项目CUDA回归302 passed；4个CUDA suffix用例均已实际执行通过。

2026-10-24完整Qwen3-0.6B BF16验收通过：64-token冷/热prefix的实际prefill工作从64降为0，partial hit只执行后缀，共享KV不变；分块与完整prefill输出一致，batch2/4 eager与Graph一致；真实并发HTTP/SSE结束后所有KV引用归还。命令为`python benchmarks/validate_system_extensions.py --fuse-projections --http-smoke`，原始结果保存在`benchmarks/results/stage27_system_validation.json`。这些是正确性与避免重复计算的证据，不代表固定端到端加速倍数。

覆盖：absolute causal suffix、完整 forward 与分块 logits/KV 对齐、单 token decode 不 gather、exact hit 无 forward、partial hit 只算 suffix、非对齐边界 logits、无 logits 时 private 重算、decode 优先与预算、有限池并发完成、最终 token 不 grow、LRU/live refs、waiting stale 元数据刷新、scratch 永久 OOM、空缓存对象、异常释放、独立采样参数、truncate 引用计数。

## 5. 性能与限制

本阶段改善的是避免重复计算、长 prompt 阻塞和资源生命周期；没有在此报告宣称具体加速倍数。虚拟最大预算预留较保守，会限制高输出上限场景的并发；未来可在支持安全 preemption/recompute 后评估超额接入。

prefill chunk 仍逐请求执行，没有把多个 prompt 打包成 ragged batch。非零位置的多 token suffix 当前 gather dense KV，适合 correctness-first chunking 和投机验证，但长上下文有额外复制成本。后续可增加直接读取 paged KV 的多 query attention。

尚未实现跨模型/量化配置共享 cache、磁盘 cache、prefix 边界 logits 的持久化或多 GPU 分配。

建议 GPU 验收：FP16/BF16 CUDA suffix 数值对拍；dandan D128 offset Flash；真实 Qwen3 warm-prefix/冷启动同输出比较；混合长 prefill 与短 decode 的 P95 TPOT；有限 KV 池压力下完成率；eager/Graph 下相同 token 流与 live KV 校验。性能测试应在空闲 GPU 上测量，并分开统计 cache 命中率、实际计算 token 数与用户输入总 token 数。
