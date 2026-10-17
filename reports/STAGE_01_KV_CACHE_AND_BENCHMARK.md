# 阶段 01 报告：Paged KV Cache 与 Benchmark 基础设施

## 1. 阶段目标

本阶段把项目从“请求调度和 block 元数据”推进到可以承载真实模型推理的基础设施：

1. 将物理 KV Cache 预分配为 PyTorch tensor；
2. 支持逻辑 token 到物理 block 的映射；
3. 支持跨 block 写入和按逻辑长度读取；
4. 为后续 CUDA Paged Attention 固定 tensor 布局和调用接口；
5. 建立统一 benchmark 计时和结果输出格式。

本阶段没有伪造端到端生成结果。Qwen3 原生增量 forward 尚未接入，因此 benchmark 工具目前用于测试任意 callable，以及后续接入 Transformers、vLLM 和 MiniQwenEngine adapter。

## 2. 新增文件

```text
mini_engine/cache/kv_cache.py
mini_engine/benchmarks/__init__.py
mini_engine/benchmarks/runner.py
benchmarks/README.md
tests/test_kv_cache.py
tests/test_benchmark.py
reports/STAGE_01_KV_CACHE_AND_BENCHMARK.md
```

同时修改了：

```text
mini_engine/cache/__init__.py
mini_engine/engine/llm_engine.py
README.md
DEVELOPMENT_PLAN.md
```

## 3. KV Cache 调用逻辑

### 3.1 组件关系

```text
GenerationRequest
        |
   Scheduler
        |
 BlockManager       分配逻辑 block -> physical block id
        |
 KVCachePool        保存 physical block 对应的 K/V tensor
        |
 Paged Attention    后续 CUDA kernel 读取 block table 和 cache
```

`BlockManager` 和 `KVCachePool` 故意分离：

- `BlockManager` 只负责资源所有权和 block 生命周期；
- `KVCachePool` 只负责 GPU/CPU tensor 存储；
- Attention kernel 通过 block table 把二者连接起来。

这样做的好处是，后续把 PyTorch fallback 替换成 CUDA kernel 时，不需要修改请求和调度器。

### 3.2 KV Cache tensor 布局

`KVCachePool` 使用以下布局：

```text
[num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

其中：

- `num_layers`：Transformer 层数；
- `num_blocks`：物理 block 总数；
- `block_size`：每个 block 保存的 token 数；
- `num_kv_heads`：K/V head 数，适配 GQA/MQA；
- `head_dim`：每个 head 的维度。

同一层的 K 和 V 分别保存在：

```python
cache.key[layer, block_id]
cache.value[layer, block_id]
```

### 3.3 写入流程

调用：

```python
cache.write(layer, block_ids, token_offset, key, value)
```

输入的 `key` 和 `value` 形状为：

```text
[tokens, num_kv_heads, head_dim]
```

写入逻辑：

1. 检查 K/V shape、层号和 token offset；
2. 计算 token offset 对应的 block 编号和 block 内偏移；
3. 计算当前 block 可写入的 token 数；
4. 写满当前 block 后切换到下一个 block；
5. 直到所有 token 写入完成。

因此一个 prompt 即使跨越多个非连续物理 block，也能被正确写入。

### 3.4 读取流程

调用：

```python
key, value = cache.gather(layer, block_ids, token_count)
```

读取流程会：

1. 根据 block table 找到物理 block；
2. 将 block 维和 token 维展平；
3. 截取前 `token_count` 个逻辑 token；
4. 返回形状为 `[token_count, num_kv_heads, head_dim]` 的 K/V。

这是 correctness-first 的参考路径。真正的 CUDA Paged Attention 不需要先 gather 成连续 tensor，而是直接根据 block table 在 kernel 内读取物理 block，从而避免额外拷贝。

## 4. Engine 中的调用逻辑

`LLMEngine.allocate_kv_cache(...)` 是上层入口：

```python
engine.allocate_kv_cache(
    num_layers=28,
    num_kv_heads=8,
    head_dim=128,
    dtype=torch.bfloat16,
)
```

调用链为：

```text
LLMEngine.allocate_kv_cache
        -> KVCachePool(...)
        -> cache.key / cache.value
```

请求调度仍然由：

```text
LLMEngine.add_request
        -> Scheduler.add_request
        -> Scheduler.schedule
        -> BlockManager.allocate
```

目前 `Scheduler` 已经可以为请求分配 block，但真正的模型 forward 尚未调用 `KVCachePool.write`。下一阶段会把模型每层产生的 K/V 接入这里。

## 5. Benchmark 调用逻辑

核心 API：

```python
result = benchmark_callable(
    fn,
    BenchmarkConfig(name="backend", repetitions=3, output_tokens=32),
    device="cuda",
)
```

调用流程：

1. 执行 warmup，排除首次编译和缓存初始化影响；
2. CUDA 场景同步设备；
3. 重置 CUDA peak memory 统计；
4. 重复执行 callable；
5. 每次执行前后同步 CUDA；
6. 记录 wall-clock latency；
7. 计算平均延迟、P50、P95、tokens/s 和峰值显存；
8. 通过 `save_results` 写入 JSON。

benchmark runner 不绑定任何推理后端。后续只需提供三个 callable adapter：

```text
Transformers adapter
vLLM adapter
MiniQwenEngine adapter
```

就可以复用完全相同的计时和结果格式。

## 6. 测试覆盖

当前测试共 6 个，全部通过：

```text
tests/test_core.py       调度、block 生命周期、fallback 算子、采样
tests/test_kv_cache.py   跨 block 写入、清理、显存统计
tests/test_benchmark.py  benchmark 指标结构
```

最近一次结果：

```text
6 passed
```

## 7. 当前限制

本阶段仍然存在这些限制：

- `KVCachePool` 是 PyTorch tensor 实现，不是 CUDA 专用 kernel；
- `gather` 会生成连续视图/读取路径，性能不代表最终 Paged Attention；
- block table 尚未传入真实 Qwen3 forward；
- benchmark 还没有 vLLM 和本地引擎的端到端 adapter；
- 尚未测量 TTFT 与 TPOT，因为生成循环还未接入。

## 8. 下一阶段

下一阶段目标是完成单请求 Qwen3-0.6B 增量 forward：

1. 从 Transformers config 读取层数、KV heads、head dimension；
2. 建立每层 K/V 写入 `KVCachePool` 的路径；
3. 完成 prefill 后的单 token decode；
4. 使用 block table 读取历史 K/V；
5. 接入 greedy sampling；
6. 增加端到端输出正确性测试；
7. 为 TTFT、TPOT benchmark 提供第一个本地 engine adapter。
