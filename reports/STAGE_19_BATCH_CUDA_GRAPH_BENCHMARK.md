# 阶段 19：真实 Batch CUDA Graph Benchmark

## 实现内容

新增 `benchmarks/benchmark_native_batch.py`，不再调用单请求 `generate_incremental()`，而是直接驱动：

```text
GenerationRequest
  -> Scheduler
  -> NativeBatchRuntime.prefill_step
  -> NativeBatchRuntime.decode_step
  -> eager forward_decode_batch / CUDA Graph bucket replay
```

脚本同时运行 eager 和 CUDA Graph 两种模式，固定模型、prompt、batch、dtype、block size、输出长度、warmup 和 repetition，并输出总耗时、实际输出 token 数、聚合 TPS 和显存。

## Graph 路径修复

真实模型 capture 暴露了几个此前单元测试没有覆盖的问题：

- graph padding 的 block 数和 `BlockManager.allocate(token_count)` 单位混用，已转换为 `blocks * block_size`；
- `forward_decode_batch` 的 GPU metadata host reduction 在 capture 时非法，trusted graph buffer 路径跳过该同步检查；
- batch KV write 的 offsets `.any().item()` 同步检查在 capture 时非法，改为只在 `validate=True` 的公开路径执行；
- Windows CUDA 自定义 kernel 的立即 launch error query 会破坏 stream capture，capture 路径不做 host-side launch 查询；
- 所有自定义 CUDA kernel 仍保留设备端边界保护，普通 eager/validation 路径继续通过 Python/C++ shape 检查。

## RTX 5080 首轮结果

配置：

```text
GPU             NVIDIA GeForce RTX 5080
模型            Qwen3-0.6B
PyTorch         2.9.1+cu128
dtype           BF16
请求数          3
输出预算        32 tokens/request
block_size      16
prefill         CUDA correctness-first
decode          CUDA Paged Attention
```

未融合路径：

| 模式 | median total | median aggregate TPS |
|---|---:|---:|
| eager | 438.38 ms | 189.33 tokens/s |
| CUDA Graph | 175.15 ms | 473.87 tokens/s |

Graph 相对 eager 约 `2.50x`。开启阶段 18 projection fusion 后：

| 模式 | median total | median aggregate TPS |
|---|---:|---:|
| eager + fusion | 397.11 ms | 209.01 tokens/s |
| CUDA Graph + fusion | 159.88 ms | 519.15 tokens/s |

这些数字包含逐请求 prefill 和 scheduler 生命周期，适合描述当前 runtime 的端到端 batch 基线；不能直接与只统计 decode 的 MiniVLLM 数字相加或相减。公开结果文件为：

- `benchmarks/results/native_batch_stage19.json`
- `benchmarks/results/native_batch_stage19_fused.json`

## 结论

完整 Qwen3 batch CUDA Graph 路径已经实际运行，Graph 的收益从单元测试级别提升为真实模型级别。当前 3 请求配置下，Graph 约 2.5x，projection fusion + Graph 达到约 519 tokens/s。下一步需要测试 batch=1/2/4/8、P50/P95、FP16、不同 block size，并把 prefill 和 decode 分开计时。

## 限制

- prefill 仍然逐请求执行；
- sampling 仍在 Graph 外使用 PyTorch reference；
- 结果尚未与 MiniVLLM 使用完全相同的 prompt、输出长度和 FP16 口径对齐；
- 自定义 kernel capture 路径暂时关闭 host-side launch error query，错误验证依赖 eager 检查、设备边界保护和 Graph replay 后同步；
- 尚未加入 vLLM、SGLang 或 TensorRT-LLM 的同机对照。
