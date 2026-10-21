# 阶段 05 报告：CUDA Paged Attention Benchmark

## 1. 阶段目标

阶段 04 已证明 CUDA kernel 的输出正确。本阶段建立独立的单算子 benchmark：

- 对比 PyTorch paged attention fallback 和 CUDA backend；
- 使用 CUDA Event 统计 kernel latency；
- 记录 warmup、重复次数、峰值显存和 speedup；
- 每组测量前验证 CUDA 与 reference 数值一致；
- 所有 JSON 结果写入项目 D 盘。

## 2. 新增文件

```text
benchmarks/benchmark_paged_attention.py
tests/test_benchmark_script.py
reports/STAGE_05_CUDA_KERNEL_BENCHMARK.md
```

## 3. 调用逻辑

```text
make_case
  -> query [batch, query_heads, head_dim]
  -> key/value [physical_blocks, block_size, kv_heads, head_dim]
  -> block_tables [batch, max_blocks]
  -> context_lens [batch]

load_cuda_extension
  -> 编译时间排除在性能统计之外

reference = paged_attention(..., backend="torch")
cuda     = paged_attention(..., backend="cuda")
  -> assert_close

time_backend
  -> warmup
  -> CUDA Event record
  -> kernel
  -> Event synchronize
  -> elapsed_time
```

`--batch` 和 `--context` 是成对输入，例如：

```text
--batch 1 4 8 --context 128 1024 4096
```

默认参数使用 Qwen3-0.6B 常见的 `heads=16`、`kv_heads=8`、`head_dim=128`、`block_size=16`，命令行可以覆盖。

## 4. 正确性保护

每个 workload 先执行 torch 和 CUDA backend，并调用：

```python
torch.testing.assert_close(cuda, reference, atol=tolerance, rtol=0.03)
```

如果 kernel 结果不一致，benchmark 立即失败，不会写入 speedup 结论。

## 5. 输出格式

每组 JSON 记录 batch、context、dtype、torch/cuda 的平均、最小、最大 latency、峰值显存和 speedup。实际数值必须由目标机器运行脚本生成，本报告不预填不存在的性能数字。

## 6. Windows 11 与 D 盘约束

脚本调用 `configure_local_cache()`，默认结果为：

```text
`benchmarks/results/paged_attention.json`
```

CUDA extension 编译文件为：

```text
`build/torch_extensions/paged_attention`
```

不需要 Linux shell、WSL 或 `/tmp`。

## 7. 测试与下一步

新增测试确认 benchmark 脚本使用项目内结果目录且没有硬编码 C 盘路径。CUDA correctness 测试仍为阶段 04 的 24 passed。

正式性能命令：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_paged_attention.py
```

阶段 05 的正式输出由目标 GPU 运行生成。拿到真实 latency 后，再针对最慢 workload 优化 CTA/warp 映射、vectorized load、GQA 复用、partition merge 和 CUDA Graph。

## 8. RTX 5080 smoke benchmark

在本机 RTX 5080、BF16、`heads=16`、`kv_heads=8`、`head_dim=128`、`block_size=16` 上执行了 2 次 warmup、5 次重复的缩短测试。结果文件为：

```text
benchmarks/results/paged_attention_smoke.json
```

| batch | context | Torch mean | CUDA mean | speedup |
|---:|---:|---:|---:|---:|
| 1 | 128 | 0.5887 ms | 0.0575 ms | 10.24x |
| 4 | 1024 | 2.0223 ms | 0.0892 ms | 22.68x |
| 8 | 4096 | 3.3396 ms | 0.4870 ms | 6.86x |

这是单算子 smoke 测试，不是端到端生成吞吐。重复次数较少，且 PyTorch fallback 包含 Python 循环和 gather；正式结论应使用默认的 10 warmup/50 repetitions，并继续测量 native Qwen3 TPOT。
