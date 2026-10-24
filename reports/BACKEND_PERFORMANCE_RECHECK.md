# 算子后端性能复核

记录日期：2026-10-24。

本轮新测，不复用阶段22受并发负载与依赖变动影响的数据。设备RTX5080、Qwen3-0.6B、FP16、fusion、3请求、每请求20 prompt token与64实际输出token、prefix关闭、warmup2、重复7次；每种配置分别测eager和真实Graph。计时包括prefill、decode、调度和采样，不包括加载和tokenizer；下表是聚合TPS和PyTorch分配显存峰值。

| 配置 | eager tokens/s | Graph tokens/s | Graph峰值MiB |
|---|---:|---:|---:|
| cuBLAS + 自有CUDA prefill（首轮） | 207.82 | 404.15 | 1614.39 |
| dandan Linear + 自有CUDA prefill | 197.00 | 413.98 | 1614.39 |
| dandan Linear + dandan Flash | 185.68 | 700.08 | 1614.39 |
| cuBLAS + dandan Flash | 267.37 | 696.96 | 1614.39 |
| W4A16 + dandan Flash | 222.37 | 776.94 | 1054.64 |
| cuBLAS + 自有CUDA prefill（末轮复测） | 269.02 | 725.04 | 1614.39 |

**同一基线的Graph首末轮404→725，相差约79%，因此不能把404→700解释为算子库提升73%。** 存在运行顺序、系统负载或其他尚未锁定的状态影响；当前不足以证明FP16后端端到端更快。dandan文件指纹在已使用它的配置中均保持`7d1aa9458f09cf7265a3855b881b491603101ab86e5105c60b0dcef458cad6cc`，文件未变不能证明GPU独占。

W4A16本次Graph峰值1054.64MiB，相对FP16的1614.39MiB降低34.67%；这是PyTorch allocated峰值，不是nvidia-smi全部显存，更不代表模型质量无损。Graph吞吐776.94是本次观察值，不据此发布严格速度比。

dandan Linear+自有prefill的64-token eager/Graph输出digest不同，各模式内部7次重复一致；其余模式各自eager/Graph一致。FP16/cuBLAS、Flash-only与全dandan的Graph输出digest相同。W4与FP16输出不同，不能将量化质量变化当作相同输出的加速。

单请求真实CLI runner（FP16、cuBLAS、fusion，64固定输出，1 warmup、5重复）另测：自有prefill TTFT15.15ms、TPOT12.68ms/token；dandan Flash TTFT16.14ms、TPOT13.57ms/token。只有20-token短prompt，未观察到Flash收益；不代表长prompt结果。该路径是eager，不能把3请求Graph聚合TPS当作Chat单路速度。

原始文件：`benchmarks/results/backend_current_*.json`。公共复现命令：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_native_batch.py `
  --model models/Qwen3-0.6B --dtype fp16 --prefill-backend cuda --linear-backend cuda `
  --fuse-projections --ignore-eos --prefix-cache disabled --max-new-tokens 64 `
  --warmup 2 --repetitions 7 --local-files-only --output benchmarks/results/backend_recheck.json
```

分别更换prefill/linear/quantization参数，并在测试后重跑同一基线。下一次正式比较应在同进程准备独立模型/runtime/Graph后交替测量A/B，采集GPU状态并保留逐次输出；先用[推理引擎性能分析指南](PROFILING.md)定位CPU launch、metadata与GPU kernel热点，避免依据不稳定的总耗时更改后端默认值。
