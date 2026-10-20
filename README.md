# MiniQwenEngine

一个面向 Qwen3-0.6B 的小型单 GPU 推理引擎实验项目。开发路线见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)。当前版本先完成可测试的引擎骨架，包含：

阶段报告索引：

- [阶段 00：项目骨架与推理调度基础](reports/STAGE_00_FOUNDATION.md)
- [阶段 01：Paged KV Cache 与 Benchmark 基础设施](reports/STAGE_01_KV_CACHE_AND_BENCHMARK.md)
- [阶段 02：Qwen3 参考增量生成路径](reports/STAGE_02_INCREMENTAL_REFERENCE.md)
- [阶段 03：原生 Qwen3 Forward、Paged Decode 与 Windows 适配](reports/STAGE_03_NATIVE_QWEN3_AND_WINDOWS.md)
- [阶段 04：Windows CUDA Paged Attention](reports/STAGE_04_WINDOWS_CUDA_PAGED_ATTENTION.md)
- [阶段 05：CUDA Paged Attention Benchmark](reports/STAGE_05_CUDA_KERNEL_BENCHMARK.md)
- [阶段 06：INT4、FP8 与 NVIDIA 精度能力矩阵](reports/STAGE_06_QUANTIZATION_PRECISION_BACKENDS.md)
- [阶段 07：Windows CUDA RMSNorm、Residual 与 SwiGLU](reports/STAGE_07_CUDA_ELEMENTWISE.md)
- [阶段 08：Windows CUDA Embedding、RoPE 与 Paged KV 写入](reports/STAGE_08_CUDA_DATA_MOVEMENT.md)
- [阶段 09+10：Continuous Batching Runtime 与 Prefix Block Cache](reports/STAGE_09_10_BATCH_RUNTIME_PREFIX_CACHE.md)
- [阶段 11：CUDA Graph Decode Buckets](reports/STAGE_11_CUDA_GRAPH_BUCKETS.md)
- [阶段 12+13：Windows CUDA Prefill、显式 CUDA Linear 与本地 Chat](reports/STAGE_12_13_PREFILL_LINEAR_CHAT.md)
- [阶段 11/12/13 补充：Graph Runtime、真实模型验收与 Benchmark](reports/STAGE_11_12_13_GRAPH_VALIDATION_BENCHMARK.md)

- continuous batching 的请求调度器；
- 固定大小 Paged KV block allocator；
- 物理 KV Cache tensor pool，支持跨 block 写入、逻辑 gather 和按 block 清理；
- PyTorch fallback 的 RMSNorm、RoPE、SwiGLU 和 paged attention；
- 可插拔的浮点和参考 W8A16 后端；
- native Qwen3 decoder projection 的 INT8、packed INT4、FP8 E4M3/E5M2 reference 后端；
- NVIDIA FP8/NVFP4/MXFP8 能力报告，明确区分 reference、reserved 和 CUDA 加速状态；
- 可选的 Transformers Qwen3 reference runner。
- 原生 Qwen3 forward：SDPA prefill + 项目 KV Cache paged decode。
- 可选 CUDA elementwise：全部 RMSNorm、融合 attention residual + post norm、MLP residual add 和 SwiGLU。
- 可选 CUDA data backend：Embedding gather、RoPE 和分页 KV 写入。
- 参考版 prefill + `past_key_values` 单 token 增量生成，以及 TTFT/TPOT 统计。
- backend-agnostic benchmark 工具，可统一记录延迟、吞吐和 CUDA 峰值显存。

## 安装与测试

```powershell
pip install -e ".[dev]"
python -m pytest -q
```

如果要加载 HuggingFace 模型：

```powershell
pip install -e ".[model]"
```

## 使用调度器和分页缓存

```python
from mini_engine import EngineConfig, LLMEngine

engine = LLMEngine(EngineConfig(device="cuda", dtype="bf16"))
request = engine.add_request([1, 2, 3, 4], max_new_tokens=32)
scheduled = engine.schedule()
print(request.request_id, scheduled.prefill, engine.block_manager.free_blocks)
```

## 运行参考生成

```python
from mini_engine import EngineConfig, LLMEngine
from mini_engine.model import Qwen3ModelRunner

config = EngineConfig(model_name_or_path="Qwen/Qwen3-0.6B", device="cuda")
runner = Qwen3ModelRunner(config.model_name_or_path, device=config.device)
engine = LLMEngine(config, model_runner=runner)
print(engine.generate_reference("介绍一下 KV Cache", max_new_tokens=64))
```

参考增量生成：

```python
metrics = engine.generate_incremental("介绍一下 KV Cache", max_new_tokens=64)
print(metrics.text, metrics.ttft_ms, metrics.tpot_ms)
```

参考生成使用 Transformers，仅用于建立正确性基线。现已另外提供 NativeQwen3Runner，使用自有 forward 和 KV Cache；native runner 尚未接通调度器的多请求执行。

## KV Cache tensor pool

```python
import torch
from mini_engine.cache import KVCachePool

cache = KVCachePool(
    num_layers=28,
    num_blocks=256,
    block_size=16,
    num_kv_heads=8,
    head_dim=128,
    dtype=torch.bfloat16,
    device="cuda",
)
print(cache.memory_bytes / 1024**3, "GiB")
```

## Benchmark 基础工具

```python
from mini_engine.benchmarks import BenchmarkConfig, benchmark_callable

result = benchmark_callable(
    lambda: engine.generate_reference("hello", max_new_tokens=32),
    BenchmarkConfig(name="transformers-qwen3", repetitions=3, output_tokens=32),
    device="cuda",
)
print(result.to_dict())
```

上述参考路径使用 Transformers 的 `past_key_values`。native 路径已另行实现；和 vLLM 的公平端到端比较仍需要完整权重验收、实际输出计数和后续优化。当前 benchmark_callable 的 output_tokens 是调用方提供的预算，并不自动统计 EOS 提前停止后的实际输出数，不能直接用于正式吞吐结论。

## Windows 11 本地 native 推理

```powershell
pip install -e ".[model]"
python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "介绍一下 KV Cache" `
  --device cuda `
  --dtype bf16 `
  --prefill-backend auto `
  --max-new-tokens 128 `
  --local-files-only
```

`auto` 会使用当前 Windows PyTorch wheel 可用的 SDPA backend；`flash` 只有在 `torch.backends.cuda.is_flash_attention_available()` 为 true 时才可用。当前环境的 Windows wheel 未编译 Flash backend，因此请使用 `auto`，不要把 math fallback 当作 Flash 性能结果。

当前已通过小型随机 Qwen3 的 CPU FP32 / GPU BF16 / GPU FP16 对齐；尚未在完整 Qwen3-0.6B 权重上验收生成。上面的模型路径是示例，请替换成实际权重目录。

项目独立环境中的验证命令（不激活环境也可运行）：

```powershell
$env:USE_TF = "0"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m mini_engine --help
```

## CUDA decode kernel

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m pytest tests/test_cuda_paged_attention.py -q
```

首次使用 `--decode-backend cuda` 时会在当前项目的 `build\torch_extensions` 中编译扩展。需要 Windows x64 MSVC、CUDA Toolkit、PyTorch CUDA wheel 和 Ninja；编译器本身仍可安装在系统盘，但项目源码、缓存和产物不写入 C 盘。

Paged Attention 单算子 benchmark：

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python benchmarks\benchmark_paged_attention.py `
  --batch 1 4 8 `
  --context 128 1024 4096 `
  --dtype bf16 `
  --warmup 10 `
  --repetitions 50
```

结果默认写入当前项目的 `benchmarks\results\paged_attention.json`。


## CUDA Norm / Residual / SwiGLU

~~~powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m pytest -q -o cache_dir=build/pytest_cache
.\.venv\Scripts\python -m mini_engine --model "D:\models\Qwen3-0.6B" --prompt "介绍一下 KV Cache" --dtype bf16 --decode-backend cuda --elementwise-backend cuda --local-files-only
~~~

--elementwise-backend cuda 在 prefill/decode 都启用自研 FP16/BF16 norm、residual 和 SiLU×Up；
默认 torch 保留参考路径。RMSNorm 输入须连续；算子是 inference-only，不提供 backward。

单算子 benchmark：

~~~powershell
.\.venv\Scripts\python benchmarks\benchmark_elementwise.py --dtype bf16
~~~

同时输出 CUDA Graph device timeline 和 eager wall time，含精度检查；结果保存在 benchmarks/results。
独立 residual add 尚未测得优于 PyTorch 的收益。完整 0.6B 权重和端到端性能仍待验收。
