# MiniQwenEngine

MiniQwenEngine 是一个面向 Qwen3-0.6B 的 Windows 11 单 GPU 推理引擎，使用 PyTorch 加载模型，并通过项目自有 CUDA 扩展实现 Paged KV Cache、PagedAttention、数据移动和 elementwise 算子。

当前实现和后续方向见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。项目仍处于研究和性能优化阶段，单算子 benchmark 不等同于 vLLM 的端到端性能结论。

## 已实现功能

### 推理引擎

- Qwen3-0.6B native forward 和增量 decode；
- Windows 11 原生 CUDA 扩展，支持 MSVC、nvcc 和 Ninja；
- 请求状态管理、BlockManager 和 continuous batching scheduler；
- Paged KV Cache tensor pool，支持跨 block 写入、回收和引用计数；
- Prefix Cache 的完整 block 共享和 LRU 生命周期管理；
- 单请求 CLI、多轮 Chat 和 scheduler 驱动的 batch decode runtime。

### CUDA 算子

- CUDA PagedAttention，支持 FP16/BF16、GQA、非连续物理 block 和 split-KV；
- CUDA RMSNorm、Residual + RMSNorm、Residual Add、SwiGLU 的 SiLU×Up；
- CUDA Embedding gather、RoPE 和分页 KV 写入；
- CUDA causal prefill correctness-first backend；
- 显式 CUDA Linear，底层使用 PyTorch 的 cuBLAS/cuBLASLt dispatch；
- CUDA Graph decode bucket，支持固定 buffer、padding slot 和 replay；
- QKV projection、Gate/Up projection fusion，减少小矩阵乘和 launch 次数。

### 精度和量化

- FP32、FP16、BF16 浮点路径；
- INT8 weight-only reference；
- packed INT4 reference；
- FP8 E4M3/E5M2 weight-only reference；
- NVIDIA FP8、NVFP4、MXFP8 能力探测；
- 量化路径区分 reference dequant 和真正的 Tensor Core GEMM。

NVFP4、MXFP8、FP8/INT8 KV Cache 和高性能 INT4/FP8 GEMM 仍在开发中。当前量化 reference 不应被当作已经完成的量化加速。

## 快速开始

项目针对 Windows 11。建议使用项目目录中的 `.venv`，并把模型、缓存和编译产物放在当前项目目录或其他 D 盘路径。

### 安装依赖

```powershell
Set-Location -LiteralPath "D:\dandan123\dandan\myvllm"

python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,model,cuda]"
```

已有项目环境可以直接执行：

```powershell
. .\scripts\enter_project.ps1
```

首次使用 CUDA backend 需要 Windows x64 MSVC、CUDA Toolkit、PyTorch CUDA wheel 和 Ninja。CUDA 扩展会编译到 `build/torch_extensions`。

### 准备模型

从 [Qwen3-0.6B 模型页](https://huggingface.co/Qwen/Qwen3-0.6B)下载模型权重和 tokenizer，放到本地目录，例如：

```text
D:\models\Qwen3-0.6B
```

模型权重不包含在本仓库中。使用前请阅读 Qwen3 模型卡和模型许可证。

### 单轮推理

```powershell
. .\scripts\enter_project.ps1

.\.venv\Scripts\python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "介绍一下 KV Cache" `
  --device cuda `
  --dtype bf16 `
  --prefill-backend auto `
  --decode-backend cuda `
  --elementwise-backend cuda `
  --data-backend cuda `
  --linear-backend cuda `
  --max-new-tokens 128 `
  --local-files-only
```

### 多轮 Chat

```powershell
. .\scripts\enter_project.ps1

.\.venv\Scripts\python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --chat `
  --device cuda `
  --dtype bf16 `
  --prefill-backend cuda `
  --decode-backend cuda `
  --elementwise-backend cuda `
  --data-backend cuda `
  --linear-backend cuda `
  --local-files-only
```

Chat 中支持：

```text
/clear  清空对话历史
/exit   退出
```

### 启用 Projection Fusion

```powershell
.\.venv\Scripts\python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "解释 continuous batching" `
  --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --fuse-projections --local-files-only
```

### 运行测试

CPU/reference 回归：

```powershell
.\.venv\Scripts\python -m pytest -q
```

CUDA 回归：

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m pytest -q -o cache_dir=build/pytest_cache
```

## 性能 Benchmark

所有结果都应记录 GPU、PyTorch/CUDA 版本、dtype、模型、prompt、warmup、重复次数、输出 token 数和计时口径。单算子 latency 不等于端到端 tokens/s。

### PagedAttention 单算子

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"

.\.venv\Scripts\python benchmarks\benchmark_paged_attention.py `
  --batch 1 8 32 `
  --context 128 512 2048 `
  --dtype fp16 `
  --block-size 256 `
  --warmup 10 `
  --repetitions 100
```

RTX 5080 上与项目 PyTorch fallback 的已有结果：

| batch/context | CUDA PagedAttention | 相对项目 PyTorch fallback |
|---|---:|---:|
| 1 / 128 | 0.0474 ms | 7.97x |
| 8 / 512 | 0.0951 ms | 27.60x |
| 32 / 2048 | 0.6999 ms | 14.71x |

同配置下，MiniVLLM Triton 分块 kernel 约为 `0.019 / 0.017 / 0.377 ms`。当前项目在 PagedAttention 上仍有优化空间，尤其是 vectorized KV load、batch=1 并行度和 flash-decoding。

### 单请求 Qwen3-0.6B

```powershell
.\.venv\Scripts\python benchmarks\benchmark_native_qwen3.py `
  --model "D:\models\Qwen3-0.6B" `
  --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --max-new-tokens 32 --warmup 2 --repetitions 7 `
  --local-files-only
```

RTX 5080、Qwen3-0.6B、BF16、全 CUDA backend 的已有单请求基线：

```text
median TTFT: 约 22–24 ms
median TPOT: 约 20–21 ms/token
```

### Batch eager / CUDA Graph

```powershell
.\.venv\Scripts\python benchmarks\benchmark_native_batch.py `
  --model "D:\models\Qwen3-0.6B" `
  --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --max-new-tokens 32 --warmup 1 --repetitions 3 `
  --local-files-only `
  --output benchmarks\results\native_batch.json
```

默认 benchmark 使用 3 条 prompt，同时运行 eager 和 CUDA Graph decode。RTX 5080 首轮真实结果如下：

| 模式 | median total | 聚合吞吐 |
|---|---:|---:|
| eager | 438.38 ms | 189.33 tokens/s |
| CUDA Graph | 175.15 ms | 473.87 tokens/s |
| Projection Fusion + CUDA Graph | 159.88 ms | 519.15 tokens/s |

配置为 BF16、3 条请求、32 token 输出预算、`block_size=16`。这些数字包含逐请求 prefill 和 scheduler 生命周期，不是只统计 decode kernel，也不是 vLLM/MiniVLLM 的同口径最终对比。

结果文件默认被 `.gitignore` 排除，避免把机器相关 benchmark JSON 和模型路径提交到公开仓库。

## 项目结构

```text
mini_engine/
├── cache/          Paged KV Cache、block allocator、prefix cache
├── engine/         scheduler、request、batch runtime、CUDA Graph buckets
├── kernels/        PyTorch fallback、CUDA dispatch 和 .cu 源码
├── model/          Qwen3 reference/native forward 和 runner
└── quantization/   INT8/INT4/FP8 reference 后端

benchmarks/         单算子、单请求和 batch benchmark
tests/              CPU、CUDA、模型对齐和 runtime 回归
reports/            详细技术报告和实验记录
PROJECT_STATUS.md   当前项目总览、已实现能力和后续方向
```

## 当前限制

- prefill 仍然逐请求执行，chunked/batched prefill 尚未完成；
- Prefix Cache 已共享完整物理 block，但尚未完全跳过 prompt 计算；
- CUDA Graph 当前重点是 batch decode，sampling 仍在 Graph 外；
- INT4/FP8 高性能 GEMM、FP8/INT8 KV、NVFP4/MXFP8 尚未完成；
- Windows HTTP/OpenAI/SSE 服务尚未完成；
- 投机解码已规划但尚未实现；
- `dandan_ops` 是后续计划接入的独立高性能算子库，当前公开主线默认使用项目 CUDA/cuBLAS backend；
- 尚未完成与 vLLM、SGLang、TensorRT-LLM 的完全同口径端到端对比。

## 许可证

项目代码使用 Apache-2.0，详见 [LICENSE](LICENSE)。Qwen3 模型权重和 tokenizer 遵循其自身许可证。
