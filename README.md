<p align="center">
  <img src="assets/cover.png" alt="mini-engine — Windows-native GPU inference" width="100%">
</p>

<p align="center"><strong>简体中文</strong> · <a href="README.en.md">English</a></p>

# mini-engine

面向 Windows 11 的轻量级单 GPU 推理引擎，支持 Qwen3-0.6B 本地对话和 HTTP/SSE 服务。项目自行实现模型 forward、Paged KV Cache、连续批处理和 CUDA 算子，使用 Transformers 加载权重与 tokenizer，并作为正确性参考。

## 主要功能

- **原生 Qwen3**：Q/K RMSNorm、RoPE、GQA、SwiGLU，支持增量生成与 batch decode。
- **Paged KV Cache**：分页读写、引用计数、LRU 回收，完整 block 的 Prefix Cache 共享与命中计算跳过。
- **连续批处理**：decode 优先调度、token 预算、chunked/ragged prefill，以及请求取消和异常回收。
- **CUDA 执行**：自有 PagedAttention、paged prefill、Norm、Residual、SwiGLU、Embedding、RoPE、KV scatter 和 greedy sampling；Linear 调用 cuBLAS/cuBLASLt，支持 QKV、Gate/Up 投影融合。
- **CUDA Graph**：单请求与 batch 的单 token decode，可将 logits 和 greedy sampling 纳入图中。
- **可选算子库**：接入 `dandan_ops` 的 FP16 Linear、D128 GQA dense Flash prefill 与 W4A16 计算。
- **推理入口**：命令行、多轮 Chat、HTTP/SSE，以及单请求 greedy prompt-lookup 投机解码。

## 快速开始

以下命令使用 PowerShell。已有项目时，进入现有项目根目录并加载 `enter_project.ps1`，跳过克隆；已有虚拟环境或模型时，也可跳过对应步骤。编译 CUDA 扩展前需准备下方列出的运行环境。

### 安装

```powershell
git clone https://github.com/dandan20031031/myvllm-mini-engine.git D:\myvllm-mini-engine
Set-Location -LiteralPath 'D:\myvllm-mini-engine'
. .\scripts\enter_project.ps1

python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e ".[model,cuda]" "transformers==4.57.6"
```

项目环境脚本将 pip、Hugging Face、Torch、CUDA 编译缓存和临时目录设置在项目内。每次新开 PowerShell 后，先进入项目根目录，再执行 `. .\scripts\enter_project.ps1`。

### 下载模型

```powershell
.\.venv\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/Qwen3-0.6B')"
```

也可使用已有的完整模型目录，需包含 config、tokenizer 和 safetensors 权重，并在启动时调整 `--model`。

### 本地对话

```powershell
.\.venv\Scripts\python -m mini_engine `
  --model .\models\Qwen3-0.6B --chat `
  --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --sampling-backend cuda --fuse-projections --cuda-graph `
  --max-new-tokens 128 --local-files-only
```

输入 `/clear` 清空对话，输入 `/exit` 退出。单轮生成将 `--chat` 替换为 `--prompt "解释一下 KV Cache"`；去掉 `--cuda-graph` 使用 eager 执行。第一次运行需要编译 CUDA 扩展并捕获所需 Graph，耗时会高于后续请求。

### HTTP / SSE 服务

```powershell
.\.venv\Scripts\python -m mini_engine.server `
  --model .\models\Qwen3-0.6B --served-model-name qwen3-0.6b `
  --device cuda --dtype bf16 --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --sampling-backend cuda --fuse-projections --cuda-graph `
  --prefill-chunk-size 128 --local-files-only
```

另开 PowerShell 发起请求：

```powershell
$body = @{model='qwen3-0.6b'; messages=@(@{role='user'; content='解释 KV Cache'}); max_tokens=64; temperature=0} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Base URL 为 `http://127.0.0.1:8000/v1`。支持 `/health`、`/v1/models`、`/v1/chat/completions`；请求设置 `stream: true` 可使用 SSE。当前为 greedy 聊天子集，要求 `temperature: 0`，支持 usage，不支持 tools 或多模态。默认 Graph 服务的上下文上限为 1024 tokens，包含输入和输出预算。

## 入口与脚本

| 入口 | 作用 |
|---|---|
| `python -m mini_engine` | 单请求生成与交互式多轮 Chat |
| `python -m mini_engine.server` | 单 GPU worker 驱动的 HTTP/SSE 服务 |
| `benchmarks/benchmark_chat_graph.py` | 同一模型交替测量单请求 eager/Graph，记录首轮、TTFT、TPOT 与实际输出吞吐 |
| `benchmarks/benchmark_native_batch.py` | 通过真实 scheduler/runtime 测量 batch eager/Graph |
| `benchmarks/benchmark_paged_sweep.py` | PagedAttention 原实现与向量化实现、不同 block size 的算子对照 |
| `benchmarks/compare_paged_backends.py` | 本项目 CUDA 与 Mini Triton 的同输入组件对照，使用各自独立环境 |
| `benchmarks/profile_native_batch.py` | 采集 CPU/GPU 时间线、NVTX 与 Chrome trace，定位运行瓶颈 |

更多对拍、prefill 消融和服务压测工具见 [Benchmark 说明](benchmarks/README.md)。

## 性能 Benchmark

以下为已记录的实验结果，完整条件与原始证据见[性能评测报告](reports/VALIDATION_AND_BENCHMARKS.md)。

### Qwen3-0.6B 单请求生成

RTX 5080，FP16，内置 CUDA 后端与 cuBLAS Linear，投影融合、CUDA greedy、decode Graph。两个 prompt，各输出 64/128 tokens，预热后每组重复 6 次。

| 指标 | 已记录结果 |
|---|---:|
| TTFT P50 | 19.75–24.40 ms |
| TPOT P50 | 2.53–2.66 ms/token |
| 总输出吞吐 | 342.11–357.51 tokens/s |

计时从 prefill 到最后一个输出 token，排除模型加载、分词、文本解码和首次 Graph 捕获。吞吐按实际输出 token 总数 / 请求总耗时计算；这是本地单请求生成结果，HTTP 与多请求吞吐需单独测量。

```powershell
# 单请求 eager / Graph 配对测量
.\.venv\Scripts\python benchmarks\benchmark_chat_graph.py --model .\models\Qwen3-0.6B --dtype fp16 --ignore-eos

# 连续批处理，脚本自动运行 eager 与 Graph 两种模式
.\.venv\Scripts\python benchmarks\benchmark_native_batch.py --model .\models\Qwen3-0.6B --dtype bf16 --fuse-projections --local-files-only --ignore-eos
```

### PagedAttention 组件对照

2026-10-26 本机 RTX 5080 测量：FP16、Hq=16、Hkv=8、head_dim=128、block_size=256，随机非连续物理页。两边使用相同输入字节并通过 FP32 oracle 对拍；下表为两次独立进程的 Graph 均摊耗时 P50 范围，单位 µs。

| Batch / Context | 本项目 CUDA | MiniVLLM Triton |
|---|---:|---:|
| 1 / 128 | 18.945–19.330 | 3.917–4.323 |
| 8 / 512 | 40.340–40.422 | 14.308–14.834 |
| 32 / 2048 | 396.343–396.744 | 361.462–366.140 |

本项目使用 PyTorch 2.9.1+cu128；Mini 使用 2.11.0+cu128、Triton 3.8.0 和固定 Windows 修复版本。软件栈不同，GPU 未锁频，范围也不是置信区间。结果仅覆盖 PagedAttention 组件，完整模型还包含 GEMM、prefill、采样与调度。Nano-vLLM 未完成本机 GPU 验收，目前没有三方同条件引擎速度排名。[对照数据与复现方法](reports/VALIDATION_AND_BENCHMARKS.md)

## 项目结构

```text
myvllm-mini-engine/
├── mini_engine/
│   ├── model/          原生 Qwen3、模型加载与 runner
│   ├── cache/          KV 池、物理页与 Prefix Cache
│   ├── engine/         请求、调度、ragged batch、runtime 与 CUDA Graph
│   ├── kernels/        CUDA 源码、后端分发与数值 reference
│   ├── quantization/   量化 reference 与 dandan W4A16 bridge
│   ├── serving/        HTTP 协议、GPU worker 与 SSE
│   ├── config.py       模型与后端配置
│   └── cli.py          CLI / HTTP 共用启动参数
├── benchmarks/         性能测量、正确性对拍与 profiling
├── scripts/            Windows 环境脚本
├── tests/              模型、算子、调度与服务回归
└── reports/            架构与实验说明，测量摘要位于 data/
```

Python 集成入口为 `mini_engine.model.NativeQwen3Runner` 和 `mini_engine.RunnerConfig`；batch 会话通过 `runner.create_batch_runtime()` 创建。调用关系与资源生命周期见[架构说明](reports/ARCHITECTURE.md)。模型、虚拟环境、缓存和构建产物均已被 Git 忽略。

## 运行环境与支持范围

- **已验证环境**：Windows 11、RTX 5080、Python 3.13、PyTorch 2.9.1+cu128、Transformers 4.57.6、MSVC C++ Build Tools 与 Ninja；需要配置可用的 CUDA Toolkit 和编译器。
- **CUDA 版本**：引擎扩展已在 Toolkit 13.3 上通过目标机测试；独立算子库此前使用 12.8 构建。其他 PyTorch / Toolkit / GPU 组合需要重新验证。
- **浮点与量化**：支持 FP32/FP16/BF16 浮点路径，自有 CUDA attention 等算子要求 FP16/BF16。INT8/INT4/FP8 E4M3、E5M2 有反量化计算 reference；高性能 FP8 GEMM/KV、NVFP4、MXFP8 尚未实现。
- **可选 dandan 后端**：单独安装算子库后，使用 `--dtype fp16 --linear-backend dandan_cuda`；dense Flash 使用 `--prefill-backend dandan`，W4A16 增加 `--quantization int4 --quant-backend dandan`，group size 为 64/128。质量与速度需按配置验收。
- **当前边界**：单 GPU、Qwen3-0.6B、greedy 服务。CUDA prefill 直接读取分页历史；其他 prefill 后端仍逐请求做 dense attention。FP16 ragged 严格 KV 逐元素检查尚未通过，短输出一致不代表 batch invariance 已完成。
- **Graph 与投机**：Graph 覆盖单 token decode，prefill 和 CPU 调度在图外。单请求 prompt lookup 使用 `--speculative prompt_lookup --max-draft-tokens 4`，需去掉 `--cuda-graph`；暂不支持 batch 投机。

安装开发依赖后，可运行回归与完整模型验收：

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev]"
$env:MINI_ENGINE_TEST_CUDA = '1'
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --dtype bf16 --http-smoke
```

完整模型验收需要本地权重与 GPU；pytest 缓存位于 `build/pytest_cache`。历史验证范围见[验证报告](reports/VALIDATION_AND_BENCHMARKS.md)。

## 参考与许可证

参考 [vLLM](https://github.com/vllm-project/vllm)、[Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 与 [MiniVLLM](https://github.com/Wenyueh/MinivLLM) 的推理机制与工程组织。代码采用 [Apache-2.0](LICENSE)；Qwen3 权重和 tokenizer 遵循其自身许可证。
