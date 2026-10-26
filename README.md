# MiniQwenEngine

面向 Qwen3-0.6B 的 Windows 11 单 GPU 推理引擎。Transformers负责权重和tokenizer加载；模型forward、分页KV、调度与CUDA算子由项目实现。支持本地Chat与HTTP/SSE。

[架构与实验报告](reports/README.md) · [后续开发计划](DEVELOPMENT_PLAN.md)

## 已实现功能

- 原生Qwen3：Q/K RMSNorm、RoPE、GQA、SwiGLU、增量与batch decode。
- Paged KV Cache：跨页写入、引用计数、LRU、完整Prefix共享、命中跳过计算及边界logits复用。
- Continuous batching：decode优先、token预算、chunked/ragged prefill、取消与异常回收。
- 自有CUDA：PagedAttention、packed/paged prefill、Norm/Residual/SwiGLU、Embedding/RoPE/KV scatter和greedy argmax。
- CUDA Graph：单请求/Chat及batch一token decode，固定metadata、独立scratch和图内greedy采样；QKV与Gate/Up投影融合。
- 可选`dandan_ops`：FP16 Linear、D128 GQA dense Flash prefill、W4A16 CUDA计算。
- Greedy prompt-lookup投机解码：草稿验证、接受/拒绝与KV回退；CLI、多轮Chat、Windows HTTP/SSE。

| 精度 | 当前支持 |
|---|---|
| FP32 / FP16 / BF16 | 浮点模型；自有CUDA attention/elementwise要求FP16或BF16 |
| INT4 + FP16 | 可选dandan W4A16真实CUDA计算；质量与速度需独立验证 |
| INT8 / INT4 / FP8 E4M3、E5M2 | Weight-only reference，反量化后计算，不宣称加速 |
| FP8 GEMM、FP8/INT8 KV、NVFP4、MXFP8 | 待实现，硬件支持不等于项目已接入 |

## 快速开始

已验证环境为Windows 11、RTX5080、PyTorch2.9.1+cu128、Transformers4.57.6、MSVC和Ninja。引擎扩展使用Toolkit13.3并通过目标机测试；独立算子库此前用CUDA12.8构建。其他版本组合需要重新验证。

### 安装与模型

先进入项目、设置项目缓存，再安装依赖，避免pip和编译产物写入默认系统盘路径：

```powershell
Set-Location -LiteralPath 'D:\dandan123\dandan\myvllm'
. .\scripts\enter_project.ps1

# 已有.venv可跳过创建
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -e ".[dev,model,cuda]"
```

将完整HuggingFace `Qwen/Qwen3-0.6B` 模型目录放在`models/Qwen3-0.6B`，应包含config、tokenizer和safetensors权重。已有本地模型无需重新下载。虚拟环境、模型、缓存、CUDA构建和临时文件均在项目D盘目录，且不提交Git。

### 本地Chat

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python -m mini_engine `
  --model .\models\Qwen3-0.6B --chat `
  --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --fuse-projections --cuda-graph --max-new-tokens 128 --local-files-only
```

`/clear`重置消息，`/exit`退出。单轮推理将`--chat`替换为`--prompt "解释一下KV Cache"`。去掉`--cuda-graph`使用eager路径。

可选算子库需要单独安装：FP16 Linear用`--dtype fp16 --linear-backend dandan_cuda`；dense Flash prefill用`--prefill-backend dandan`；W4A16另加`--quantization int4 --quant-backend dandan`。不支持的dtype/head_dim/后端组合会报错，具体契约见[架构报告](reports/ARCHITECTURE.md)。

投机解码使用eager路径，在命令中去掉`--cuda-graph`，加入`--speculative prompt_lookup --max-draft-tokens 4`，可选`--adaptive-draft`。普通问答未必受益，当前不支持服务batch或Graph投机。

### HTTP / SSE

```powershell
.\.venv\Scripts\python -m mini_engine.server `
  --model .\models\Qwen3-0.6B --served-model-name qwen3-0.6b `
  --device cuda --dtype bf16 --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --fuse-projections --cuda-graph --prefill-chunk-size 128 --local-files-only
```

默认`http://127.0.0.1:8000/v1`。支持`GET /health`、`GET /v1/models`、`POST /v1/chat/completions`以及SSE与usage。请求使用`temperature: 0`，不支持tools、多模态或完整OpenAI API。默认Graph服务上下文上限为1024 tokens，超限请求提前拒绝。

另开PowerShell调用：

```powershell
$body = @{model='qwen3-0.6b'; messages=@(@{role='user'; content='解释KV Cache'}); max_tokens=64; temperature=0} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Python集成使用`mini_engine.model.NativeQwen3Runner`；batch会话通过`runner.create_batch_runtime()`创建，并在结束时关闭runtime和runner。早期仅转发reference的`LLMEngine/EngineConfig`已移除；启动配置统一为`RunnerConfig`。

## 性能 Benchmark

详细配置、逐次样本、数值边界和复现方法见[系统验证与性能评测](reports/VALIDATION_AND_BENCHMARKS.md)，全部工具见[benchmark目录](benchmarks/README.md)。以下均是原实验范围，不能互相拼接或当作行业排名。

| 实验 | 已记录结果 | 计时范围与限制 |
|---|---|---|
| FP16单请求Chat Graph，两prompt、64/128输出 | TPOT P50约2.53–2.66 ms；总输出342–358 TPS | 含prefill到末token，排除分词与文本decode；已预热，不是HTTP性能 |
| FP16单请求选择性LM head，512-token prompt | 额外allocated峰值150.63→21.13 MiB | forward P50为76.29→75.91ms；其他长度未稳定提速 |
| CUDA D128 PagedAttention | 相对原CUDA kernel观察1.74–1.80x | 配对Graph内单算子均摊，不是生成吞吐 |
| BF16 Prefix Cache | 冷/热/5-token后缀实际计算64/0/5 tokens | 输出一致、共享页不变；不据此推断通用加速比 |

Ragged Prefill已接入：CUDA直接读paged历史，其他dense后端合批投影/MLP后逐请求attention。FP16 B2的严格KV逐元素检查仍未通过，短样例输出一致不证明batch invariance；多请求forward时间只作为探索记录，不发布正式倍率。

```powershell
# 单请求eager / Graph，首轮与热启动分别记录
.\.venv\Scripts\python benchmarks\benchmark_chat_graph.py --model .\models\Qwen3-0.6B --dtype fp16 --ignore-eos
# 真实调度runtime的batch测量
.\.venv\Scripts\python benchmarks\benchmark_native_batch.py --help
# CPU/CUDA时间线，结果放build/profiles；带插桩时间不作为跑分
.\.venv\Scripts\python benchmarks\profile_native_batch.py --dtype fp16 --output-dir build/profiles/eager
```

主对照为Nano-vLLM与Wenyueh/MinivLLM：架构和固定源码差异已核对，共同离线adapter及同条件性能矩阵尚未完成，当前没有三方速度排名。vLLM/SGLang只作架构参考，不作为Windows运行依赖。

最新同输入PagedAttention组件测量中，Mini Triton在B1/B8短上下文更快，B32/2048时更接近。两边PyTorch版本不同，该结果不代表完整引擎TPS；数值对拍、条件和复现脚本见[当前对照记录](reports/VALIDATION_AND_BENCHMARKS.md)。

## 测试

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = '1'
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --dtype bf16 --http-smoke
```

pytest缓存统一写到`build/pytest_cache`。该环境变量开启显式受控的CUDA专项；其他测试也可能按CUDA是否可用运行GPU，不代表取消变量就强制CPU。完整模型验收需要本地权重和GPU。历史验收及本轮结构精简后的结果见[验证报告](reports/VALIDATION_AND_BENCHMARKS.md)。

本轮全量468项测试通过、2个既有warning、无跳过；真实BF16模型的chunk/Prefix/batch Graph及HTTP/SSE验收通过。本轮结构整理没有新增性能结论。

## 项目结构

```text
mini_engine/        仅推理运行代码
├── config.py      RunnerConfig与QuantConfig
├── cli.py         CLI/HTTP共享启动参数
├── model/         原生Qwen3、runner、Transformers对拍与结果类型
├── cache/         KV池、物理页管理、Prefix缓存
├── engine/        请求、调度、PrefillBatch、runtime和decode Graph
├── kernels/       CUDA源码、后端分发与数值reference
├── quantization/  量化reference与dandan W4A16 bridge
├── serving/       HTTP协议、单GPU worker和SSE传输
└── __main__.py / server.py    Chat与HTTP入口
benchmarks/        测量、对拍、profiling及共享实验工具
scripts/           Windows项目环境脚本
tests/            模型、算子、调度与服务回归
reports/           主题报告；实验JSON集中在data/
DEVELOPMENT_PLAN.md 当前优先级与开发约定
```

CLI直接运行单请求循环；HTTP worker使用`NativeBatchRuntime`。两者共享模型、KV与算子，调用方通过`finally`关闭自己创建的执行会话；资源所有权见[架构报告](reports/ARCHITECTURE.md)。模型、Toolkit、虚拟环境、缓存与build产物是本地运行依赖，均已忽略，不属于源码结构。

## 当前限制

- CUDA prefill仍以正确性为先；尚无算子库varlen Flash或统一prefill/decode混合GPU批次。
- Prefix只共享完整block；KV保守预留，尚无preemption/recompute或异步overlap scheduler。
- Graph覆盖一token decode与greedy，prefill、CPU调度、EOS和文本处理在图外。
- W4A16质量集与多形状性能待扩展；FP8 GEMM/KV、NVFP4/MXFP8未实现。
- 投机仅单请求greedy prompt lookup；小draft、随机校正及batch投机未实现。
- 服务是本地greedy子集；多GPU、PD分离、多模态和新模型架构继续后置。

## 许可证

[Apache-2.0](LICENSE)。Qwen3权重与tokenizer遵循其自身许可证。
