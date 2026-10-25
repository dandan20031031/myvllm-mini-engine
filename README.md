# MiniQwenEngine

MiniQwenEngine 是一个面向 Qwen3-0.6B 的 Windows 11 单 GPU 推理引擎，使用 Transformers 加载权重和 tokenizer，并通过项目自有 CUDA 扩展实现 Paged KV Cache、PagedAttention、数据移动和 elementwise 算子。

当前实现和后续方向见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，架构与实验依据见 [技术报告目录](reports/README.md)。项目仍处于研究和性能优化阶段，单算子 benchmark 不等同于 vLLM 的端到端性能结论。

## 已实现功能

### 推理引擎

- Qwen3-0.6B native forward 和增量 decode；
- Windows 11 原生 CUDA 扩展，支持 MSVC、nvcc 和 Ninja；
- 请求状态管理、BlockManager 和 continuous batching scheduler；
- Paged KV Cache tensor pool，支持跨 block 写入、回收和引用计数；
- Prefix Cache完整block共享、LRU、命中跳过计算和边界logits复用；
- decode优先的chunked prefill、token预算与KV容量预留；
- 单请求CLI、多轮Chat、batch decode，以及Windows HTTP/SSE服务；
- greedy prompt-lookup投机解码：多token验证、接受/拒绝与KV回退，可选接受率驱动draft预算。

### CUDA 算子

- CUDA PagedAttention，支持 FP16/BF16、GQA、非连续物理block、split-KV和D128向量化；
- CUDA RMSNorm、Residual + RMSNorm、Residual Add、SwiGLU 的 SiLU×Up；
- CUDA Embedding gather、RoPE 和分页 KV 写入；
- CUDA causal prefill，支持rectangular suffix与绝对位置causal mask；
- 可选CUDA greedy argmax（稳定tie/NaN规则，支持Graph）；
- 显式 CUDA Linear，底层使用 PyTorch 的 cuBLAS/cuBLASLt dispatch；
- CUDA Graph decode bucket，支持固定 buffer、padding slot 和 replay；
- QKV projection、Gate/Up projection fusion，减少小矩阵乘和 launch 次数。
- 可选 `dandan_ops` FP16 Linear 和 D=128 GQA Flash prefill，保持独立安装；
- 可选 W4A16 CUDA GEMV/Tensor Core，支持融合投影和 CUDA Graph decode。

### 精度和量化

- FP32、FP16、BF16 浮点路径；
- INT8 weight-only reference；
- packed INT4 reference；
- FP8 E4M3/E5M2 weight-only reference；
- NVIDIA FP8、NVFP4、MXFP8 能力探测；
- 量化路径区分 reference dequant 和真正的 Tensor Core GEMM。

INT4 可选择 `reference` 或 `dandan` 计算后端。后者使用真实 W4A16 CUDA kernel；是否加速取决于 batch/形状，需要实测。NVFP4、MXFP8、FP8/INT8 KV Cache 和 FP8 GEMM 仍在开发中。

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

### CUDA Graph 与 Projection Fusion

单轮推理和Chat命令均可加`--cuda-graph --sampling-backend cuda`，将decode和greedy采样捕获进CUDA Graph。要求`--decode-backend cuda --data-backend cuda`，FP16/BF16均可。首次使用某个bucket会有capture预热，后续轮次复用；`/clear`清空历史，退出自动释放Graph scratch。当前不能与`--speculative`同时使用。

```powershell
.\.venv\Scripts\python -m mini_engine `
  --model "D:\models\Qwen3-0.6B" `
  --prompt "解释 continuous batching" `
  --device cuda --dtype bf16 `
  --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --fuse-projections --local-files-only
```

### 可选 dandan_ops 后端

先按照独立算子库的构建说明，将 `dandan_ops` 安装到本项目 `.venv`，匹配 Python/PyTorch/CUDA ABI。引擎不自动下载或编译该库，也没有本机绝对路径依赖；本机已验证 PyTorch 2.9.1+cu128、CUDA 12.8、SM120、W4A16 contract v2。

在上面的启动命令中替换或添加以下参数即可：

| 模式 | 参数 |
|---|---|
| BF16 基线 | `--dtype bf16 --linear-backend cuda --prefill-backend cuda` |
| FP16 GEMM + Flash prefill | `--dtype fp16 --linear-backend dandan --prefill-backend dandan --fuse-projections` |
| W4A16 + Flash prefill | `--dtype fp16 --linear-backend cuda --prefill-backend dandan --quantization int4 --quant-backend dandan --quant-group-size 128 --fuse-projections` |

`dandan` Linear 允许库按形状选择 cuBLAS/HGEMM/TMA；`dandan_cuda` 强制自研 HGEMM。W4A16 支持源 group64/128，转换后存储group64，保留原始对称量化值；embedding、LM head、Norm 和 KV 保持 FP16。Flash prefill 严格要求 CUDA FP16、head_dim=128，decode 继续使用引擎分页 kernel。

### HTTP / SSE服务

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python -m mini_engine.server `
  --model .\models\Qwen3-0.6B --served-model-name qwen3-0.6b `
  --device cuda --dtype bf16 --prefill-backend cuda --decode-backend cuda `
  --elementwise-backend cuda --data-backend cuda --linear-backend cuda `
  --fuse-projections --cuda-graph --prefill-chunk-size 128 --local-files-only
```

默认监听`127.0.0.1:8000`。支持`GET /health`、`GET /v1/models`和`POST /v1/chat/completions`，包括`stream: true`与`stream_options: {include_usage: true}`。单GPU worker负责continuous batching；队列有界，断开连接会取消请求并回收页。接口是greedy聊天子集（`temperature: 0`），不支持tools、多模态或完整OpenAI API。默认Graph bucket覆盖到1024个上下文token，超过范围的服务请求会提前拒绝。

另开PowerShell调用：

```powershell
$body = @{model='qwen3-0.6b'; messages=@(@{role='user'; content='解释KV Cache'}); max_tokens=64; temperature=0} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

### 投机解码

在单轮/Chat启动命令中加入`--speculative prompt_lookup --max-draft-tokens 4`，可选`--adaptive-draft`。不需要额外draft模型；有重复文本的场景更可能受益。目前是单请求eager路径，尚未接入服务batch或Graph。普通问答可能无收益，性能脚本会检查与普通greedy的输出一致性。

### 运行测试

CPU/reference 回归：

```powershell
.\.venv\Scripts\python -m pytest -q -o cache_dir=build/pytest_cache
```

CUDA 回归：

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = "1"
.\.venv\Scripts\python -m pytest -q -o cache_dir=build/pytest_cache
```

2026-10-25本机全量回归为`403 passed, 2 warnings`，CUDA与HTTP测试实际执行；本轮重新验证完整0.6B BF16的分块/共享前缀、batch2/4 eager/Graph与并发HTTP/SSE，关闭后页引用归零。FP16 dandan和W4A16完整模型的既有验收记录保留原范围，详见[系统验证与性能评测](reports/VALIDATION_AND_BENCHMARKS.md)。后端数值验收可运行：

```powershell
.\.venv\Scripts\python benchmarks\validate_dandan_qwen3.py `
  --model "D:\models\Qwen3-0.6B" --linear-backend dandan_cuda `
  --prefill-backend dandan --fuse-projections --decode-steps 4
```

该验收区分量化变化和相同权重下的kernel误差。INT4可能改变模型输出，仍需评估任务质量。

启动配置、服务分层、资源清理与安装包检查的本轮证据见[结构维护验收摘要](reports/structure_validation_summary.json)。本轮未重新测量性能，不从结构调整推导加速比。

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

以下是优化前历史eager/Event结果，与项目PyTorch fallback比较：

| batch/context | CUDA PagedAttention | 相对项目 PyTorch fallback |
|---|---:|---:|
| 1 / 128 | 0.0474 ms | 7.97x |
| 8 / 512 | 0.0951 ms | 27.60x |
| 32 / 2048 | 0.6999 ms | 14.71x |

历史MiniVLLM报告为`0.019 / 0.017 / 0.377 ms`。新kernel需重新对齐其计时方式后再比较，不能用下方Graph结果替代eager数据。

新的D128向量化kernel与原kernel在同进程、相同输入下交替测量（FP16，block16，每Graph20次，20组样本）：

| batch/context | 原kernel Graph均摊中位数 | 新kernel Graph均摊中位数 | 配对比值 |
|---|---:|---:|---:|
| 1 / 128 | 33.99 us | 19.34 us | 1.76x |
| 8 / 512 | 72.38 us | 41.50 us | 1.74x |
| 32 / 2048 | 713.76 us | 400.30 us | 1.78x |

block16/32/64/128/256的15组测试均通过对拍。未独占GPU，这是本机算子观察值，不是服务加速或超越工业引擎的结论。复现：`python benchmarks/benchmark_paged_sweep.py`。

### 单请求 Chat CUDA Graph

RTX5080、FP16、融合投影、CUDA greedy、cuBLAS Linear，同一runner交替配对，两条prompt分别固定输出64/128 token、各6对，无profiler：

| 指标 | eager | CUDA Graph |
|---|---:|---:|
| TPOT P50 | 15.58–16.94 ms | 2.53–2.66 ms |
| 总生成吞吐 | 57.7–62.8 tokens/s | 342.1–357.5 tokens/s |

四组吞吐观察比5.65–6.05x，输出全部一致；TTFT未见一致改善。首次capture单列，测量期无capture且源码/后端指纹一致。TPOT分位数按请求统计；总TPS排除加载、tokenization和文本decode。GPU未严格独占，不代表行业排名或HTTP服务性能。[条件与原始数据摘要](reports/VALIDATION_AND_BENCHMARKS.md)。

```powershell
.\.venv\Scripts\python benchmarks\benchmark_chat_graph.py --model "D:\models\Qwen3-0.6B" --max-new-tokens 64 --max-new-tokens 128 --ignore-eos --warmup 2 --repetitions 6
```

### 历史单请求 Qwen3-0.6B

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

新后端对照可添加 `--dtype fp16 --fuse-projections --ignore-eos --prefix-cache disabled`，再分别选择 Linear/Prefill/量化后端。脚本记录实际输出token、重复样本P95及算子库文件指纹变化。本轮dandan性能样本受到GPU并发负载和库文件修改影响，不能与上表混算加速比；可提交数值摘要见 [验证记录](reports/stage22_backend_validation_summary.json)。

### 共享前缀、投机与HTTP对照

完整Qwen3-0.6B BF16实验中，64-token完整prefix命中后实际prefill计算64→0，5-token后缀只计算5个token，输出一致。重复句子投机样例32个输出的target调用32→16，总耗时中位537.39→300.87ms（观察1.79x）；普通KV问答的target调用仍为32次，几乎无收益。均为短样例，不能当作通用性能承诺。

```powershell
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --fuse-projections --http-smoke
.\.venv\Scripts\python benchmarks\benchmark_speculative.py --model .\models\Qwen3-0.6B --fuse-projections --ignore-eos
.\.venv\Scripts\python benchmarks\benchmark_http.py --base-url http://127.0.0.1:8000/v1 --model qwen3-0.6b --concurrency 4 --repetitions 3
```

HTTP工具记录实际usage、TTFT、TPOT估计、P50/P95、失败和吞吐；stream chunk不当作token。`--compare LEFT_JSON RIGHT_JSON`在环境manifest、实际输出或负载条件不一致时拒绝计算速度比。项目性能对照聚焦MiniVLLM；vLLM/SGLang只比较框架结构与源码设计，见[架构报告](reports/ARCHITECTURE.md)。HTTP工具仍可连接兼容服务，不要求安装工业引擎。

### 性能分析

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\profile_native_batch.py --dtype fp16 --output-dir build/profiles/eager
.\.venv\Scripts\python benchmarks\profile_native_batch.py --dtype fp16 --cuda-graph --output-dir build/profiles/graph
```

输出算子表、Chrome trace与摘要，分析CPU调度、launch、H2D/D2H及GPU kernel。真实模型eager/Graph trace已采集成功；profiling时间包含插桩开销。详细判读与Nsight命令见[推理引擎性能分析指南](reports/PROFILING.md)，算子库最新测试的适用边界见[算子后端性能复核](reports/BACKEND_PERFORMANCE_RECHECK.md)。

## 项目结构

```text
mini_engine/
├── __main__.py     单请求/Chat 命令行入口
├── cli.py          两个入口共享的模型启动参数与配置转换
├── config.py       RunnerConfig 静态约束；QuantConfig 精度约定
├── cache/          Paged KV Cache、block allocator、prefix cache
├── engine/         scheduler、request、batch runtime、CUDA Graph buckets
├── kernels/        PyTorch fallback、CUDA dispatch 和 .cu 源码
├── model/          Qwen3 forward、runner 和公共生成结果 outputs.py
├── quantization/   INT8/INT4/FP8 reference 与 dandan W4A16 bridge
├── serving/        protocol 请求校验、service 推理 worker、http JSON/SSE
└── server.py       HTTP 命令行入口与兼容导出

benchmarks/         单算子、单请求和 batch benchmark
tests/              CPU、CUDA、模型对齐和 runtime 回归
reports/            主题技术报告和实验摘要，入口README.md
PROJECT_STATUS.md   当前项目总览、已实现能力和后续方向
```

单请求/Chat经`__main__.py → cli.py/RunnerConfig → NativeQwen3Runner.generate_incremental → NativeQwen3`执行；HTTP经`server.py → serving.http → InferenceService → NativeBatchRuntime → Scheduler/NativeQwen3`执行。两者共享模型和算子，保留各自的生成循环；`server.py`继续兼容原有启动命令与类导入。

调用方用`finally`关闭自己创建的执行会话：HTTP worker关闭batch runtime，归还请求页、Prefix引用及自有Graph scratch；随后服务入口关闭runner。CLI退出关闭runner的Chat Graph和scratch。模型权重及KV tensor仍由runner持有，外部提供的Graph manager和padding页按借用关系处理，详见[架构与生命周期](reports/ARCHITECTURE.md)。2026-10-25本轮以结构维护为目标，Qwen3.5等新架构和脱离Transformers的独立权重加载器暂缓。

## 当前限制

- chunked prefill已完成；多个请求的prompt尚未打包为同一ragged batch，suffix验证仍需gather KV；
- Prefix只复用完整block；为防止KV耗尽采用保守最大预算预留，尚无preemption；
- CUDA Graph 覆盖单请求/Chat与batch的一token decode和greedy采样；prefill、CPU调度、EOS判断与文本处理在图外，非greedy batch采样也在图外；
- W4A16 已有可选 CUDA 后端，量化质量和多形状性能仍需评估；FP8 GEMM、FP8/INT8 KV、NVFP4/MXFP8 尚未完成；
- HTTP/SSE为本地greedy接口子集；尚无多机服务、安全网关和结构化输出；
- 投机仅greedy prompt lookup；小draft、EAGLE/MTP/DFlash、随机接受概率校正尚未实现；
- `dandan_ops` 为可选独立算子库；默认仍使用项目 CUDA/cuBLAS backend；
- MiniVLLM的同条件性能对照与完整量化质量集尚待完成；vLLM/SGLang仅作架构与源码对照，不承担性能跑分交付。

## 许可证

项目代码使用 Apache-2.0，详见 [LICENSE](LICENSE)。Qwen3 模型权重和 tokenizer 遵循其自身许可证。
