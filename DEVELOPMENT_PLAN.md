# MiniQwenEngine 开发方案

本文件是当前开发约定；路线变化同步更新，历史阶段报告保留当时状态。

## 2026-10-23：面向实习作品的执行计划（优先于下方历史顺序）

目标是能解释、复现和部署的单卡推理系统。项目不追求复制完整 vLLM/SGLang，也不把尚未测量的优化写成加速结果。保留旧阶段编号及未完成项，下面是本轮实施顺序；每阶段分别提交代码、测试与调用链报告。

### 行业对照与可行性

| 对照 | 可借鉴的机制 | 当前项目的缺口与落地范围 |
|---|---|---|
| [vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | 复用前缀KV，避免重复prefill | 现有共享物理页仍重算prompt；实现真正跳过、边界logits、LRU与活跃引用隔离 |
| [SGLang](https://arxiv.org/abs/2312.07104) | 前缀复用和请求调度协同 | 先复用已有完整block索引；不为小项目重写完整Radix树或声称已实现RadixAttention |
| [SGLang speculative](https://docs.sglang.io/docs/advanced_features/speculative_decoding) | NGRAM、多种draft、动态验证预算 | 无需训练的prompt lookup适配0.6B；真正多token验证与KV回退优先 |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) | 异步调度、batch invariant、确定性采样、confidence调度 | 借鉴调度/验收设计；DP/CP、MoE EP/EPLB及DeepSeek专用MTP不适合本项目单卡dense模型 |

DeepInfer当前开源目标模型为DeepSeek-V4.1-Flash，官方说明是offline inference，不能当作本机Qwen3公平跑分对象。vLLM/SGLang的外部对照允许另行采集，但必须记录OS、版本、模型、dtype、实际token和计时边界；不把不同系统/模型结果拼成同条件排名。

### 论文与实验选择

| 方法 | 来源与取舍 | 可行性/验收 |
|---|---|---|
| Chunked prefill、decode优先 | [SARATHI, 2023](https://arxiv.org/abs/2308.16369)、[FastGen, 2024](https://arxiv.org/abs/2401.08671) | 高；先分块调度和正确suffix attention。不同请求不合并GEMM时不能宣称复现论文的混合batch收益 |
| Split-KV decode | [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) | 高；已有partition+merge，继续做向量化、形状对照和block size sweep，属于工程改进而非新算法 |
| 精确投机解码 | [Speculative Sampling, 2023](https://arxiv.org/abs/2302.01318) | 高；greedy prompt lookup先落地。随机版必须有p/q接受率和拒绝分布校正，不能靠token相等判定 |
| 自适应draft预算 | 历史接受长度与验证成本驱动的项目实验 | 中；比较关闭/固定/自适应，报告无命中场景回退和额外成本。不称为DeepInfer confidence实现或论文原创 |
| EAGLE-3 | [2025论文](https://arxiv.org/abs/2503.01840) | 条件研究；需要与target匹配的特征draft及训练/权重，不能只增加命令行选项就算完成 |
| DFlash | [2026论文](https://arxiv.org/abs/2602.06036) | 条件研究；block diffusion并行draft需要专用模型与特征接口。0.6B目标很小，先测验证成本再决定训练投入 |
| 低精度KV | 独立于W4A16权重量化 | 中；先明确逐token/逐head scale、直接paged读取、误差与显存；不靠每层全量反量化声称加速 |

### 本轮实施阶段

| 阶段 | 交付与验收 | 状态 |
|---|---|---|
| 23 | Prefix命中实际跳过、多token suffix、decode优先chunk预算、共享页保护和内存压力处理 | 已实现并通过真实0.6B验收；64-token完整命中实际计算为0 |
| 24 | Prompt lookup greedy投机、KV回退、EOS/预算、固定/接受率自适应draft | 已实现；两条真实prompt逐token对拍通过。重复文本观察1.79x，普通问答无明显收益 |
| 25（承接14） | Windows HTTP chat completions子集/SSE、单GPU worker continuous batching、断线取消、有界队列 | 已实现；真实0.6B并发JSON/SSE通过，退出后KV引用为0 |
| 26（承接20） | D128 PagedAttention向量化、保留baseline、5种block size；恢复launch检查、CUDA greedy、构建hash隔离 | 已实现；FP16/BF16/Graph对拍通过，15组配对算子实验观察1.74–1.80x |
| 27（承接15/19/22） | HTTP公平计量/条件校验、模型集成验收和可提交实验摘要 | 工具与本机验收已完成；外部vLLM/SGLang实测、完整质量集和独占GPU性能采集仍待执行 |
| 28 | 真实引擎PyTorch Profiler/NVTX入口、CPU与GPU热点、Graph对照 | PyTorch eager/Graph采集与2项恢复/异常测试通过；Nsight已安装，但本执行环境CLI未生成report，硬件counter待采集 |

本轮完成可独立验收的23–27。旧计划的FP8 GEMM、INT8/FP8 KV、NVFP4/MXFP8、训练型draft仍是后续研究工作，不因本轮结束自动标记完成。它们分别需要独立kernel、模型质量验收以及Windows硬件/工具链证据；保留在阶段16/17/21中继续推进，不提供伪实现。已有W4A16不重复开发。

### 验收与简历表达

2026-10-24验收：325 tests通过（CUDA/Graph/HTTP实际运行），完整0.6B的BF16、FP16 dandan与W4A16三组chunk/prefix/batch2/4 Graph全部通过；BF16并发HTTP/SSE完成后KV引用为0。详见`reports/STAGE_27_REPRODUCIBLE_BENCHMARK.md`和紧凑JSON摘要。

- 核心证据：相同权重下输出/数值对拍，prefix实际执行token计数，speculative接受/拒绝回退，混合长短请求无饥饿，SSE取消后page引用归还。
- 性能矩阵：batch1/2/4/8，短/长prompt，共享/不共享前缀，greedy固定输出，BF16/cuBLAS及FP16/dandan/W4A16分别报告；不同精度不能混算优化收益。
- 统计单次请求TTFT、逐token间隔/TPOT、总TPS、P50/P95、cache hit tokens、显存；明确离线runtime与HTTP服务边界。
- GPU被其他任务占用或库文件变动时，结果只用于功能验证；不发布性能排名。原始结果放benchmarks/results，精简结果与方法放reports。
- 实习作品主线：Windows自研CUDA + 真实Paged KV/Graph + 共享前缀与调度 + 精确投机 + 可部署服务。用实际测得的收益写简历，不将论文收益或reference对比冒充超越工业引擎。

## 1. 项目目标和约束

面向 Qwen3-0.6B 实现单 GPU 推理引擎，重点是自研 CUDA、Paged KV Cache、多精度和可复现实验。先控制模型与分布式范围，再逐项提高性能。

- Windows 11 原生，PowerShell + MSVC + nvcc；不依赖 Linux、WSL、Triton。
- 项目源码、.venv、模型、编译输出、临时文件及可配置缓存均放在当前 D 盘项目目录。
- Transformers 负责加载和正确性参考；native forward 和缓存由项目实现。
- 不做多GPU、多模型通用适配；greedy prompt-lookup投机已实现，训练型draft继续独立评估。
- 每阶段完成必要测试、代码解读报告、README/计划更新和本地 Git commit。

## 2. 架构与数据流

~~~text
CLI / future HTTP
        |
LLMEngine -- Scheduler -- Request
        |          |
        |       BlockManager
        |
NativeQwen3Runner -- NativeQwen3 -- QuantBackend / ReferenceQuantLinear
        |                    |
    KVCachePool          CUDA kernels / PyTorch reference
~~~

单请求NativeQwen3Runner持有BlockManager和KVCachePool；create_batch_runtime接到Scheduler、NativeBatchRuntime和DecodeGraphManager。Prefill按请求分块并优先让decode推进；完整prefix可直接复用logits。HTTP线程通过单GPU worker提交请求。

KV 布局是 [layer, physical_block, token_in_block, kv_head, head_dim]。KV 精度当前与浮点模型一致；低精度权重不代表低精度 KV。

## 3. 全算子 CUDA 覆盖

每一类推理算子都应有明确 CUDA 后端和独立正确性证据。保留 PyTorch reference。融合后不要求每个逻辑操作单独 launch。

| 模块 | 目标 | 截至阶段 07 的状态 |
|---|---|---|
| Embedding | 自研 CUDA gather | PyTorch |
| Q/K/V/O projection | CUDA 库 GEMM/GEMV，评估形状专用优化 | 浮点 PyTorch Linear；可选低精度 reference |
| Input/Post-attention/Final RMSNorm | 自研 CUDA reduction | 已集成，可选 torch/cuda |
| Q/K head RMSNorm | 自研 CUDA | 已集成，可选 torch/cuda |
| Residual add | 自研或融合 | Attention residual 与 post norm 融合；MLP 后独立 CUDA add |
| RoPE/位置数据 | 自研 CUDA，复用位置数据 | PyTorch |
| KV 写入 | CUDA scatter，与 RoPE 等融合 | PyTorch copy |
| Prefill Attention | Windows Flash 风格 CUDA、保留 SDPA | SDPA；本机 Windows PyTorch 无 Flash backend |
| Decode Attention | 自研 CUDA Paged Attention | FP16/BF16、GQA、split-KV 已实现 |
| SwiGLU 的 SiLU×Up | 自研 fused CUDA | 已集成，包含对齐时的向量化读写 |
| Gate/Up/Down projection | CUDA GEMM/低精度后端 | 浮点 PyTorch Linear；可选低精度 reference |
| LM head | CUDA GEMM/GEMV | 浮点 PyTorch Linear |
| Greedy/Temperature/Top-k/Top-p | 自研 CUDA sampling | PyTorch reference |
| Pack/Dequant/Scale | CUDA + 对应格式 GEMM | INT8/INT4/FP8 reference |

矩阵乘先使用成熟 cuBLAS/cuBLASLt；按 profiling 评估 CUTLASS 或自研 GEMV/GEMM。所有 Linear 和 LM head 都纳入覆盖审计，但不把从零编写全部 GEMM 当作阶段前置条件。

Tokenizer、HTTP、请求队列和 block 所有权由 CPU 处理。

## 4. 多精度与 NVIDIA 格式

分别描述权重存储、激活、累加、KV 格式和实际计算后端。格式存在不等于 Tensor Core 加速。

| 格式 | 当前状态 | 下一步 |
|---|---|---|
| FP32 | 正确性 reference | 保留 |
| FP16/BF16 | 默认浮点及自研 kernel | 全算子覆盖 |
| TF32 | FP32 GEMM 策略，尚未提供项目级开关 | 显式控制和记录，不视为存储量化 |
| W8A16/W8A32 INT8 | per-channel 打包、FP32 scale、native Linear reference | fused dequant / CUDA GEMM |
| W4A16/W4A32 INT4 | reference 保留；可选 dandan W4A16 CUDA GEMV/Tensor Core，source group64/128 无损转换为存储group64 | 形状路由、模型质量和速度优化 |
| FP8 E4M3FN/E5M2 | scaled weight-only reference、native Linear，非 W8A8 | 评估 FP8 Tensor Core GEMM、激活缩放 |
| MXFP8 | reserved | 定义微缩放、布局、Windows 后端 |
| NVFP4 | reserved | FP4 E2M1、块 scale 和额外 scale 策略、Windows CUDA GEMM |
| FP8/INT8 KV | 未实现 | scale、写入和 paged attention 共同适配 |

RTX 5080 是 SM 12.0。PyTorch 的 float4 dtype 及硬件候选能力不能证明 NVFP4 已可用。不能以 INT4 或普通 FP8 冒充 NVFP4/MXFP8。每种格式分别验收：接口、reference、CUDA kernel、模型集成、性能收益。

当前 native 量化替换 decoder projection，保留浮点 embedding、norm、residual、lm_head 和 KV。reference 每次先反量化为 FP32 再 linear。可选 dandan 后端在 tile 内反量化，支持先融合 QKV/GateUp 再量化；存在 CUDA 计算路径不代表已取得端到端加速。

## 5. 阶段状态

| 阶段 | 交付内容 | 状态 |
|---|---|---|
| 00 | 配置、请求、调度和 block allocator | 完成 |
| 01 | KV tensor pool、benchmark 基础 | 完成 |
| 02 | Transformers 增量 reference | 完成 |
| 03 | native Qwen3、自有 KV、Windows CLI | 完成 |
| 04 | Windows CUDA Paged Attention | 完成 |
| 05 | 单算子 benchmark 与 smoke 实测 | 完成 |
| 06 | INT4/FP8 reference、native Linear、能力矩阵 | 完成 |
| 07 | 所有 RMSNorm、Residual、SwiGLU CUDA，边界/模型/Graph 测试及单算子实测 | 完成 |
| 08 | Embedding/RoPE/KV 写入 CUDA，复用 block table、减少每层同步 | 完成；正确性已验收 |
| 09+10 | Scheduler 接通 native model execution、batch decode、Prefix/Block Cache 生命周期 | 完成；prefill跳过在23完成 |
| 11 | CUDA Graph decode shape bucket、固定 metadata、runtime replay | 完成；真实 batch benchmark 已通过，持续优化 capture 覆盖 |
| 12 | Windows CUDA prefill correctness-first backend | 完成；尚不等同 FlashAttention 性能 |
| 13 | 显式 CUDA Linear/cuBLAS 边界、完整 Qwen3 验收和本地 Chat | 部分完成；INT4/FP8/NVFP4/MXFP8 高性能 GEMM 待做 |
| 14 | Windows HTTP/OpenAI/SSE 服务 | greedy聊天子集在25完成 |
| 15 | Transformers、具体 MiniVLLM、vLLM/SGLang/TensorRT-LLM 正式对比 | 待实现 |
| 16 | 精度感知 KV、FP8/INT8 KV、NVFP4/MXFP8 | 待实现 |
| 17 | 投机解码 | greedy prompt lookup/回退/接受率预算在24完成；小draft、随机校正和batch投机待做 |
| 18 | Projection Fusion：QKV、Gate/Up 合并矩阵乘，减少 decode GEMM/launch | 完成首轮；单请求和 batch Graph 已验证，继续优化 |
| 19 | 真实 batch CUDA Graph：eager/Graph 公平 TPOT、TPS、P50/P95 | 完成首轮；3 请求/32 token Graph benchmark 已通过，完整分位数和多 batch sweep 待扩展 |
| 20 | PagedAttention kernel：vectorized load、flash-decoding、block size sweep | D128向量化与5种block sweep在26完成；更广形状/partition调优待做 |
| 21 | 高性能量化：INT4/FP8 Tensor Core GEMM、FP8/INT8 KV、NVFP4/MXFP8 探索 | INT4 CUDA接入已在22完成；其余待实现，性能/质量继续评估 |
| 22 | 可选 dandan_ops：FP16 Linear、W4A16 bridge、GQA Flash prefill、融合/Graph验证和同口径 benchmark | 功能/数值验收完成，201 tests通过；公平性能重测待空闲GPU和固定库版本，见 reports/STAGE_22_DANDAN_BACKEND.md |

后续可按实测瓶颈调整算子顺序，并更新本文件。vLLM 不作为 Windows 引擎运行依赖；若比较环境使用不同 OS，必须披露差异，不能宣称严格同环境。

## 6. 正确性与性能验收

- shape/dtype/device、非整齐维度、空 batch、page 边界与坏 metadata。
- FP16/BF16 舍入点、FP32 reduction、小值和极值。
- 当前 CUDA stream、算子 CUDA Graph capture/replay。
- native logits 和 greedy，与 reference/Transformers 逐步对齐；量化需分开比较权重误差和缓存语义。
- TTFT、TPOT、实际输出 tokens/s、显存、并发和分位数延迟；量化记录困惑度或固定任务质量。
- 单算子记录 device timeline 和 eager wall time，端到端包含调度与同步；不混用口径。
- 所有阶段生成 reports/STAGE_XX_*.md，解释每个模块、数据布局、调用链、测试结果、限制和下一阶段接口。

2026-10-23阶段22全量回归为201 passed、2 warnings；CUDA测试在RTX 5080上实际执行通过。完整Qwen3-0.6B已有历史生成/Graph benchmark，并新增FP16 dandan、W4A16和Flash prefill短序列数值验收。本轮backend性能对照受并发负载与外部库修改影响，不作为正式加速结论。

阶段 05 Event 测量可能包括 Python 调度空隙，reference 包含 gather/同步/循环，历史加速比不能视为纯 kernel 或相对 vLLM 的结论。阶段 07 同时报告 Graph 重放和 eager wall；Graph 时间也包含节点调度，不称作纯指令时间。Paged Attention 和 elementwise 的加速数字是单算子或 reference 对比，不能直接外推为端到端 tokens/s。

算子 CUDA Graph 测试通过，不等于所有服务路径都已完成 CUDA Graph 性能验收。当前真实 batch benchmark 已通过 `NativeBatchRuntime.decode_step` 的 Graph replay 路径；旧的单请求 benchmark 脚本中的 `--cuda-graph` 仍只记录配置，正式 Graph 对照请使用 `benchmark_native_batch.py`。独立 residual add 在实测 shape 下略慢于 PyTorch，保留可选后端，后续以融合降低开销。

## 7. Git 工作流和存储

- 已配置 GitHub origin；按用户约定，每阶段在 `codex/` 分支完成验证和 commit，完成的工作合入 main 并同步远端。保留未完成的独立改动，不提交模型/缓存或凭据。
- d451ade：阶段 00–05 历史基线，不伪造早期分阶段历史。
- 082fd64：阶段 06 INT4/FP8 reference。
- 阶段 07 在测试、报告和计划检查完成后创建本地 commit；后续每阶段同样执行。
- 使用已有提交身份，不擅改凭据或全局 Git 设置。
- 不提交 .venv、模型、缓存、build 产物或大体积 trace；可在报告保存紧凑实测摘要。
- 测试失败不标记阶段完成；执行受权限阻塞时保留改动并说明未执行的动作。

## 8. 路线调整后的系统优先级

阶段 08 已完成 Embedding、RoPE 和 KV 写入 CUDA，136 个测试通过。阶段 09+10 已完成 batch decode runtime 和 Prefix block 生命周期，139 个测试通过。阶段 11/12/13 已完成 Graph runtime 接入、真实模型验收脚本、CUDA prefill/Linear backend 和本地 chat，完整 Qwen3-0.6B 已在 RTX 5080 BF16 全 CUDA backend 下通过真实生成验收。后续优先把已有 CUDA 算子贯通为系统能力：

| 阶段 | 重点 |
|---|---|
| 09 | Scheduler 接通 native model execution，真实 continuous batching、取消和回收 |
| 10 | Prefix/Block Cache，prefix hash、引用计数和共享 block |
| 11 | CUDA Graph decode，shape bucket 和固定 metadata |
| 12 | Windows Prefill CUDA，在线 softmax/causal/GQA | 完成 correctness-first backend |
| 13 | Linear/LM head 与 INT8/INT4/FP8 GEMM | 完成显式 CUDA Linear/cuBLAS 边界；量化 GEMM 待做 |
| 14 | Windows API、SSE、本地 RAG/代码助手（后置可选） | 待实现 |
| 15 | 完整 Qwen3 benchmark，与 Transformers、具体 MiniVLLM、vLLM/SGLang/TensorRT-LLM 对比 |
| 16 | 精度感知 KV、FP8/INT8 KV、NVFP4/MXFP8 能力 |
| 17 | 投机解码：greedy 多 token 验证、KV 回退、confidence-scheduled draft length、随机采样校正 |
| 18 | Projection Fusion：QKV、Gate/Up 合并 GEMM，先优化瘦 GEMM 的 launch 和权重读取 |
| 19 | CUDA Graph batch benchmark：把真实 scheduler/runtime 接入固定 bucket，测 eager 对照 |
| 20 | PagedAttention：vectorized KV load、split-KV/flash-decoding、block size 16/32/64/128/256 |
| 21 | 量化 GEMM/KV：INT4、FP8、FP8/INT8 KV，最后评估 NVFP4/MXFP8 的可用 Windows 路径 |

系统创新主线为：CUDA kernels + Paged KV Cache + Continuous Batching + Prefix Cache + CUDA Graph + Projection Fusion + Precision-aware KV + Windows local application。

### 性能优化执行顺序

2026-10-23 调整：18/19 首轮完成后，先执行阶段22接入现成的 dandan_ops，不要求引擎等待算子库最终优化。三种正式路径为 BF16 CUDA/cuBLAS 基线、FP16 dandan 可选后端、INT4+FP16 W4A16。保持默认后端不变；显式后端缺失能力/不兼容配置要提前报错。记录库源码、二进制及路由 profile 指纹；每次升级重新验证 logits/KV/Graph 和端到端性能。

2026-10-24更新：阶段23–26已补实际Prefix跳过、chunked prefill、greedy投机、HTTP/SSE、D128向量化和CUDA greedy；27提供真实模型验收和外部HTTP比较工具。接下来做直接paged多query验证、ragged batched prefill、低精度KV/FP8 GEMM与受控行业采集。下面18/19顺序保留为历史背景。

MiniVLLM 的 RTX 5080 profiler 显示，eager decode 的主要问题是 CPU kernel launch 空洞，GPU 时间主要集中在小 batch GEMM；其 PagedAttention 只占较小比例。因此优化顺序固定为：

1. **先做可验证的 projection fusion**：将每层 Q/K/V 合并为一次 QKV matmul，将 Gate/Up 合并为一次 matmul；保留未融合和量化 reference 回退路径，先验证 logits/greedy 一致，再测 TPOT。
2. **再做真实 batch CUDA Graph**：benchmark 已调用 `NativeBatchRuntime`，统一 batch、prompt、输出长度、dtype 和 block size；当前 3 请求/32 token 的首轮结果是 eager `189.3 tokens/s`、Graph `473.9 tokens/s`，约 `2.50x`，projection fusion + Graph 为 `519.2 tokens/s`。后续扩展 batch=1/2/4/8、P50/P95 和更长输出。
3. **随后优化 PagedAttention**：先做向量化 KV load 和 block size sweep，再做 batch=1 的 split-KV/flash-decoding；每个改动都要和当前 CUDA kernel 做对拍。
4. **最后做量化 GEMM 和低精度 KV**：INT4/FP8 Tensor Core GEMM、FP8/INT8 KV 需要分别记录权重误差、模型输出、显存和端到端收益；不能把当前 dequant reference 当作加速结果。

每个优化阶段都必须同时保留 baseline、正确性证据和可复现 benchmark，单算子加速不直接等同于端到端收益。

### 投机解码子路线

投机解码正式列为阶段 17，按以下顺序实现：

1. draft token 接口和目标模型多 token causal 验证；
2. greedy 接受连续前缀，拒绝位置回退 KV Cache；
3. prompt lookup / 小 draft model 两种候选来源；
4. confidence-scheduled draft length（动态 1～N）；
5. 随机采样接受概率和校正采样；
6. 记录 acceptance rate、每轮提交 token 数、额外显存和真实 TPOT。

投机解码必须先证明 greedy 输出与普通 decode 一致，再测加速；不能只比较草稿和目标模型是否采出相同 token 来实现随机采样版本。
