# MiniQwenEngine 开发方案

本文件是当前开发约定；路线变化同步更新。技术报告按主题合并至 [技术报告目录](reports/README.md)，阶段编号用于关联历史交付和实验文件。

## 2026-10-25：主对照调整为 Nano-vLLM 与 MiniVLLM

两个小型引擎同时对比架构与性能：Nano-vLLM采用[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，MiniVLLM采用[Wenyueh/MinivLLM](https://github.com/Wenyueh/MinivLLM)。vLLM/SGLang保留为工业架构参考，不要求性能跑分。固定源码、区别上游和本地补丁；不以名称相近的其他mini-vllm仓库代替。

| 交付 | 验收与现状 |
|---|---|
| 源码结构对照 | 已核对Nano本地7074c188/upstream快照bb823b3e、Mini上游00db3b47/Windows分支59466f60；调度、ragged prefill、prefix、抢占、Graph边界、加载和服务差异并入架构报告 |
| 既有性能证据审计 | 已核对本项目FP16单请求、Mini实际FP32批量历史报告、Nano官方4070 Laptop结果；条件不一致，不计算三方倍率或排名 |
| 共用离线benchmark adapter | 待实现；统一token IDs、模型哈希、实际权重/KV dtype、固定输出、计时边界和逐次样本。HTTP只测本项目服务，不用缺少同类HTTP接口的对照实现作伪服务对比 |
| Windows可行性 | Mini已有本地Windows/Graph修复；须显式选择worktree并纠正迁移后的editable导入路径。Nano当前仍有NCCL/flash-attn依赖，尚无本机可运行GPU基线；不为跑分引入Linux/WSL或静默替换attention |
| 统一性能矩阵 | 首轮FP16、TP1、B=1/4/8/16、prompt=128/512、output=128，分别eager/Graph。Nano/Mini原采样拒绝greedy，使用共同temperature=0.6、top-k禁用/top-p=1与固定输出；任何greedy或ignore-EOS适配都记录补丁 |
| 下一项引擎优化 | 优先评估prefill只对最后一行及Prefix边界行计算LM head，随后补真正ragged/paged prefill；完成各自正确性和独立消融后才宣称加速 |

完整协议、已有数值与待补条件见[性能评测报告](reports/VALIDATION_AND_BENCHMARKS.md)和[benchmark目录](benchmarks/README.md)。本次仅完成源码与历史性能证据审计，没有运行新的三方性能实验。下面旧计划中仅写MiniVLLM的对照目标均扩展为Nano-vLLM与MiniVLLM；原实验数据保留原范围。

## 2026-10-25：先完成现有项目的结构维护

本轮优先整理已可运行的Qwen3-0.6B引擎。Qwen3.5等新架构、混合注意力/多模态和脱离Transformers的独立权重加载器暂缓，继续使用Transformers加载权重、tokenizer和正确性参考。

| 工作 | 当前落地与验收边界 |
|---|---|
| 统一启动配置 | `RunnerConfig`负责模型/后端/精度的静态组合约束，`cli.py`共享参数注册与转换；Chat、HTTP保留各自的请求/服务参数，硬件和算子能力由runner检查 |
| 分离服务职责 | `serving/protocol.py`校验请求与定义错误，`service.py`管理worker和队列，`http.py`负责连接及JSON/SSE；`server.py`保留兼容导入和原启动命令 |
| 提取公共结果 | `model/outputs.py`统一native/reference的`IncrementalGenerationResult`，不借此更换模型执行循环 |
| 集中资源清理 | batch会话提供`close()`，明确请求页、Prefix、Graph/scratch的自有与借用关系；CLI、worker与服务入口通过`finally`依次退出 |
| 验收与报告 | 已完成两个入口参数、旧导入兼容性、启动失败/关闭/重复关闭、借用Graph资源回归；全量403 passed、2 warnings，真实BF16模型chunk/prefix/batch2/4 Graph/HTTP-SSE通过，见主题报告与紧凑摘要 |

该调整以职责边界和可维护性为验收目标，不声称增加吞吐或减少延迟。保持已有命令、历史实验数据和阶段记录；不新增一套重复的阶段报告。以下历史阶段状态与性能证据保留原实验范围。

本轮结构维护已验收，benchmark、profiling与系统验证脚本统一调用`runtime.close()`；每轮性能测量间的Prefix清空仍保留在原计时边界外。后续先补Nano-vLLM/MiniVLLM共同benchmark与选择性LM head，再推进ragged/paged prefill；低精度KV按profiling结果后置，暂不扩大模型架构范围。

## 2026-10-23：面向实习作品的执行计划（优先于下方历史顺序）

目标是能解释、复现和部署的单卡推理系统。项目不追求复制完整 vLLM/SGLang，也不把尚未测量的优化写成加速结果。保留旧阶段编号及未完成项，下面是本轮实施顺序；每阶段提交代码、测试，并更新对应主题的调用链与验收记录。

### 行业对照与可行性

| 对照 | 可借鉴的机制 | 当前项目的缺口与落地范围 |
|---|---|---|
| [vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | 复用前缀KV，避免重复prefill | 本项目已完成完整block命中跳过、边界logits、LRU与活跃引用隔离；对照block hash索引与当前tuple扫描的设计取舍 |
| [SGLang](https://arxiv.org/abs/2312.07104) | 前缀复用和请求调度协同 | 先复用已有完整block索引；不为小项目重写完整Radix树或声称已实现RadixAttention |
| [SGLang speculative](https://docs.sglang.io/docs/advanced_features/speculative_decoding) | NGRAM、多种draft、动态验证预算 | 无需训练的prompt lookup适配0.6B；真正多token验证与KV回退优先 |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer) | 异步调度、batch invariant、确定性采样、confidence调度 | 借鉴调度/验收设计；DP/CP、MoE EP/EPLB及DeepSeek专用MTP不适合本项目单卡dense模型 |

2026-10-25范围调整：Nano-vLLM与MiniVLLM均做架构及同机、同精度、同负载的性能对照；vLLM、SGLang只做框架结构和源码设计参考，记录所读commit、模块职责、请求调用链、调度/KV/执行接口及本项目的取舍，取消两者性能跑分交付。Transformers用于正确性参考。HTTP采集工具保留为通用能力，服务稳定性在本项目独立验收。

DeepInfer保留为异步调度、确定性与投机机制的阅读参考，不纳入Qwen3速度排名；多GPU、MoE与PD分离继续后置。源码对照不代表实现了对方所有能力，不以文件数量或代码行数评价优劣；结论与方法更新至`reports/ARCHITECTURE.md`。

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
| 27（承接15/19/22） | HTTP公平计量/条件校验、模型集成验收和可提交实验摘要 | 工具与本机验收已完成；Nano-vLLM/MiniVLLM同口径实测、完整质量集和受控性能采集仍待执行；vLLM/SGLang改为源码结构参考 |
| 28 | 真实引擎PyTorch Profiler/NVTX入口、CPU与GPU热点、Graph对照 | PyTorch eager/Graph采集与2项恢复/异常测试通过；Nsight已安装，但本执行环境CLI未生成report，硬件counter待采集 |
| 29 | CLI Chat完整decode Graph、固定pinned/GPU metadata、图内greedy、同runner交替配对TTFT/TPOT/P50/P95 | 功能完成；FP16/BF16跨页/多轮/异常回收与batch缩容对拍，全量356 tests通过；真实模型结果见主题验收报告 |

阶段29后优先级：结构维护与403项回归已完成，Nano-vLLM/MiniVLLM源码差异已核对；下一步补共同离线计量接口、选择性LM head与同条件性能数据，再深化ragged/paged prefill。本项目服务稳定性独立验收；PD分离、多GPU与训练型draft继续后置。

本轮完成可独立验收的23–27。旧计划的FP8 GEMM、INT8/FP8 KV、NVFP4/MXFP8、训练型draft仍是后续研究工作，不因本轮结束自动标记完成。它们分别需要独立kernel、模型质量验收以及Windows硬件/工具链证据；保留在阶段16/17/21中继续推进，不提供伪实现。已有W4A16不重复开发。

### 验收与简历表达

2026-10-24验收：325 tests通过（CUDA/Graph/HTTP实际运行），完整0.6B的BF16、FP16 dandan与W4A16三组chunk/prefix/batch2/4 Graph全部通过；BF16并发HTTP/SSE完成后KV引用为0。详见[系统验证与性能评测](reports/VALIDATION_AND_BENCHMARKS.md)和紧凑JSON摘要。

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
- 每阶段完成必要测试、对应主题报告、README/计划更新和本地 Git commit。

## 2. 架构与数据流

~~~text
启动共享: cli.add_runner_arguments -> RunnerConfig -> NativeQwen3Runner.from_pretrained

CLI/Chat: __main__ -> runner.generate_incremental -> NativeQwen3 / Chat Graph
HTTP: serving.http -> InferenceService worker -> NativeBatchRuntime.step
                                                  |              |
                                              Scheduler     NativeQwen3
                                                  |              |
                                          BlockManager / KVCachePool / PrefixCache

共同后端: CUDA kernels / PyTorch-cuBLAS / optional dandan_ops / quantization
~~~

两条路径共享模型、缓存实现与算子，生成循环仍各自独立。早期`LLMEngine`保留为facade/reference接口，不是原生Chat或HTTP的必经入口。`server.py`只保留服务命令行启动与兼容导出；服务协议、worker、传输分别归`serving`的三个模块。

NativeQwen3Runner持有模型、BlockManager和KVCachePool；`create_batch_runtime()`创建使用它们的可关闭batch会话。Prefill按请求分块并优先让decode推进；完整prefix可直接复用logits。runtime.close归还会话请求页、Prefix引用和自有Graph资源，外部提供的manager/padding页遵守借用边界；调用方应先停止worker并关闭runtime，再关闭或复用runner。CLI的finally关闭runner持有的Chat Graph与scratch。

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
| 15 | Transformers正确性参考，Nano-vLLM/MiniVLLM架构与性能；vLLM/SGLang源码参考 | 两个小型引擎的共同adapter/同口径实测待完成；不要求工业引擎跑分 |
| 16 | 精度感知 KV、FP8/INT8 KV、NVFP4/MXFP8 | 待实现 |
| 17 | 投机解码 | greedy prompt lookup/回退/接受率预算在24完成；小draft、随机校正和batch投机待做 |
| 18 | Projection Fusion：QKV、Gate/Up 合并矩阵乘，减少 decode GEMM/launch | 完成首轮；单请求和 batch Graph 已验证，继续优化 |
| 19 | 真实 batch CUDA Graph：eager/Graph 公平 TPOT、TPS、P50/P95 | 完成首轮；3 请求/32 token Graph benchmark 已通过，完整分位数和多 batch sweep 待扩展 |
| 20 | PagedAttention kernel：vectorized load、flash-decoding、block size sweep | D128向量化与5种block sweep在26完成；更广形状/partition调优待做 |
| 21 | 高性能量化：INT4/FP8 Tensor Core GEMM、FP8/INT8 KV、NVFP4/MXFP8 探索 | INT4 CUDA接入已在22完成；其余待实现，性能/质量继续评估 |
| 22 | 可选 dandan_ops：FP16 Linear、W4A16 bridge、GQA Flash prefill、融合/Graph验证和同口径 benchmark | 功能/数值验收完成；后续系统验收及公平性能复核见[系统验证与性能评测](reports/VALIDATION_AND_BENCHMARKS.md) |

后续可按实测瓶颈调整算子顺序，并更新本文件。所有对照实现均不作为引擎运行依赖；Nano-vLLM/MiniVLLM跑分须记录硬件/OS/软件版本、实际dtype、输入输出、补丁与计时边界。平台不支持时记录未测，不以替代实现冒充原版。

## 6. 正确性与性能验收

- shape/dtype/device、非整齐维度、空 batch、page 边界与坏 metadata。
- FP16/BF16 舍入点、FP32 reduction、小值和极值。
- 当前 CUDA stream、算子 CUDA Graph capture/replay。
- native logits 和 greedy，与 reference/Transformers 逐步对齐；量化需分开比较权重误差和缓存语义。
- TTFT、TPOT、实际输出 tokens/s、显存、并发和分位数延迟；量化记录困惑度或固定任务质量。
- 单算子记录 device timeline 和 eager wall time，端到端包含调度与同步；不混用口径。
- 实现与调用链更新至 `reports/ARCHITECTURE.md`，验收与计时口径更新至 `reports/VALIDATION_AND_BENCHMARKS.md`；保留必要JSON实验摘要，专项后端复核与性能分析指南独立维护，不再逐阶段新增重复报告。

2026-10-23阶段22全量回归为201 passed、2 warnings；CUDA测试在RTX 5080上实际执行通过。完整Qwen3-0.6B已有历史生成/Graph benchmark，并新增FP16 dandan、W4A16和Flash prefill短序列数值验收。本轮backend性能对照受并发负载与外部库修改影响，不作为正式加速结论。

阶段 05 Event 测量可能包括 Python 调度空隙，reference 包含 gather/同步/循环，历史加速比不能视为纯 kernel 或相对 vLLM 的结论。阶段 07 同时报告 Graph 重放和 eager wall；Graph 时间也包含节点调度，不称作纯指令时间。Paged Attention 和 elementwise 的加速数字是单算子或 reference 对比，不能直接外推为端到端 tokens/s。

算子 CUDA Graph 测试通过，不等于所有服务路径都已完成 CUDA Graph 性能验收。真实 batch benchmark 调用`NativeBatchRuntime`；阶段29已修复单请求脚本的`--cuda-graph`为真实执行，并拒绝伪batch参数。推荐用`benchmark_chat_graph.py`做同runner交替配对的单请求实验，用`benchmark_native_batch.py`测batch。独立 residual add 在实测shape下略慢于PyTorch，后续以融合降低开销。

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
| 15 | 完整Qwen3验收、Nano-vLLM/MiniVLLM架构与性能；vLLM/SGLang只读源码参考 |
| 16 | 精度感知 KV、FP8/INT8 KV、NVFP4/MXFP8 能力 |
| 17 | 投机解码：greedy 多 token 验证、KV 回退、confidence-scheduled draft length、随机采样校正 |
| 18 | Projection Fusion：QKV、Gate/Up 合并 GEMM，先优化瘦 GEMM 的 launch 和权重读取 |
| 19 | CUDA Graph batch benchmark：把真实 scheduler/runtime 接入固定 bucket，测 eager 对照 |
| 20 | PagedAttention：vectorized KV load、split-KV/flash-decoding、block size 16/32/64/128/256 |
| 21 | 量化 GEMM/KV：INT4、FP8、FP8/INT8 KV，最后评估 NVFP4/MXFP8 的可用 Windows 路径 |

系统创新主线为：CUDA kernels + Paged KV Cache + Continuous Batching + Prefix Cache + CUDA Graph + Projection Fusion + Precision-aware KV + Windows local application。

### 性能优化执行顺序

2026-10-23 调整：18/19 首轮完成后，先执行阶段22接入现成的 dandan_ops，不要求引擎等待算子库最终优化。三种正式路径为 BF16 CUDA/cuBLAS 基线、FP16 dandan 可选后端、INT4+FP16 W4A16。保持默认后端不变；显式后端缺失能力/不兼容配置要提前报错。记录库源码、二进制及路由 profile 指纹；每次升级重新验证 logits/KV/Graph 和端到端性能。

2026-10-25更新：阶段23–29及结构维护已验收。主对照为Nano-vLLM/MiniVLLM架构与性能，工业框架只作源码参考；新增选择性LM head评估，再深化paged多query/ragged prefill。下面18/19顺序保留为历史背景。

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
