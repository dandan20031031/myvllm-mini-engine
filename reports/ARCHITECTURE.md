# 推理引擎架构与实现

截至2026-10-24，整合原阶段00–29的实现说明。范围为Windows 11单GPU、Qwen3-0.6B；HuggingFace负责权重加载、tokenizer和正确性参考，native forward、缓存、调度与CUDA扩展由项目实现。运行命令见 [项目 README](../README.md)，证据见 [系统验证与性能评测](VALIDATION_AND_BENCHMARKS.md)。

## 调用链与代码入口

```text
CLI / Chat -> NativeQwen3Runner.generate_incremental -> NativeQwen3 / DecodeGraphBucket
HTTP -> InferenceService单GPU worker -> NativeBatchRuntime.step -> Scheduler
                                       |                         |
                                 NativeQwen3                GenerationRequest
                                       |                         |
                                  KVCachePool <--------- BlockManager / PrefixCache
                                       |
                        CUDA kernels / cuBLAS / optional dandan_ops
```

两条生成路径共享模型、KV与算子实现，但CLI/Chat直接执行单请求循环，HTTP/batch经调度runtime执行；不能把示意图理解为所有入口已收敛到同一个执行循环。`engine/llm_engine.py`保留早期facade/reference接口，并非当前原生Chat和服务的必经热路径。

| 入口 | 职责 |
|---|---|
| `mini_engine/model/native_runner.py` | 加载、配置校验、单请求生成、创建batch runtime |
| `mini_engine/model/native_qwen3.py` | Embedding、Q/K RMSNorm、RoPE、GQA、SwiGLU及prefill/decode/suffix forward |
| `mini_engine/cache/` | 物理页引用、逻辑块表、KV tensor和完整prefix索引 |
| `mini_engine/engine/request.py`、`scheduler.py`、`runtime.py` | 请求状态、token预算、decode优先执行、采样与回收 |
| `mini_engine/engine/cuda_graph.py` | 固定形状bucket、scratch页、capture/replay |
| `mini_engine/kernels/`、`quantization/` | Torch reference、CUDA与可选dandan后端 |
| `mini_engine/speculative.py`、`server.py` | 单请求greedy投机、HTTP/SSE及worker生命周期 |

## KV所有权、Prefix与调度

KV布局为 `[layer, physical_block, token_in_block, kv_head, head_dim]`。`BlockManager`管理allocate/attach/grow/pin/unpin/truncate，`KVCachePool`保存tensor；请求引用与Prefix Cache引用独立。LRU淘汰只撤销缓存pin，活跃请求持有的页继续有效。

Prefix仅共享完整block。部分命中从 `position=prefix_len` 计算suffix；完全命中且有对应边界logits时跳过forward。非对齐prompt保存最后一个完整block末尾那一行logits；不能把整个prompt末尾logits挂在较短的prefix key上。admission重新查询缓存，避免等待期间淘汰导致旧页号失效；替换entry先pin新页再unpin旧页。完整共享页不写回。

每轮先执行decode，再用剩余 `max_num_batched_tokens` 预算推进prefill；`prefill_chunk_size`限制单请求chunk。decode超预算时轮转。调度器按请求最大KV需求做保守虚拟预留，物理页按需分配；永久超容量请求报错，暂时不足等待。最终输出token不再送回模型，KV需求为 `prompt_len + max_new_tokens - 1`。EOS、取消和执行/采样异常均释放请求引用。

模型接口 `model(suffix_ids, cache, block_ids, position=n)` 返回 `[suffix_length, vocab_size]`。单token且 `position>0` 直接读paged KV；多token suffix在写入新KV后gather前缀和suffix，再做 `query_start=n` 的绝对位置causal attention。CUDA/dandan保持原生GQA，Torch分支按需扩展KV heads。

当前chunk仍逐请求执行，多query验证仍gather dense KV；尚未实现ragged batched prefill、直接paged多query attention或安全preemption。最大预算预留会限制高输出上限场景的并发。

## CUDA Graph与投影融合

Graph bucket固定batch size、block table宽度和tensor地址，与KV物理block是不同概念。`DecodeGraphManager`选择能容纳活跃请求的bucket；replay更新metadata后重放decode forward。CPU调度和请求进出仍在图外；新建batch Graph和单请求Chat Graph均捕获greedy采样，非greedy batch采样使用logits在图外处理。

CLI/Chat默认eager，显式`--cuda-graph`进入`generate_incremental(use_cuda_graph=True)`。先运行prefill并采样首token；若还需decode，再通过`_chat_bucket()`延迟capture，后续调用`replay_metadata(..., sample=True)`得到图内argmax token。`item()`只传回token用于EOS与文本处理，不再额外逐token进行device-wide同步。首轮EOS或只生成1 token时不capture；首次capture开销计入该请求的total/TPOT，热启动benchmark单独排除首次调用。

单请求一次预留`prompt_len + max_new_tokens - 1`所需物理页，跨页decode不需grow或修改表。runner保留B1 manager和一页独立scratch，跨Chat轮次与`/clear`复用bucket；每次请求的live页在finally释放，CLI退出调用`close()`同步并释放Graph与scratch。`/clear`清空消息历史；多轮仍会重新prefill完整消息，不表示会话KV复用。

每个bucket持有固定GPU metadata和pinned CPU staging。token/position合并为一次H2D copy，block table内容不变时跳过其H2D；活跃行减少或表变短时补齐scratch与-1。下一次复用host staging前等待上次H2D event；跨stream使用等待旧stream完成。输出仍为固定buffer视图。batch runtime直接提交Python metadata，不再先构造临时GPU token/position/table再拷入bucket。`profile_native_batch.py`同时标记旧`replay`和新`replay_metadata`入口。

capture本身也会写KV，必须使用与live页分离的scratch页；每个inactive slot的第0列指向不同物理页，padding context_len为1。Graph返回固定logits buffer的视图，跨replay保存或对拍时需clone。切换后端/权重需重新capture。公开入口保留metadata检查，可信Graph路径只跳过会同步到CPU的检查；CUDA launch错误检查已恢复并通过capture/replay回归，旧阶段“关闭launch检查”的说明不再适用。

`FusedLinear`沿输出维拼接Q/K/V或Gate/Up权重，一次矩阵乘后split，减少GEMM和dispatch次数。Q/K norm、RoPE、KV写入及SiLU×Up仍是后续算子。需要量化时先融合再量化，已量化模型拒绝事后融合；普通CUDA Linear底层仍是PyTorch/cuBLAS/cuBLASLt。

## CUDA与量化后端

| 模块 | 实现与约束 |
|---|---|
| PagedAttention | FP16/BF16、GQA、非连续物理页、split-KV；D128使用Pack4对齐访问和warp归约，其他维度走原实现；`cuda_baseline`保留实验对照 |
| Elementwise/data | RMSNorm、Residual+RMSNorm、Residual Add、SiLU×Up、Embedding、RoPE和分页KV scatter；保留Torch reference |
| Prefill/suffix | 自有CUDA支持矩形Q/K及绝对位置causal mask；correctness kernel最长8192；可选dandan dense Flash |
| Greedy采样 | 显式可选CUDA两级argmax，覆盖tie/NaN/Inf和词表尾部；仅temperature=0，默认Torch仍支持原采样 |
| 构建 | `cuda_extension.py`按源码、PyTorch、Toolkit与目标设备生成hash，隔离Windows DLL/module路径；不自动删除旧进程可能正在使用的产物 |

`dandan_ops`独立安装、延迟导入，提前检查FP16/CUDA、设备、能力和W4A16 contract v2；引擎记录Python、binary与route profile的SHA256。`dandan` Linear允许库按形状选择cuBLAS/HGEMM/TMA，`dandan_cuda`强制HGEMM。Flash要求FP16、D128，支持GQA与offset suffix；它读取dense K/V，单token decode继续用引擎PagedAttention。

| 精度路径 | 当前状态 |
|---|---|
| FP32/FP16/BF16 | 浮点reference或CUDA路径 |
| INT8、INT4、FP8 E4M3/E5M2 reference | packed weight-only存储，每次FP32反量化后Linear；不代表低精度Tensor Core GEMM |
| dandan W4A16 | decoder projection使用INT4、激活FP16，真实CUDA GEMV/Tensor Core计算；embedding、Norm、LM head和KV保持浮点 |
| FP8 GEMM、FP8/INT8 KV、NVFP4/MXFP8 | 尚未实现可用计算路径；后两者为reserved |

W4 bridge不重新量化：将signed nibble映射到unsigned、zero=8，打包为8 nibble/int32；源group64/128转换为存储group64，group128的scale重复。先准备所有替换再修改模型，checkpoint校验packing/version。数值oracle必须使用相同packed state、FP16舍入后的反量化权重；`storage_bytes`仅是投影权重及scale/zero，不能当整模型显存。存在CUDA量化计算不等于已获得端到端加速。

## Greedy投机解码

`prompt_lookup()`匹配历史后缀的最长ngram，同长度优先最近位置，复制其后已有token。`generate_speculative()`先正常prefill输出首token，再将 `[pending_token, draft...]` 一次送目标模型；接受连续匹配草稿，首处拒绝输出纠正token，全接受则输出bonus。无命中退回普通greedy；草稿长度受剩余输出预算限制，为纠正/bonus预留位置。

拒绝后仅保留pending和已接受草稿的逻辑KV前缀；`truncate`归还完整尾页，保留页内旧内容由后续forward覆盖，attention不得读取逻辑长度之外。EOS、长度上限和异常清理同样生效。`AcceptanceBudget`使用最近8个有提案轮次的接受比例调整预算，没有模型置信度信号。

目前为单请求eager、无额外draft模型；未接随机分布校正、batch/Graph、HTTP投机或训练型draft。多token验证成本和浮点舍入均需独立评估，不能从greedy短样例推导通用质量与收益。

## HTTP/SSE与资源生命周期

`server.py`提供 `/health`、`/v1/models`、`/v1/chat/completions`。主线程加载一次模型；单GPU worker负责tokenizer、runtime、调度、KV和模型调用。HTTP线程只校验和收发，经有界提交/结果队列交互，新请求可加入正在decode的runtime。

服务是greedy聊天子集：`temperature=0`，messages接受system/user/assistant非空字符串，未支持字段报400。SSE依次发送role/content、finish、可选 `choices=[]` usage及 `[DONE]`；usage统计实际生成token，包含文本隐藏的特殊token。服务默认仅监听127.0.0.1，无认证/TLS或多租户隔离；没有tools、多模态、随机采样或服务端会话记忆。

默认推理请求上限32、每请求结果队列32、HTTP连接线程64、body 1MiB、messages 128项、socket超时5秒、请求超时120秒、max_tokens上限512，均受配置及上下文/KV约束。Graph开启后还受bucket并发和上下文覆盖范围约束。

断线、慢消费者和超时触发worker取消并回收页；慢SSE不会阻塞GPU worker。输入错误只终止对应请求，无法归因的batch执行异常终止整个active batch；不可恢复GPU错误可能需重启。关闭时释放active/waiting、Prefix pin和Graph scratch引用，正在执行的CUDA kernel须返回后才能完成退出。

## 与vLLM、SGLang的框架及源码对照

2026-10-25按项目范围调整：两者只作结构与代码设计参考，性能对照保留给MiniVLLM。以下是源码阅读结论，不是安装、移植、运行或速度验证；不比较代码行数，也不将同名功能视为同等覆盖。

为避免main漂移，固定所读版本：vLLM [`5f63ea46`](https://github.com/vllm-project/vllm/commit/5f63ea461ccd567ece6f50dd3701430c0f321039)，SGLang [`773cb938`](https://github.com/sgl-project/sglang/commit/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed)。本地依据为阶段29提交`66dc76d`；后续源码变更需重新核对路径与配置分支。

### 先看职责分工

vLLM V1把请求处理、EngineCore调度、Executor/Worker和ModelRunner分层；Executor也有单进程实现，不应理解为只能多卡。SGLang Python SRT围绕TokenizerManager、Scheduler、ScheduleBatch/ForwardBatch、ModelRunner组织工作，normal/overlap是不同执行模式。进程与通信设计服务于更大的模型、设备和部署范围；我们的单GPU worker保持同进程，先借鉴接口边界。[vLLM架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)、[SGLang服务入口](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/entrypoints/engine.py#L892)

| 对照点 | vLLM所读实现 | SGLang所读实现 | 本项目当前实现与取舍 |
|---|---|---|---|
| 请求执行 | [`EngineCore.step/step_with_batch_queue`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/engine/core.py#L657) 经Executor执行，支持非阻塞调用与在途批次 | [`Scheduler.event_loop_normal/overlap`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/managers/scheduler.py#L1995) 管接收、选批、执行与结果处理 | [`InferenceService._run`](../mini_engine/server.py) 同步推进runtime；HTTP线程与GPU worker已经分开，但没有CPU调度与GPU批次执行的完整overlap |
| 调度模型 | [`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/core/sched/scheduler.py#L677) 以已计算token与目标token进度统一分配工作，并处理抢占/恢复 | [`PrefillAdder`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/managers/schedule_policy.py#L584) 管输入/chunk/KV等预算，ScheduleBatch支持内存检查与retract | [`Scheduler.schedule`](../mini_engine/engine/scheduler.py) 显式区分decode与prefill，虚拟预留最坏输出增长；设计简单、准入保守，无安全preemption/recompute |
| 执行批次 | [`GPUModelRunner._prepare_inputs`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/worker/gpu_model_runner.py#L1924) 构造扁平token/query与持久batch metadata | [`ScheduleBatch`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/managers/schedule_batch.py) 准备extend/decode，再转换为执行用ForwardBatch | [`NativeBatchRuntime`](../mini_engine/engine/runtime.py) decode真正成批，prefill chunks仍逐请求执行；需要补执行批次接口后才能做真正ragged prefill |
| KV与Prefix索引 | [`KVCacheManager`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/core/kv_cache_manager.py#L264) 用block hash和coordinator管理命中、分配与cache groups | [`RadixCache.match_prefix`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/mem_cache/radix_cache.py#L364) 用树段与锁引用管理共享前缀；[`ReqToTokenPool`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/mem_cache/memory_pool.py#L375) 分离请求地址与KV存储 | [`BlockManager`](../mini_engine/cache/block_manager.py) 管页所有权；[`PrefixCache`](../mini_engine/cache/prefix_cache.py) 用完整token tuple、线性最长前缀扫描与LRU。完整命中已实际跳过，不等于hash APC或RadixAttention实现 |
| Graph策略 | [`CudagraphDispatcher`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/cudagraph_dispatcher.py#L233) 按FULL/PIECEWISE/NONE、token数和uniform decode等条件选择模式 | [`DecodeCudaGraphRunner`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L844) 管eligibility、load_batch与执行；[`FullCudaGraphBackend`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py#L99) 管capture/replay | [`DecodeGraphBucket`](../mini_engine/engine/cuda_graph.py) 只覆盖一token decode，按batch和块表宽度分桶；固定metadata和独立scratch，支持greedy capture；prefill不在图中 |
| 采样与后端契约 | [`Sampler`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/sample/sampler.py#L129) 配合SamplingMetadata、独立采样接口及拒绝采样模块 | [`ModelRunner.init_attention_backends`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/model_executor/model_runner.py#L1096) 与attention registry分离prefill/decode后端能力 | [`sampling.py`](../mini_engine/sampling.py) 的CUDA路径仅argmax；[`dandan_backend.py`](../mini_engine/kernels/dandan_backend.py) 显式校验dtype/设备/contract，量化另有bridge。后端可选不等于所有精度/布局/Graph组合均支持 |

路径随版本变化：该vLLM提交同时存在经典`worker/gpu_model_runner.py`与新`worker/gpu/model_runner.py`分支，新Graph描述还涉及query长度与microbatch，不能把两条runner串成同一调用链。该SGLang提交的Graph代码已拆到`model_executor/runner/`及`runner_backend/`，不继续引用旧版`model_executor/cuda_graph_runner.py`。[vLLM新Graph模块](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/worker/gpu/cudagraph_utils.py#L71)

一个值得解释的具体差异：在所读vLLM默认APC分支中，`get_computed_blocks()`将可命中长度限制在末token之前，以重新计算输出logits；我们给完整block边界缓存`next_logits`，在精确边界命中时可直接采样。它有额外logits存储成本，只适用于当前单模型/固定权重与对应缓存语义，不能据此宣称优于所有APC方案。[vLLM对应分支](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/core/kv_cache_manager.py#L264)

### 从源码对照得到的开发取舍

1. **先清晰执行接口**：Chat与batch可继续有不同入口，但逐步共享输入准备、模型执行、采样及清理边界，减少两套循环的行为漂移。把调度决策与GPU执行metadata分开，不为此引入多进程Executor。
2. **再做真正batched/ragged prefill**：调度输出明确每请求的query范围、绝对位置与KV页；配套attention和模型对拍。把多请求放进队列或把prompt切块，本身不算完成混合执行批次。
3. **补强后端契约**：明确dense/paged、dtype/head_dim、suffix offset、Graph eligibility和workspace约束。`dandan_ops`仍独立演进，引擎在版本升级后重新验收，Graph不得沿用旧权重/路由捕获。
4. **按证据决定索引与异步复杂度**：Prefix扫描成为CPU热点后再选block hash或radix；CPU/GPU overlap需要in-flight批次、取消、过期结果、buffer及KV最后使用时刻的协议，不能只把`.cpu()`改成non_blocking就算完成。
5. **继续控制范围**：保留单卡dense、Windows原生和可解释的资源所有权；多GPU、PD分离、主机层级缓存及多模态保持后置。现有continuous batching、Prefix和Graph已形成闭环，适用范围与工业框架仍有明确差异。

上述顺序是基于当前代码的工程判断，尚未实现新的执行器、Radix索引或overlap scheduler。性能验收继续做项目内部消融与MiniVLLM同口径对照，不要求安装vLLM/SGLang；源码阅读也不引入Linux、WSL或Triton运行依赖。
