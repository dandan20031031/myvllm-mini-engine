# 推理引擎架构与实现

更新至2026-10-26。范围为Windows 11单GPU、Qwen3-0.6B；HuggingFace负责权重加载、tokenizer和正确性参考，native forward、缓存、调度与CUDA扩展由项目实现。运行命令见[README](../README.md)，当前路线见[开发方案](../DEVELOPMENT_PLAN.md)，数值与性能证据见[验证报告](VALIDATION_AND_BENCHMARKS.md)。

## 调用链与代码入口

```text
启动: __main__.main / server.main -> cli -> RunnerConfig -> NativeQwen3Runner.from_pretrained

CLI / Chat -> NativeQwen3Runner.generate_incremental -> NativeQwen3 / DecodeGraphBucket

HTTP -> serving.http -> InferenceService.submit -> 单GPU worker._run
                                                       |
                                      NativeBatchRuntime.step -> Scheduler
                                               |                    |
                                          NativeQwen3         GenerationRequest
                                               |                    |
                                          KVCachePool <- BlockManager / PrefixCache

共同后端: CUDA kernels / cuBLAS / optional dandan_ops
```

两条生成路径共享模型、KV与算子实现，但CLI/Chat直接执行单请求循环，HTTP/batch经调度runtime执行；不能把示意图理解为所有入口已收敛到同一个执行循环。Python调用以`NativeQwen3Runner`和它创建的batch runtime为执行入口。

| 入口 | 职责 |
|---|---|
| `mini_engine/config.py`、`cli.py` | RunnerConfig静态约束、共享启动参数与转换；不执行模型加载或请求调度 |
| `mini_engine/__main__.py`、`server.py` | 应用参数、启动/退出；server保留原模块的公共导入兼容性 |
| `mini_engine/model/native_runner.py` | 加载、硬件/后端能力校验、单请求生成、创建batch会话、Chat Graph生命周期 |
| `mini_engine/model/outputs.py` | native与Transformers reference共用的生成结果值类型 |
| `mini_engine/model/native_qwen3.py` | Embedding、Q/K RMSNorm、RoPE、GQA、SwiGLU及prefill/decode/suffix forward |
| `mini_engine/cache/` | 物理页引用、逻辑块表、KV tensor和完整prefix索引 |
| `mini_engine/engine/request.py`、`scheduler.py`、`runtime.py` | 请求状态、token预算、decode优先执行、采样与回收 |
| `mini_engine/engine/batch.py` | PrefillBatch的packed query边界、绝对位置、页表/写槽与共享页校验 |
| `mini_engine/engine/cuda_graph.py` | 固定形状bucket、scratch页、capture/replay |
| `mini_engine/kernels/`、`quantization/` | Torch reference、CUDA与可选dandan后端 |
| `mini_engine/speculative.py` | 单请求greedy投机、提案与验证、KV逻辑回退 |
| `mini_engine/serving/protocol.py` | 聊天子集字段校验与APIError；不调用tokenizer/model |
| `mini_engine/serving/service.py` | 请求队列、单GPU worker、tokenizer、runtime、取消与退出 |
| `mini_engine/serving/http.py` | HTTP连接限流、路由、JSON/SSE编码与断线检测 |

### 启动与接口边界

启动参数由`add_runner_arguments()`统一注册，经`runner_config_from_args()`生成不可变`RunnerConfig`，再由`runner_kwargs()`转换为既有`from_pretrained()`参数。配置仅校验静态组合；CUDA设备、算子库版本及模型head_dim等能力检查继续在runner/backend执行。CLI与HTTP保留各自的生成上限、队列及超时参数，服务只暴露其支持的参数子集，不能把共享注册理解为两个入口支持全部相同模式。

`IncrementalGenerationResult`提取到`model/outputs.py`，native/reference共同引用；没有替换Transformers权重加载流程。服务拆分保持`python -m mini_engine.server`和`from mini_engine.server import APIError, ChatHTTPServer, InferenceService`可用。CLI仍直接运行单请求生成，HTTP worker仍使用batch runtime。

结构维护的原始验收保留在[摘要](data/structure_validation_summary.json)，最新状态集中到[验证报告](VALIDATION_AND_BENCHMARKS.md)，不在模块说明中重复累计测试数。Qwen3.5、混合注意力/多模态和独立权重加载器暂缓。

2026-10-26整理移除未进入真实Chat/HTTP路径的早期`LLMEngine`/`EngineConfig`门面，不保留同名空壳。原Python调用方改用`from mini_engine.model import NativeQwen3Runner`，由`from_pretrained()`加载，单请求使用`generate_incremental()`，调度执行使用`create_batch_runtime()`；按下文资源边界关闭会话。CLI和HTTP启动入口保持不变，底层`Scheduler`、cache与原生模型仍可直接用于测试/实验。

文档入口收敛为README、开发路线与主题报告，重复项目总览、阶段Markdown和一次性迁移快照退出活动目录。整理前的工作区原字节备份于忽略的`build/archive/structure_cleanup_20261026`；Git仍保留历史实现与实验来源。主题报告和支持既有结论的JSON保留原有证据范围，不因文件减少而重算或升级性能结论。

## KV所有权、Prefix与调度

KV布局为 `[layer, physical_block, token_in_block, kv_head, head_dim]`。`BlockManager`管理allocate/attach/grow/pin/unpin/truncate，`KVCachePool`保存tensor；请求引用与Prefix Cache引用独立。LRU淘汰只撤销缓存pin，活跃请求持有的页继续有效。

Prefix仅共享完整block。部分命中从 `position=prefix_len` 计算suffix；完全命中且有对应边界logits时跳过forward。非对齐prompt保存最后一个完整block末尾那一行logits；不能把整个prompt末尾logits挂在较短的prefix key上。admission重新查询缓存，避免等待期间淘汰导致旧页号失效；替换entry先pin新页再unpin旧页。完整共享页不写回。

每轮先执行decode，再用剩余 `max_num_batched_tokens` 预算推进prefill；`prefill_chunk_size`限制单请求chunk。decode超预算时轮转。调度器按请求最大KV需求做保守虚拟预留，物理页按需分配；永久超容量请求报错，暂时不足等待。最终输出token不再送回模型，KV需求为 `prompt_len + max_new_tokens - 1`。EOS、取消和执行/采样异常均释放请求引用。

模型接口 `model(suffix_ids, cache, block_ids, position=n)` 默认返回 `[suffix_length, vocab_size]`。普通单token且`position>0`继续使用既有paged decode；prefill/多token suffix共享packed模型路径。CUDA prefill直接读取paged历史KV，dense后端按请求gather已有前缀并计算绝对位置causal attention。CUDA/dandan保持原生GQA，Torch分支按需扩展KV heads。

CUDA后端已实现ragged batched prefill和直接paged多query attention。`auto/math/flash/dandan`仍逐请求执行dense attention，其投影和MLP合批。调度仍先decode、后prefill，不是一个统一的混合GPU执行批次；尚无安全preemption，最大预算预留会限制高输出上限场景的并发。

## Ragged Prefill与选择性LM head

同轮多个chunk不补齐到最大prompt长度，而是拼成`[total_query_tokens, hidden_size]`执行Embedding、投影、Norm和MLP。attention仍按请求分界：一个请求只能看到自己的历史KV，以及当前chunk中不晚于query绝对位置的key。不同prefix长度不改变这一隔离规则。

```text
Scheduler.prefill ranges -> NativeBatchRuntime.prefill_step
                                    |
                       PrefillBatch.from_sequences
                                    |
           NativeQwen3.forward_prefill_batch / B1 public forward
                                    |
           packed Embedding -> QKV/Norm/RoPE -> flat KV scatter
                                    |
              CUDA paged varlen attention / per-request dense attention
                                    |
                    packed O projection / Residual / MLP
                                    |
             selected query rows -> final Norm / LM head -> Prefix / sampling
```

`engine/batch.py`将CPU调度结果转为一次执行的只读metadata。构造前在host校验请求数量、token/页范围、容量、重复写槽和共享页写入；允许完整只读prefix页共享，禁止任一请求写本批其他请求引用的页。公开模型入口校验metadata与cache/model匹配，逐层CUDA调用使用可信metadata，避免逐层device-to-host检查。

| Metadata | 形状与含义 |
|---|---|
| input_ids / positions | `[T]`；真实query token及逐请求绝对位置，不使用flatten后的全局序号作RoPE位置 |
| cu_seqlens_q | `[B+1]`；每请求在packed Q中的起止，`T=sum(query_lengths)` |
| query_starts / context_lens | `[B]`；本轮前已有KV长度与本轮后总长度，后者等于start+query length |
| block_tables / slot_mapping | `[B,max_blocks]`与`[T]`；读历史的逻辑页表，以及新KV的物理写槽 |
| host tuples | query_lengths、query_offsets、starts、block_ids；用于结果映射与dense reference，不把Q边界误当历史KV边界 |

对请求r的新query i，位置为`starts[r]+i`，写槽为`block_tables[r, position//block_size]*block_size + position%block_size`。`kernels/data.py::write_kv_cache_slots`一次提交所有新K/V；`kernels/prefill.py::paged_prefill_attention`及`csrc/prefill.cu`按cu_seqlens_q定位请求，直接读其paged KV、应用各自的绝对causal上界，不gather整段历史。该CUDA路径支持FP16/BF16与GQA，仍是项目自有kernel，不宣称达到FlashAttention性能。

`--prefill-backend cuda`选择上述直接paged路径。`auto/math/flash/dandan`保持显式后端语义：每请求调用既有dense prefill API，只有投影/MLP等合批；有历史前缀时仍gather。`flash`继续是严格PyTorch Flash选择，`dandan`继续检查库的FP16/D128能力，不能静默改为本项目CUDA或宣称算子库已支持varlen Flash。

选择性输出通过`logits_rows`指定packed query行，先选hidden rows再执行final Norm和LM head。CLI普通生成只需prompt末行；runtime取完成prompt的末行与本轮最新完整Prefix边界行的并集，保留行号到请求的映射。空选择仍执行decoder与KV写入，但跳过final Norm/LM head；未完成prompt不采样，启用Prefix时中间chunk仍可能为边界缓存计算logits。精确Prefix命中没有新query，留在packed batch外直接使用已缓存logits。

`forward_prefill_batch(batch, cache, logits_rows=...)`通过Module调用进入公开forward，保留模型hook；B1保留原单请求hook/position语义。旧`forward(input_ids, cache, block_ids, position=...)`默认完整logits不变，reference和投机验证继续使用；不具备新batch接口的外部reference runner仍走兼容执行协议。packed阶段发生无法归因的异常时中止本轮相关chunks并释放请求引用；已共享的prefix页只读。

正确性、资源归还与三路径forward消融见[验证报告](VALIDATION_AND_BENCHMARKS.md)和[ragged摘要](data/ragged_prefill_summary.json)。消融分别固定B1完整logits、B1末行logits、packed末行logits；仅测预建metadata后的forward，不等同于TTFT、生成TPS或HTTP吞吐。Graph仍只覆盖decode，本轮未把ragged prefill捕获进图。FP16合批的严格逐元素KV阈值仍未通过，多请求时间仅作探索，不发布正式倍率；短输出一致不能证明batch invariance，选择性LM head也未观察到稳定延迟收益。

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
| Prefill/suffix | 新packed CUDA直接读paged KV、逐请求绝对causal与flat scatter；旧dense correctness helper保留8192 KV上限，新paged路径受模型/页表容量与head_dim约束；dandan仍为dense Flash |
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

`server.py`组装服务并保留兼容入口，路由实现在`serving/http.py`，提供 `/health`、`/v1/models`、`/v1/chat/completions`。`protocol.py`检查请求并构造APIError；`service.py`接收job和运行worker。主线程加载一次模型；单GPU worker负责tokenizer、runtime、调度、KV和模型调用。HTTP线程只校验和收发，经有界提交/结果队列交互，新请求可加入正在decode的runtime。

服务是greedy聊天子集：`temperature=0`，messages接受system/user/assistant非空字符串，未支持字段报400。SSE依次发送role/content、finish、可选 `choices=[]` usage及 `[DONE]`；usage统计实际生成token，包含文本隐藏的特殊token。服务默认仅监听127.0.0.1，无认证/TLS或多租户隔离；没有tools、多模态、随机采样或服务端会话记忆。

默认推理请求上限32、每请求结果队列32、HTTP连接线程64、body 1MiB、messages 128项、socket超时5秒、请求超时120秒、max_tokens上限512，均受配置及上下文/KV约束。Graph开启后还受bucket并发和上下文覆盖范围约束。

断线、慢消费者和超时触发worker取消并回收页；慢SSE不会阻塞GPU worker。输入错误只终止对应请求，无法归因的batch执行异常终止整个active batch；不可恢复GPU错误可能需重启。

资源按执行会话明确关闭：

| 所有者 | 创建与清理边界 |
|---|---|
| runner | 持有模型、BlockManager和KV tensor；单请求finally释放live请求页，runner.close同步并清理Chat Graph/scratch |
| batch runtime | 持有本会话scheduler/Prefix状态；close先同步设备，再abort waiting/running、清除Prefix pin、退休自有Graph bucket、释放自建scratch并恢复scheduler回调 |
| 外部Graph调用方 | 显式传入的Graph manager和padding页为借用；runtime不整体清空外部manager或释放外部页，仅退休绑定本会话自有scratch的bucket并恢复被暂替的旧bucket |
| service与入口 | worker的finally调用runtime.close；service.close发出停止并join worker，server入口关闭HTTP后再关闭service和runner；CLI入口finally关闭runner |

runtime.close可重复调用，关闭后不再接受step；它不销毁runner的模型权重和KV tensor。外部仍持有的退休bucket引用不能继续replay。共享manager/allocator不得由两个runtime并发使用，复用runner前须先关闭前一会话。service.close在worker超时未退出时报告错误，调用方不能继续释放它使用中的buffer；CUDA kernel执行完毕才可完成清理。启动中途失败也走入口finally，创建batch会话失败时释放已分配的自有scratch。

`benchmark_native_batch.py`、`profile_native_batch.py`与`validate_system_extensions.py`也采用相同关闭接口，不再操作Graph scratch的内部owner名称。benchmark重复间仍清空Prefix以固定工作量，最终close在计时之外；profiler先退出插桩上下文、恢复方法和hook，再关闭runtime。注入构造、分词、执行错误的回归验证了这些边界。

## 与Nano-vLLM、MiniVLLM的主要对照

两者同时作为**架构、源码与性能对照对象**。功能对照基于固定提交的实际执行分支；性能另按模型、硬件、dtype、KV预算、输入token、采样与计时边界验收，不能从README吞吐或同名功能推导速度结论。

| 对照版本 | 固定依据与本地差异 |
|---|---|
| Nano-vLLM upstream | [`bb823b3e`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8)，本地`nano-vllm`的`origin/main`快照；不是滚动latest声明 |
| Nano-vLLM本地 | `7074c1885c2e2e233822556cddde23044da40e82`，在上述版本上添加CPU测试、延迟导入和类型导入调整；GPU模型、attention和runner未改，不代表已适配Windows |
| MiniVLLM upstream | [`00db3b477`](https://github.com/Wenyueh/MinivLLM/commit/00db3b477f33be1de91400483ae7d210c4d61868)；源码依据`git show REV:path`，不使用该checkout中未提交的Windows/Graph改动冒充upstream |
| MiniVLLM本地Windows分支 | [`59466f60`](https://github.com/dandan20031031/MinivLLM/commit/59466f60ac82d388b2cfd8d9805a2090c33fd7a1)，增加gloo单进程回退、Windows/sm_120依赖与Graph修复；不是upstream原样执行。该checkout的README另有未提交编辑，不纳入源码结论 |
| 本项目 | 结构维护基线`c9376f6b`，叠加本轮packed/paged prefill与选择性LM head；以本报告中的接口与ragged验收摘要为准 |

### 按实际计算与资源路径比较

| 对照点 | Nano-vLLM固定版本 | MiniVLLM固定版本 | 本项目当前实现 |
|---|---|---|---|
| 模型执行与投影 | 自有Qwen3 forward，默认合并QKV与Gate/Up；[`Qwen3模型`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/models/qwen3.py#L42) | 自有Qwen3/Llama模型和合并投影；[`模型入口`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/engine/model_runner.py#L29) | 原生dense Qwen3；QKV、Gate/Up融合可选，保留未融合与reference对拍路径 |
| 权重加载 | 直接逐safetensors文件/参数加载，参数上的loader处理融合和TP；[`loader`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/utils/loader.py#L12) | 直接读safetensors，但先收集CPU权重字典，再映射/拼接；[`loader`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/utils/loader.py#L54) | HF加载浮点模型后通过meta/assign复用参数storage；尚无独立低峰值分片加载器 |
| Prefill与decode attention | Triton写KV；prefill调用外部FlashAttention varlen，decode调用其paged KV接口；[`Attention.forward`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/attention.py#L59) | 自有Triton varlen prefill与paged decode；[`Attention.forward`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/layers/attention.py#L493)；不是调用相同外部FlashAttention库 | Windows CUDA paged decode与flat KV写入；CUDA prefill已直接读paged历史，其他后端逐请求dense attention；不等同外部varlen Flash |
| Ragged与chunked prefill | flatten多个请求，准备cu_seqlens和绝对位置；支持chunked prefill，缓存命中时提供block tables；[`prepare_prefill`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L129) | 支持多请求flatten的varlen prefill；调度仍要求整条prompt满足token预算，所读版本无chunk进度协议；[`prepare_prefill`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/engine/model_runner.py#L265) | chunk进度接入PrefillBatch，投影/MLP合批；CUDA attention按请求边界直接paged执行，显式dense后端仍逐请求attention；完整Prefix命中留在batch外 |
| 调度与显存压力 | prefill优先，有prefill则该轮不decode；decode缺页时preempt并回队重算；[`Scheduler`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/scheduler.py#L30) | prefill优先、prefill/decode分轮，decode压力下抢占；另有永久超容量/无进度保护；[`Scheduler`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/engine/scheduler.py#L21) | decode优先，剩余预算推进prefill；按最坏输出增长做保守准入，淘汰Prefix pin；无安全preemption/recompute |
| Prefix真实语义 | chained xxhash、token复核与页引用；释放请求后可保留空闲页hash供后续复用；最后一block不直接当完整命中；[`BlockManager`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/block_manager.py#L35) | 有hash/refcount结构，但释放页时清token_ids，完成请求不能成为后续warm prefix；prefill只处理本轮新K/V，不能据metadata字段宣称完整paged prefix复用已验收；[`释放与约束`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/engine/block_manager.py#L59) | 独立PrefixCache、完整block tuple索引、LRU/refcount；缓存边界logits，精确命中可跳过forward；当前是线性最长前缀查找 |
| CUDA Graph边界 | 按batch分桶捕获transformer hidden-state forward；LM head、随机采样在图外；[`run_model/capture`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L195) | upstream与Windows固定提交均为hidden-state Graph，LM head/采样在图外；Windows提交修正bucket选取、padding slot和capture初始化；[`Windows修复`](https://github.com/dandan20031031/MinivLLM/blob/59466f60ac82d388b2cfd8d9805a2090c33fd7a1/src/myvllm/engine/model_runner.py#L354) | batch与block表宽度分桶；捕获完整decode logits并可包含greedy sampling，固定metadata与独立scratch；prefill、CPU调度仍在图外 |
| Prefill的LM head行数 | 每请求仅取最后hidden row再投影词表；[`ParallelLMHead`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/layers/embed_head.py#L56) | 同样选择cu_seqlens_q标识的末行；[`ParallelLMHead`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/layers/embedding_head.py#L70) | 生成已选择末行与必要Prefix边界行；空选择不执行LM head；默认完整logits继续供reference和投机验证 |
| 采样 | temperature softmax/exponential随机采样，所读SamplingParams明确拒绝greedy；[`采样约束`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/sampling_params.py#L4) | 同类compiled随机采样，所读版本也拒绝temperature=0；[`采样约束`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/sampling_parameters.py#L4) | greedy与Torch temperature/top-k/top-p；CUDA采样仅greedy。公平比较需明确采样适配，不能混用不同模式再比较输出一致性 |
| 量化与服务范围 | 所读源码为浮点离线generate、TP与共享内存worker；未见INT4/FP8量化执行或OpenAI/SSE服务路径 | 所读源码为浮点离线引擎、Qwen3/Llama与TP；未见对应量化执行或OpenAI/SSE服务路径 | dandan W4A16实际计算、其他低比特reference；HTTP/SSE有界队列、取消/超时；单GPU，另有单请求greedy prompt lookup投机 |
| Windows与依赖 | spawn启动，但TP1也无条件NCCL；强制Triton和flash-attn；[`runner`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/nanovllm/engine/model_runner.py#L26)、[`依赖`](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/pyproject.toml#L14) | upstream同样无条件NCCL，依赖含vllm；Windows分支增加gloo单进程回退和triton-windows/cu128配置，未移除Triton；[`upstream`](https://github.com/Wenyueh/MinivLLM/blob/00db3b477f33be1de91400483ae7d210c4d61868/src/myvllm/engine/model_runner.py#L26)、[`Windows依赖`](https://github.com/dandan20031031/MinivLLM/blob/59466f60ac82d388b2cfd8d9805a2090c33fd7a1/pyproject.toml#L7) | Windows原生MSVC/Ninja CUDA，不依赖NCCL、Linux、WSL或Triton；dandan_ops是可选独立库 |

上述对照不把“源码中存在分支”等同“已完成本机性能验收”。Nano-vLLM本地CPU测试提交不解决NCCL/attention依赖；MiniVLLM的upstream、Windows提交、其他Graph/benchmark工作树也不能合并成一个未注明版本的结果。性能测试记录被测commit、dirty diff、平台适配、后端与图内范围；对照依赖单独隔离，不作为本引擎新增运行依赖。

### 源码对照后的落地与继续优化

1. **Prefill选择性词表投影已落地**：保留默认完整logits接口，生成选择末行和必要Prefix边界行。继续分别计量行选择与合批收益，评估不同上下文、batch与词表投影尺寸；不从减少行数直接推导端到端倍率。
2. **CUDA ragged/paged prefill已落地**：使用flatten queries、cu_seqlens、绝对位置和KV slots，直接读取历史页。其他后端仍逐请求dense attention；后续按profiling评估CUDA tile/融合、算子库varlen支持与更广形状，不把所有后端都标为相同覆盖。
3. **控制比较变量**：两者均有投影融合、不同Graph范围和随机采样；先统一模型/dtype、token工作量、缓存冷暖与可解释的采样模式。性能结果放在验证报告；这里的架构取舍不替代benchmark。

## 与vLLM、SGLang的框架及源码对照

2026-10-25按项目范围调整：两者只作结构与代码设计参考，主要架构及性能对照对象为Nano-vLLM与MiniVLLM。以下是源码阅读结论，不是安装、移植、运行或速度验证；不比较代码行数，也不将同名功能视为同等覆盖。

为避免main漂移，固定所读版本：vLLM [`5f63ea46`](https://github.com/vllm-project/vllm/commit/5f63ea461ccd567ece6f50dd3701430c0f321039)，SGLang [`773cb938`](https://github.com/sgl-project/sglang/commit/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed)。本地依据为阶段29提交`66dc76d`；后续源码变更需重新核对路径与配置分支。

### 先看职责分工

vLLM V1把请求处理、EngineCore调度、Executor/Worker和ModelRunner分层；Executor也有单进程实现，不应理解为只能多卡。SGLang Python SRT围绕TokenizerManager、Scheduler、ScheduleBatch/ForwardBatch、ModelRunner组织工作，normal/overlap是不同执行模式。进程与通信设计服务于更大的模型、设备和部署范围；我们的单GPU worker保持同进程，先借鉴接口边界。[vLLM架构说明](https://docs.vllm.ai/en/latest/design/arch_overview/)、[SGLang服务入口](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/entrypoints/engine.py#L892)

| 对照点 | vLLM所读实现 | SGLang所读实现 | 本项目当前实现与取舍 |
|---|---|---|---|
| 请求执行 | [`EngineCore.step/step_with_batch_queue`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/engine/core.py#L657) 经Executor执行，支持非阻塞调用与在途批次 | [`Scheduler.event_loop_normal/overlap`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/managers/scheduler.py#L1995) 管接收、选批、执行与结果处理 | [`InferenceService._run`](../mini_engine/serving/service.py) 同步推进runtime；HTTP线程与GPU worker已经分开，但没有CPU调度与GPU批次执行的完整overlap |
| 调度模型 | [`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/core/sched/scheduler.py#L677) 以已计算token与目标token进度统一分配工作，并处理抢占/恢复 | [`PrefillAdder`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/managers/schedule_policy.py#L584) 管输入/chunk/KV等预算，ScheduleBatch支持内存检查与retract | [`Scheduler.schedule`](../mini_engine/engine/scheduler.py) 显式区分decode与prefill，虚拟预留最坏输出增长；设计简单、准入保守，无安全preemption/recompute |
| 执行批次 | [`GPUModelRunner._prepare_inputs`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/worker/gpu_model_runner.py#L1924) 构造扁平token/query与持久batch metadata | [`ScheduleBatch`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/managers/schedule_batch.py) 准备extend/decode，再转换为执行用ForwardBatch | [`PrefillBatch`](../mini_engine/engine/batch.py)打包query chunks、绝对位置及页表，CUDA直接paged prefill；decode与prefill仍分开，dense attention后端仍逐请求执行 |
| KV与Prefix索引 | [`KVCacheManager`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/core/kv_cache_manager.py#L264) 用block hash和coordinator管理命中、分配与cache groups | [`RadixCache.match_prefix`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/mem_cache/radix_cache.py#L364) 用树段与锁引用管理共享前缀；[`ReqToTokenPool`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/mem_cache/memory_pool.py#L375) 分离请求地址与KV存储 | [`BlockManager`](../mini_engine/cache/block_manager.py) 管页所有权；[`PrefixCache`](../mini_engine/cache/prefix_cache.py) 用完整token tuple、线性最长前缀扫描与LRU。完整命中已实际跳过，不等于hash APC或RadixAttention实现 |
| Graph策略 | [`CudagraphDispatcher`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/cudagraph_dispatcher.py#L233) 按FULL/PIECEWISE/NONE、token数和uniform decode等条件选择模式 | [`DecodeCudaGraphRunner`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L844) 管eligibility、load_batch与执行；[`FullCudaGraphBackend`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py#L99) 管capture/replay | [`DecodeGraphBucket`](../mini_engine/engine/cuda_graph.py) 只覆盖一token decode，按batch和块表宽度分桶；固定metadata和独立scratch，支持greedy capture；prefill不在图中 |
| 采样与后端契约 | [`Sampler`](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/sample/sampler.py#L129) 配合SamplingMetadata、独立采样接口及拒绝采样模块 | [`ModelRunner.init_attention_backends`](https://github.com/sgl-project/sglang/blob/773cb938fe9484a8f7e7e46fb1b1fe96412c59ed/python/sglang/srt/model_executor/model_runner.py#L1096) 与attention registry分离prefill/decode后端能力 | [`sampling.py`](../mini_engine/sampling.py) 的CUDA路径仅argmax；[`dandan_backend.py`](../mini_engine/kernels/dandan_backend.py) 显式校验dtype/设备/contract，量化另有bridge。后端可选不等于所有精度/布局/Graph组合均支持 |

路径随版本变化：该vLLM提交同时存在经典`worker/gpu_model_runner.py`与新`worker/gpu/model_runner.py`分支，新Graph描述还涉及query长度与microbatch，不能把两条runner串成同一调用链。该SGLang提交的Graph代码已拆到`model_executor/runner/`及`runner_backend/`，不继续引用旧版`model_executor/cuda_graph_runner.py`。[vLLM新Graph模块](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/worker/gpu/cudagraph_utils.py#L71)

一个值得解释的具体差异：在所读vLLM默认APC分支中，`get_computed_blocks()`将可命中长度限制在末token之前，以重新计算输出logits；我们给完整block边界缓存`next_logits`，在精确边界命中时可直接采样。它有额外logits存储成本，只适用于当前单模型/固定权重与对应缓存语义，不能据此宣称优于所有APC方案。[vLLM对应分支](https://github.com/vllm-project/vllm/blob/5f63ea461ccd567ece6f50dd3701430c0f321039/vllm/v1/core/kv_cache_manager.py#L264)

### 当前改进重点

1. **先补数值边界**：逐层定位FP16合批的KV误差，扩展teacher-forced和较长生成，明确预设阈值与任务质量；在此之前不发布多请求正式倍率。
2. **再按热点优化prefill**：区分attention、投影、metadata和词表投影成本。现有CUDA已经直接paged；dense后端仍逐请求attention，不能从合批接口推导所有后端已覆盖varlen。分别做行选择、合批和kernel消融，再验收完整生成。
3. **减少执行重复**：Chat与batch保留必要入口，逐步共享输入准备、采样与异常协议，不再新增教学门面或多进程Executor。dandan升级后重新验收dtype/布局/Graph，不沿用旧权重或路由的capture。
4. **受控比较后再扩范围**：补Nano-vLLM/MiniVLLM共同离线计量；索引、overlap与preemption先有profile证据及取消/资源协议，再决定实现。多GPU、PD、低精度KV和训练型draft保持后置，路线详见[开发方案](../DEVELOPMENT_PLAN.md)。

现有continuous batching、Prefix和Graph已形成单卡闭环，但尚无Radix索引或overlap scheduler。工业源码只作设计参考，不要求安装vLLM/SGLang，也不引入Linux、WSL或Triton运行依赖。
