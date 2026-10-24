# 推理引擎架构与实现

截至2026-10-24，整合原阶段00–29的实现说明。范围为Windows 11单GPU、Qwen3-0.6B；HuggingFace负责权重加载、tokenizer和正确性参考，native forward、缓存、调度与CUDA扩展由项目实现。运行命令见 [项目 README](../README.md)，证据见 [系统验证与性能评测](VALIDATION_AND_BENCHMARKS.md)。

## 调用链与代码入口

```text
CLI / Chat / HTTP / benchmark
             |
NativeQwen3Runner / HTTP单GPU worker
             |
NativeBatchRuntime -> Scheduler -> GenerationRequest
       |                 |
  PrefixCache       BlockManager
       |                 |
NativeQwen3 -------- KVCachePool
       |
CUDA kernels / cuBLAS / optional dandan_ops
```

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
