# MiniQwenEngine 项目总览

更新：2026-10-24。面向实习展示的 Windows 11 单卡 Qwen3-0.6B 推理引擎；Transformer权重加载与tokenizer使用HuggingFace，native forward、KV管理、调度与CUDA算子由项目实现。

## 已实现能力

| 模块 | 当前实现 |
|---|---|
| 原生模型 | Qwen3 embedding、Q/K RMSNorm、RoPE、GQA、SwiGLU；完整prefill、单token/batch decode、多token suffix |
| KV管理 | 分页tensor pool、跨页写入、引用计数、LRU、逻辑长度回退与物理尾页释放 |
| Prefix | 完整block共享；命中跳过计算、边界logits复用，避免覆盖共享页 |
| 调度 | continuous batching，decode优先、有token预算的chunked prefill，保守最大KV预留，取消与异常回收 |
| CUDA | FP16/BF16 PagedAttention（GQA/split-KV/D128向量化）、Norm/Residual/SwiGLU、Embedding/RoPE/KV scatter、causal suffix prefill、可选greedy argmax |
| Graph与融合 | 真实batch decode Graph bucket、固定buffer、独立padding页；QKV与Gate/Up融合 |
| 精度 | FP32/FP16/BF16；INT8/INT4/FP8 weight-only reference；可选dandan FP16 GEMM/Flash和W4A16 CUDA计算 |
| 投机 | 无额外模型的greedy prompt lookup、并行验证、接受/拒绝、KV回退、接受率自适应draft预算 |
| 接口 | CLI、多轮Chat、Windows HTTP chat completions子集/SSE；单GPU worker调度，有界输入/输出队列和断线取消 |
| 实验 | kernel配对sweep、真实batch eager/Graph、投机对拍、完整模型系统验收、兼容服务HTTP负载与比较条件检查 |

## 代码调用链

```text
CLI / HTTP线程 / benchmark
        |
NativeQwen3Runner / 服务单GPU worker
        |
NativeBatchRuntime -> Scheduler -> GenerationRequest
        |                  |
  PrefixCache          BlockManager
        |                  |
NativeQwen3 ---------- KVCachePool
        |
CUDA kernels / cuBLAS / optional dandan_ops
```

- `mini_engine/engine/runtime.py`：decode优先，再执行未缓存prompt块；模型结果交给请求采样与状态更新。
- `mini_engine/cache/`：block的物理所有权、完整prefix边界与KV tensor存储分离。
- `mini_engine/model/native_qwen3.py`：单token直接paged decode；多token suffix按绝对位置causal验证。
- `mini_engine/speculative.py`：将pending token与草稿一次送入目标，提交连续接受前缀，拒绝部分截断KV逻辑长度并归还尾页。
- `mini_engine/server.py`：HTTP线程仅验证/收发；tokenizer、runtime、KV变更全部由单worker处理。
- `mini_engine/kernels/cuda_extension.py`：内容hash隔离Windows DLL构建；旧会话可继续加载旧DLL，新版本编译到不同路径。launch错误检查已恢复，Graph中只跳过会同步到CPU的tensor metadata检查。

各阶段逐文件解读保存在 `reports/STAGE_23_*.md` 至 `reports/STAGE_27_*.md`，历史阶段报告保留当时的实现状态。

## 验证与性能证据

设备：RTX5080 SM120，PyTorch2.9.1+cu128。项目CUDA使用Toolkit13.3；独立算子库此前用12.8构建。两者均须通过目标机回归，不能因为版本提示而忽略运行测试。

- 最终全量325 passed、2 warnings；CUDA、Graph及本机HTTP loopback实际执行。
- 完整Qwen3 BF16：64-token冷/热prefix的计算64→0，5-token后缀仅算5；共享页保持不变，完整/分块输出一致。
- batch2/4 eager/Graph输出一致，capture真实执行；真实并发HTTP/SSE正常返回usage/DONE，关闭后KV引用全部为0。
- FP16 dandan GEMM/Flash与W4A16也通过完整模型chunk/prefix/Graph验收；这比较相同精度的执行路径，不代表INT4与BF16质量相同。
- 新D128向量化PagedAttention的15组FP16配对测量观察到1.74–1.80x，基线为项目原CUDA kernel，计时为Graph内单算子均摊；非服务性能。
- BF16重复句子投机实测：32输出token，target调用32→16，537.39→300.87ms（观察1.79x）；普通KV问答调用仍32次，几乎无收益。短序列输出一致，不代表所有浮点配置batch invariant。
- 历史3请求BF16 batch：eager189.3、Graph473.9、fusion+Graph519.2 tokens/s。只作历史参考，不能与本轮不同精度/负载/计时直接拼比。

原始结果在 `benchmarks/results/`，不提交模型或机器相关大文件；报告保存紧凑摘要。GPU未严格独占的数据不作为工业引擎排名证据。

## 仍需完成

1. **真正ragged batched prefill、直接paged多query验证**：当前chunk仍逐请求执行，suffix需要gather KV。
2. **FP8/INT8 KV与FP8 GEMM**：定义scale、直接分页读取和模型误差；已有W4A16不能代表这些格式完成。
3. **NVFP4/MXFP8**：尚无项目可用计算后端，保留reserved，不能用INT4冒充FP4。
4. **更完整投机**：小draft、随机接受/校正分布、batch/Graph调度、EAGLE-3/DFlash训练型draft；现有接受率策略不是DeepInfer置信度实现。
5. **服务生产能力**：目前greedy聊天子集，无tools/多模态/安全网关；服务执行错误可中止整个active batch。异步调度、preemption、随机CUDA采样尚待完成。
6. **行业公平跑分与质量集**：HTTP工具已经可用，尚无本项目与vLLM/SGLang同条件实测；DeepInfer目标模型不同，不纳入Qwen3速度排名。
7. **工具验证**：新kernel尚未做compute-sanitizer memcheck/racecheck；数值/Graph测试不能替代。

## 实习展示方式

展示完整调用链、独立CUDA与参考对拍、缓存跳过计数、投机拒绝回退、真实HTTP请求，以及可重复性能实验。简历可以写已经实现并验收的工程能力与具体实验配置；不要写“全面超过vLLM”、将全部研究计划写成完成，或把论文速度当作项目速度。

学术与行业来源、可行性和验收门槛统一保存在 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)。启动及benchmark命令见 [README.md](README.md)。
