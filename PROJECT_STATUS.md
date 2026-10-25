# MiniQwenEngine 项目总览

更新：2026-10-25。面向实习展示的 Windows 11 单卡 Qwen3-0.6B 推理引擎；Transformer权重加载与tokenizer使用HuggingFace，native forward、KV管理、调度与CUDA算子由项目实现。

## 已实现能力

| 模块 | 当前实现 |
|---|---|
| 原生模型 | Qwen3 embedding、Q/K RMSNorm、RoPE、GQA、SwiGLU；完整prefill、单token/batch decode、多token suffix |
| KV管理 | 分页tensor pool、跨页写入、引用计数、LRU、逻辑长度回退与物理尾页释放 |
| Prefix | 完整block共享；命中跳过计算、边界logits复用，避免覆盖共享页 |
| 调度 | continuous batching，decode优先、有token预算的chunked prefill，保守最大KV预留，取消与异常回收 |
| Ragged prefill | 多请求query chunks打包，投影/Norm/MLP合批；自有CUDA直接读取paged历史KV，dense后端明确逐请求attention |
| 选择性logits | CLI取prompt末行；runtime取完成prompt末行和必要Prefix边界行并集，默认forward仍保留全部query logits |
| CUDA | FP16/BF16 PagedAttention（GQA/split-KV/D128向量化）、Norm/Residual/SwiGLU、Embedding/RoPE/KV scatter、causal suffix prefill、可选greedy argmax |
| Graph与融合 | batch与CLI Chat decode Graph、图内greedy、固定pinned/GPU metadata、独立scratch；QKV与Gate/Up融合 |
| 精度 | FP32/FP16/BF16；INT8/INT4/FP8 weight-only reference；可选dandan FP16 GEMM/Flash和W4A16 CUDA计算 |
| 投机 | 无额外模型的greedy prompt lookup、并行验证、接受/拒绝、KV回退、接受率自适应draft预算 |
| 接口 | CLI、多轮Chat、Windows HTTP chat completions子集/SSE；单GPU worker调度，有界输入/输出队列和断线取消 |
| 模块边界 | CLI/HTTP共享RunnerConfig与启动参数；服务协议、worker、HTTP传输分层；native/reference共享生成结果类型 |
| 实验 | kernel配对sweep、真实batch eager/Graph、投机对拍、完整模型系统验收、兼容服务HTTP负载与比较条件检查 |
| Profiling | PyTorch CPU/CUDA trace、NVTX阶段范围、eager/Graph对照；Nsight命令与当前采集限制见阶段28报告 |

## 代码调用链

```text
CLI/Chat: __main__ -> cli/RunnerConfig -> NativeQwen3Runner
                                             |
                            generate_incremental -> NativeQwen3 / Chat Graph

HTTP: server -> cli/RunnerConfig -> serving.http -> InferenceService worker
                                                        |
                                   NativeBatchRuntime -> Scheduler / Request
                                            |                  |
                                       NativeQwen3       BlockManager / PrefixCache

共享执行基础: NativeQwen3 -> KVCachePool + CUDA / cuBLAS / optional dandan_ops
```

- `mini_engine/config.py`、`cli.py`：统一启动参数与静态组合校验；模型/硬件能力仍由runner检查，请求和服务限额留在各入口。
- `mini_engine/model/native_runner.py`：加载权重、单请求生成、Chat Graph、创建batch会话；CLI直接走该生成循环，不经过batch scheduler。
- `mini_engine/model/outputs.py`：native与Transformers reference共享`IncrementalGenerationResult`，结果字段不再依赖reference runner定义。
- `mini_engine/engine/runtime.py`：decode优先，再合并未缓存prompt chunks；选择完成请求/Prefix边界所需logits，完成prompt后才采样。
- `mini_engine/engine/batch.py`：`PrefillBatch`保存packed query边界、绝对位置、分页表和flat KV写入槽，校验共享只读页与写入隔离。
- `mini_engine/cache/`：block的物理所有权、完整prefix边界与KV tensor存储分离。
- `mini_engine/model/native_qwen3.py`：共享packed prefill与旧单请求接口；CUDA直接paged多query attention，其他后端逐请求dense attention；decode Graph路径保留。
- `mini_engine/speculative.py`：将pending token与草稿一次送入目标，提交连续接受前缀，拒绝部分截断KV逻辑长度并归还尾页。
- `mini_engine/serving/protocol.py`、`service.py`、`http.py`：分别负责请求协议/错误、单GPU worker/队列生命周期、HTTP连接/JSON/SSE；tokenizer、runtime、KV变更由worker处理。`mini_engine/server.py`保留启动入口和兼容导出。
- `mini_engine/kernels/cuda_extension.py`：内容hash隔离Windows DLL构建；旧会话可继续加载旧DLL，新版本编译到不同路径。launch错误检查已恢复，Graph中只跳过会同步到CPU的tensor metadata检查。

技术报告按主题整理，入口见 [技术报告目录](reports/README.md)：[推理引擎架构与实现](reports/ARCHITECTURE.md)、[系统验证与性能评测](reports/VALIDATION_AND_BENCHMARKS.md)、[算子后端性能复核](reports/BACKEND_PERFORMANCE_RECHECK.md)和[推理引擎性能分析指南](reports/PROFILING.md)；原始JSON实验摘要保留。

2026-10-25主对照调整为Nano-vLLM与MiniVLLM，同时比较架构和性能；已固定源码并核对ragged prefill、抢占、prefix、Graph边界、加载、采样和Windows适配。vLLM/SGLang源码对照保留为工业参考。当前没有三方同条件实测排名，详情并入架构与性能主题报告。

## 2026-10-25：结构维护与资源所有权

本轮已整理现有Qwen3路径：共享启动配置、拆分服务职责、提取公共输出类型，并集中batch会话的清理。CLI和HTTP两套生成循环继续保留；全量403项回归与真实BF16模型系统验收通过，本轮没有新增性能结果。

`NativeBatchRuntime.close()`在所属worker停止执行后同步CUDA，取消waiting/running请求、清除Prefix引用，释放本会话创建的Graph和scratch，并恢复scheduler回调。借用的Graph manager只移除绑定本会话scratch的bucket；外部提供的padding页不由runtime释放。runner继续持有模型权重与KV tensor，关闭runtime后才可复用其allocator。共享manager/allocator不支持两个runtime并发使用。

HTTP worker的`finally`调用runtime.close；服务入口先关闭HTTP连接、等待worker结束，再调用runner.close。CLI在`finally`关闭Chat Graph与scratch。worker未退出时，service.close会报错，避免提前释放仍被GPU使用的资源。Qwen3.5、混合注意力/多模态适配与独立权重加载器暂缓，继续保留Transformers加载和正确性参考。

batch benchmark、profiling和系统验证同样通过`finally`/执行会话调用`runtime.close()`；关闭不进入性能计时，profiling先恢复临时插桩再释放资源。wheel构建及新模块/CUDA源码打包检查通过。证据见[结构维护验收摘要](reports/structure_validation_summary.json)。

## 验证与性能证据

设备：RTX5080 SM120，PyTorch2.9.1+cu128。项目CUDA使用Toolkit13.3；独立算子库此前用12.8构建。两者均须通过目标机回归，不能因为版本提示而忽略运行测试。

- 结构维护基线403 passed、2 warnings；CUDA/Graph/HTTP实际执行，覆盖启动配置、兼容导入、创建失败回滚、关闭幂等、借用资源保护、Graph退休后页重用与插桩恢复。JUnit保存在`build/validation/structure_pytest.xml`；不作为后续ragged实现的测试计数。
- Ragged实现最终全量467 passed、2 warnings、无跳过；完整BF16 CUDA、FP16 dandan与W4A16重新通过chunk/Prefix/batch2/4 Graph验收，BF16另过HTTP/SSE。B1选择性LM head明显减少额外分配峰值，未观察到稳定延迟收益；FP16 B2的严格逐元素KV阈值仍未通过，多请求时间仅作探索，不发布正式倍率。见[本轮摘要](reports/ragged_prefill_summary.json)。
- 本轮重新通过完整Qwen3-0.6B BF16分块/Prefix、batch2/4 eager/Graph与并发HTTP/SSE验收；所有会话关闭后页引用归零。原始记录为`build/validation/structure_system_validation.json`，不作为性能测量。
- 完整Qwen3 BF16：64-token冷/热prefix的计算64→0，5-token后缀仅算5；共享页保持不变，完整/分块输出一致。
- batch2/4 eager/Graph输出一致，capture真实执行；真实并发HTTP/SSE正常返回usage/DONE，关闭后KV引用全部为0。
- FP16 dandan GEMM/Flash与W4A16也通过完整模型chunk/prefix/Graph验收；这比较相同精度的执行路径，不代表INT4与BF16质量相同。
- 新D128向量化PagedAttention的15组FP16配对测量观察到1.74–1.80x，基线为项目原CUDA kernel，计时为Graph内单算子均摊；非服务性能。
- BF16重复句子投机实测：32输出token，target调用32→16，537.39→300.87ms（观察1.79x）；普通KV问答调用仍32次，几乎无收益。短序列输出一致，不代表所有浮点配置batch invariant。
- 历史3请求BF16 batch：eager189.3、Graph473.9、fusion+Graph519.2 tokens/s。只作历史参考，不能与本轮不同精度/负载/计时直接拼比。
- 单请求FP16 Chat Graph：两prompt×64/128输出，各6对交替测量，TPOT P50从15.58–16.94降为2.53–2.66ms，总TPS观察5.65–6.05x；TTFT未一致改善。输出与源码/后端指纹对齐，测量期无capture；非HTTP或行业排名，详见主题验收报告。

原始结果在 `benchmarks/results/`，不提交模型或机器相关大文件；报告保存紧凑摘要。GPU未严格独占的数据不作为工业引擎排名证据。

2026-10-24新增性能复核发现同一cuBLAS基线首末轮404→725 TPS，尚不足以给出dandan FP16端到端加速比。W4A16本轮Graph分配显存峰值1054.64MiB，对应FP16为1614.39MiB；具体条件和输出差异见[算子后端性能复核](reports/BACKEND_PERFORMANCE_RECHECK.md)。真实FP16 eager/Graph CPU+CUDA trace已采集，另有2项profiling恢复/异常测试通过，见[推理引擎性能分析指南](reports/PROFILING.md)。

## 仍需完成

选择性LM head、packed projection/MLP与CUDA直接paged prefill已接入；其数值、资源及forward消融见[当前验证报告](reports/VALIDATION_AND_BENCHMARKS.md)与[ragged摘要](reports/ragged_prefill_summary.json)。精确Prefix命中不进入packed batch；未完成prompt不采样，无末行/Prefix边界需求时不调用LM head。Graph仍仅decode，full-logits接口继续供reference/投机验证使用。

接下来补两个主对照的共同离线adapter与受控端到端计量。Nano原版在当前Windows环境不可直接运行，Mini需固定本地修复分支及实际导入路径，不把历史数据当同条件实测。

1. **深化ragged后端与性能验收**：先扩展FP16形状相关误差与较长生成/质量验收，明确KV误差预算，保留严格失败记录；再做受控端到端测量。CUDA已直接paged多query；`auto/math/flash/dandan`的attention仍按请求dense执行，仅投影/MLP合批。按profiling决定attention分块/融合或算子库varlen支持；prefill与decode尚未统一为同一混合执行批次。
2. **FP8/INT8 KV与FP8 GEMM**：定义scale、直接分页读取和模型误差；已有W4A16不能代表这些格式完成。
3. **NVFP4/MXFP8**：尚无项目可用计算后端，保留reserved，不能用INT4冒充FP4。
4. **更完整投机**：小draft、随机接受/校正分布、batch/Graph调度、EAGLE-3/DFlash训练型draft；现有接受率策略不是DeepInfer置信度实现。
5. **服务生产能力**：目前greedy聊天子集，无tools/多模态/安全网关；服务执行错误可中止整个active batch。异步调度、preemption、随机CUDA采样尚待完成。
6. **Nano-vLLM/MiniVLLM性能与质量集**：补同机、同精度、同负载的离线对照；目前Mini历史实际FP32批量、Nano官方4070 Laptop与本项目FP16单请求不能排名。首轮共同temperature=0.6、固定输出长度，greedy适配另行披露；HTTP服务单独测。vLLM/SGLang和DeepInfer只作结构/机制参考，不纳入速度排名。
7. **工具验证**：新kernel尚未做compute-sanitizer memcheck/racecheck；数值/Graph测试不能替代。
