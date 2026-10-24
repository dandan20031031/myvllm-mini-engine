# 系统验证与性能评测

汇总截至2026-10-24已记录的验收与实验。本次文档整理未重新跑模型、测试或benchmark。实现约束见 [推理引擎架构与实现](ARCHITECTURE.md)，全部数据索引见 [技术报告目录](README.md)。

## 已执行的正确性验收

设备为Windows 11、RTX5080 SM120、PyTorch2.9.1+cu128。引擎自有扩展记录使用Toolkit13.3，独立dandan库此前使用12.8构建，不能混写为同一构建环境。

原系统验收全量 **325 passed、2 warnings**，CUDA/Graph与真实HTTP loopback实际执行，日志为 `build/internship_pytest_final.log`。warning为Toolkit版本差异提示和setuptools私有MSVC接口。后续profiling工具另有2项恢复/异常测试通过；未将两次执行拼成新的全量测试数。

| 完整Qwen3-0.6B配置 | chunk/full、cold/warm/partial prefix、batch2/4 eager/Graph | 真实并发HTTP/SSE |
|---|---|---|
| BF16、cuBLAS + 自有CUDA prefill | 通过 | 通过 |
| FP16、dandan_cuda + dandan Flash | 通过 | 本组未执行 |
| W4A16、dandan INT4投影 + dandan Flash | 通过 | 本组未执行 |

三组均使用64-token prompt、8-token输出的短序列验收：完整prefix命中后实际计算 **64→0**，5-token suffix只计算5，完整共享页不变；Graph确实capture batch2/4，与各自eager输出对拍；清理后页引用归零。量化验收比较相同量化模型的执行路径，不代表W4与BF16质量相同。数值、配置与指纹见 [stage27_system_summary.json](stage27_system_summary.json)。

另有FP16/W4A16短序列kernel验收：2条prompt、prefill后4次decode，logits阈值 `atol=0.125, rtol=0.01`；FP16对cuBLAS，W4对同packed state的FP16-rounded权重与FP32累加oracle，teacher-forced argmax及独立greedy匹配。原FP16到INT4的最大logit变化约9.3–9.8，部分输出改变，不能宣称质量无损。见 [stage22_backend_validation_summary.json](stage22_backend_validation_summary.json)。

回归覆盖页边界/尾组、非法metadata、非默认stream、FP16/BF16、Graph replay、prefix边界logits、LRU/live引用、有限KV容量、投机拒绝回退/EOS、服务取消/慢消费者/异常释放。新的D128 kernel尚未完成compute-sanitizer memcheck/racecheck，数值与Graph对拍不能代替它们。

## 保留的性能结果

### D128 PagedAttention配对实验

FP16、Hq16/Hkv8/D128，block16，每Graph20次调用、20组样本，同输入交替测原kernel与向量化kernel；单位为Graph内每次算子的均摊中位数。

| Batch/context | 原kernel | 新kernel | 配对比值 |
|---|---:|---:|---:|
| 1/128 | 33.99 us | 19.34 us | 1.76x |
| 8/512 | 72.38 us | 41.50 us | 1.74x |
| 32/2048 | 713.76 us | 400.30 us | 1.78x |

block16/32/64/128/256共15组均通过reference对拍，比值范围1.74–1.80x，最大绝对误差≤0.0002442。外部GPU负载未受控，结果仅支持本机单算子观察，不能推导服务加速或工业引擎排名。配置、误差与指纹见 [stage26_paged_summary.json](stage26_paged_summary.json)。

### 单请求greedy投机

BF16、项目CUDA、projection fusion，固定32实际输出token（ignore EOS），warmup1、重复3次；普通与投机逐token输出一致。

| 场景 | 普通总耗时中位数 | 投机总耗时中位数 | target调用 | 接受率 |
|---|---:|---:|---:|---:|
| 重复指定英文句子 | 537.39 ms | 300.87 ms | 32→16 | 80% |
| 用一句话解释KV Cache | 422.81 ms | 415.17 ms | 32→32 | 0% |

观察比值1.79x/1.02x。计时含prefill、草稿查找、验证、接受/回退，排除加载、tokenization及最终文本decode。两条路径均为单请求eager，未证明独占GPU；不代表Graph/HTTP收益、长上下文或所有浮点配置输出不变。原始样本为 `benchmarks/results/stage24_speculative_qwen3.json`，紧凑结果在 [系统摘要](stage27_system_summary.json)。

### 历史batch Graph与elementwise

BF16、3请求、32-token输出预算、block16，计时含逐请求prefill、decode及scheduler生命周期；实际输出数用于TPS，并非每请求固定输出32。

| 模式 | median total | 聚合tokens/s |
|---|---:|---:|
| eager | 438.38 ms | 189.33 |
| Graph | 175.15 ms | 473.87 |
| fusion + Graph | 159.88 ms | 519.15 |

这是历史runtime基线，原始文件为 `benchmarks/results/native_batch_stage19*.json`；不能与后续FP16/W4A16或只统计decode的实验拼接比较。

历史elementwise Graph实验中，BF16的1×1024与128×3072形状下，RMSNorm相对Torch为5.38x/4.98x，Residual+RMSNorm为4.75x/3.85x，SiLU×Up为1.89x/1.73x；独立Residual Add为0.89x/0.95x，略慢。每Graph32调用、20次重放，另存eager wall样本，二者不可混比。完整FP16/BF16结果见 [stage07_elementwise_summary.json](stage07_elementwise_summary.json)。

### 后端性能复核与性能分析

2026-10-24的FP16后端复核中，同一cuBLAS Graph基线首末轮 **404→725 TPS**，目前不足以认定dandan FP16端到端更快。W4A16 Graph的PyTorch allocated峰值1054.64MiB，相应FP16为1614.39MiB，降低34.67%；不等于整机显存或无质量损失。64-token复核还发现一组dandan Linear eager/Graph输出digest不同，短序列验收不能覆盖所有生成长度。完整条件、输出差异与复现见 [算子后端性能复核](BACKEND_PERFORMANCE_RECHECK.md)。

真实FP16 eager/Graph CPU+CUDA trace已采集；CPU launch下降，物理GPU kernel总时长接近。带插桩wall time不能作为正常推理性能。热点、命令和Nsight未成功采集的限制见 [推理引擎性能分析指南](PROFILING.md)及 [stage28_profile_summary.json](stage28_profile_summary.json)。

## 计量与HTTP比较约定

| 测量入口 | 范围与含义 |
|---|---|
| `benchmark_paged_sweep.py`、`benchmark_elementwise.py` | 算子Graph/Event与eager wall分开，不能外推tokens/s |
| `benchmark_native_batch.py` | 真实scheduler/runtime，实际输出token与总耗时；重复样本P95不是在线请求P95 |
| `benchmark_speculative.py` | 单请求普通/投机交替执行、输出对拍；TPOT是整体平均，不是逐token流式间隔 |
| `benchmark_http.py` | 建连、网络、服务排队和流解析；closed-loop并发上限，非固定请求到达率 |

HTTP指标保留以下定义：

- latency：worker准备发送至收到DONE或失败；线程池等待计入负载总wall，不计入单请求latency。
- TTFT：请求开始至首个非空content，不是服务端首枚token时间。
- TPOT估计：`(last_content_ms-first_content_ms)/(completion_tokens-1)`，仅token数>1且有可见内容时提供。
- 总TPS：成功请求实际completion token之和/整个测量阶段wall，分母包含失败请求耗时。任一成功请求缺少有效usage时总TPS为null；SSE chunk数不充当token数。
- P50/P95：原始请求样本的线性插值分位数；warmup完全结束后开始测量，错误率和失败记录独立保留。

`--backend-manifest` 的comparison至少包含checkpoint/tokenizer固定revision或hash、chat_template_sha256、dtype/quantization、kv_cache_dtype、max_model_len、kv_cache_bytes和prefix_cache_policy；驱动/CUDA等影响条件也应补入。`implementation`记录具体引擎、启动参数及构建指纹。

`--compare`检查model、负载hash、sampling、OS/GPU声明、comparison manifest、逐请求实际token数及输出文本hash；有失败、缺usage/原始记录或条件不一致即拒绝倍率。通过检查仅说明声明相符，`remote_conditions_independently_verified=false`仍保留。当前没有vLLM/SGLang同条件实测、完整任务质量集或长期服务压测结果。

## Chat完整decode Graph与单请求配对测量（2026-10-24）

阶段29全量回归 **356 passed、2 warnings**，CUDA实际执行。新增覆盖FP16/BF16跨页输出、多轮bucket复用、单token/EOS不capture、容量与capture失败回收、batch活跃slot缩减，以及实际token吞吐统计。完整Qwen3-0.6B CLI已实际进行两轮Graph对话和`/clear`，正常退出。实现调用链见[架构](ARCHITECTURE.md)；原始样本在`benchmarks/results/chat_graph_stage29.json`，可提交记录见[紧凑摘要](stage29_chat_graph_summary.json)。

另对完整FP16模型执行3请求×16输出的真实batch eager/Graph回归，各模式2次测量，每次实际48 token，跨模式与模式内输出digest一致；仅作为batch功能检查，不纳入以下配对性能结论。

性能配置：RTX5080、FP16、CUDA prefill/decode/elementwise/data、cuBLAS Linear、融合QKV/GateUp、CUDA greedy；无量化、无profiler。相同runner/权重，两个固定prompt，固定64/128输出、忽略EOS。每case先单列首次调用、各模式预热2次，再6对交替执行eager→Graph/Graph→eager。全部逐token输出与finish reason一致，测量期无capture，源码与后端开始/结束指纹相同。

| prompt / 输出 | TPOT P50 eager→Graph（ms） | TPOT P95 eager→Graph（ms） | 总TPS eager→Graph | 总TPS观察比 |
|---|---:|---:|---:|---:|
| 英文KV / 64 | 16.719→2.530 | 17.519→2.570 | 59.13→357.51 | 6.05x |
| 英文KV / 128 | 15.583→2.657 | 17.400→2.784 | 62.83→355.04 | 5.65x |
| 中文batching / 64 | 16.938→2.587 | 18.877→2.705 | 57.73→342.11 | 5.93x |
| 中文batching / 128 | 16.862→2.648 | 17.868→2.670 | 60.82→355.11 | 5.84x |

TTFT P50分别为20.60→20.44、19.63→19.75、18.91→23.49、22.18→24.40 ms，没有一致改善。首个8/16-block bucket的预热+capture为66.21/38.91 ms，计入各自首次请求，热启动表格排除这些调用。单请求只持有1页scratch，退出释放。多轮Chat仍重新prefill完整消息，不表示复用了会话KV。

TPOT是每次请求生成间隔的平均值，表中P50/P95为六次请求的分位数，并非逐token间隔分位数。总TPS是实际输出token数之和除以请求latency之和，latency从prefill到末token，排除加载、tokenization和文本decode；完整外层call TPS另在摘要记录。收益包含Graph执行、metadata复用和图内采样，不能归因于某个kernel独立加速。WDDM GPU未锁定独占，样本较少；这不是HTTP并发性能、长上下文或MiniVLLM/vLLM排名。

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_chat_graph.py --model models\Qwen3-0.6B --max-new-tokens 64 --max-new-tokens 128 --warmup 2 --repetitions 6 --ignore-eos --output benchmarks\results\chat_graph.json
```

`benchmark_native_qwen3.py --cuda-graph`现已调用实际单请求Graph，`--batch-size`只接受1，真实batch继续使用`benchmark_native_batch.py`。prefill、CPU控制流与请求调度仍在图外；当前Chat Graph只支持greedy，不与投机同时使用。

## 复现入口

在项目根目录PowerShell执行，模型与可选dandan依赖须已准备好。测试和验收使用以下命令，性能实验另在受控GPU条件下运行。

```powershell
. .\scripts\enter_project.ps1
$env:MINI_ENGINE_TEST_CUDA = '1'
.\.venv\Scripts\python -m pytest -q -o cache_dir=build/pytest_cache

.\.venv\Scripts\python benchmarks\validate_system_extensions.py --fuse-projections --http-smoke
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --dtype fp16 --linear-backend dandan_cuda --prefill-backend dandan --fuse-projections
.\.venv\Scripts\python benchmarks\validate_system_extensions.py --dtype fp16 --prefill-backend dandan --quantization int4 --quant-backend dandan --fuse-projections
.\.venv\Scripts\python benchmarks\validate_dandan_qwen3.py --model models\Qwen3-0.6B --linear-backend dandan_cuda --prefill-backend dandan --fuse-projections --decode-steps 4
```

```powershell
.\.venv\Scripts\python benchmarks\benchmark_paged_sweep.py --block-sizes 16 32 64 128 256 --batches 1 8 32 --contexts 128 512 2048 --dtype fp16 --output benchmarks\results\paged_sweep.json
.\.venv\Scripts\python benchmarks\benchmark_speculative.py --model models\Qwen3-0.6B --dtype bf16 --fuse-projections --ignore-eos --max-new-tokens 32 --warmup 1 --repetitions 3 --output benchmarks\results\speculative.json
.\.venv\Scripts\python benchmarks\benchmark_native_batch.py --model models\Qwen3-0.6B --dtype bf16 --max-new-tokens 32 --warmup 1 --repetitions 3 --local-files-only --output benchmarks\results\native_batch.json

# 服务启动方法见项目README；远端条件通过backend manifest另行声明
.\.venv\Scripts\python benchmarks\benchmark_http.py --base-url http://127.0.0.1:8000/v1 --engine-label mini_engine --model qwen3-0.6b --server-os 'Windows 11' --gpu 'RTX 5080' --concurrency 4 --repetitions 10 --max-tokens 32 --warmup-requests 2 --seed 41 --output benchmarks\results\http_mini_engine.json
.\.venv\Scripts\python benchmarks\benchmark_http.py --compare benchmarks\results\http_mini_engine.json benchmarks\results\http_other.json --output benchmarks\results\http_comparison.json
```

优先待办：直接paged多query/ragged prefill、FP8/INT8 KV与FP8 GEMM、投机batch/随机校正、sanitizer、受控性能复测及量化质量集。NVFP4/MXFP8与训练型draft仍属后续研究，当前完成项不替代这些验收。
