# 推理引擎性能分析指南

记录日期：2026-10-24（原阶段28）。实测摘要见 [stage28_profile_summary.json](stage28_profile_summary.json)，常规性能结论见 [系统验证与性能评测](VALIDATION_AND_BENCHMARKS.md)。

## 入口与数据

新增`benchmarks/profile_native_batch.py`，加载完整本地Qwen3，构造真实Scheduler/NativeBatchRuntime；固定输入token形状和输出数，关闭Prefix缓存。模型加载、tokenizer、CUDA编译、Graph capture及warmup均在采集之前完成。

脚本用临时wrapper标记`engine/prefix_lookup`、`schedule`、`prefill`、`decode`、`sampling`和Graph buffer/replay；模型算子标记`op/linear`、`rms_norm`、`apply_rope_backend`、`paged_attention`等。`--layers`将投影细分到具体层。退出与异常时恢复全部方法/函数，默认引擎不增加profiling开销。

- `--tool torch`：PyTorch Profiler，输出`trace.json`、`operators.txt`、`summary.json`。
- `--tool nvtx`：仅NVTX标记，供Nsight采集；不与PyTorch Profiler竞争CUPTI。
- `--cuda-profiler-api`：预热后调用cudaProfilerStart/Stop，限定外部采集窗口。
- `--cuda-graph`：分析真实decode replay；`--shapes/--memory/--stacks`按需打开，避免第一轮采集过重。

所有产物放项目D盘`build/profiles`。本机PyTorch报告支持CPU和CUDA activity，实际Chrome trace中包含CUDA kernel时间线。

## 首轮正确性与热点

工具验证2项测试通过，覆盖输出保持、范围标记、异常退出与wrapper恢复，以及请求页回收。真实Qwen3-0.6B FP16/cuBLAS、batch1、128-token prompt、8输出、fusion，在eager和Graph两条路径采集成功，与各自未插桩warmup输出一致。

| 观察项 | eager trace | Graph trace |
|---|---:|---:|
| 物理GPU kernel总时长（求和，可能有流重叠） | 24.64 ms | 24.00 ms |
| cuBLAS GEMV主要kernel | 11.13 ms | 11.09 ms |
| D128 PagedAttention kernel | 4.60 ms | 4.25 ms |
| 自有prefill kernel | 2.73 ms | 2.74 ms |
| CPU cudaLaunchKernel self时间 | 35.26 ms | 6.48 ms |

带插桩wall time为240.05/77.45ms，**含profiler和Python标记开销，不用于声称正常推理速度或计算GPU利用率**。物理kernel时间基本相近，CPU launch明显下降，与Graph减少host提交的机制一致。eager的CPU decode包装、Linear调用和norm包装开销也较多；Graph下CPU `cudaMemcpyAsync`约19.35ms，提示后续应核对小metadata复制、buffer填充与D2H采样同步。

GPU热点首先是GEMV/Linear，其次是当前形状的PagedAttention。不能只看kernel时间最长就忽略CPU空洞，也不能因为`cudaStreamSynchronize`耗时就认定同步调用本身有问题：它可能在等待前面的GPU工作。

注意CPU/device可以重叠，父子range的inclusive时间也会重复计算；`op/*` CUDA注解覆盖kernel。脚本单独从`cat=kernel`统计物理kernel，不能把所有`key_averages`的CUDA时间相加。GPU kernel求和仍不等于多stream的critical path。

## PyTorch Profiler复现

```powershell
Set-Location -LiteralPath 'D:\dandan123\dandan\myvllm'
. .\scripts\enter_project.ps1

.\.venv\Scripts\python benchmarks\profile_native_batch.py `
  --dtype fp16 --batch-size 1 --prompt-tokens 128 --max-new-tokens 8 `
  --output-dir build\profiles\fp16_eager

.\.venv\Scripts\python benchmarks\profile_native_batch.py `
  --dtype fp16 --batch-size 1 --prompt-tokens 128 --max-new-tokens 8 `
  --cuda-graph --output-dir build\profiles\fp16_graph
```

`operators.txt`先看Self CPU、调用次数与Self Device；`trace.json`可在[Perfetto](https://ui.perfetto.dev)通过Open trace file打开，查看CPU线程、CUDA API、GPU stream和memcpy。`--layers`主要用于eager：已经capture的Graph replay不再执行Python层循环。

后端消融：

```powershell
# 仅换Flash prefill，保留cuBLAS Linear
.\.venv\Scripts\python benchmarks\profile_native_batch.py --dtype fp16 `
  --prefill-backend dandan --cuda-graph --output-dir build\profiles\flash_graph

# 另测dandan Linear，比较op/linear和GPU GEMM/GEMV
.\.venv\Scripts\python benchmarks\profile_native_batch.py --dtype fp16 `
  --linear-backend dandan --prefill-backend dandan --layers `
  --output-dir build\profiles\dandan_eager

# W4A16（与FP16质量/存储分开比较）
.\.venv\Scripts\python benchmarks\profile_native_batch.py --dtype fp16 `
  --prefill-backend dandan --quantization int4 --cuda-graph `
  --output-dir build\profiles\w4_graph
```

短/长prompt、batch1/4、eager/Graph分别采集。长prompt多并发需要相应增加`--num-blocks`；比如batch4/prompt2048可用1024页。`--prefill-chunk-size`用于区分整段prefill与分块调度。profiling定位后回到无插桩benchmark验证收益。

## Nsight Systems与Nsight Compute

原验收记录中已安装Nsight Systems2026.1.3及Nsight Compute2026.2.1，但Systems CLI返回`No reports were generated`，未成功生成report。因此下面命令是待验证的PowerShell采集配置，**尚未在原执行环境成功验收**；当前已有可用的PyTorch trace。

```powershell
nsys profile --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none `
  --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop `
  --show-output=true --output=build/profiles/nsys_graph `
  .\.venv\Scripts\python.exe benchmarks\profile_native_batch.py `
  --tool nvtx --cuda-profiler-api --cuda-graph --dtype fp16 `
  --batch-size 1 --prompt-tokens 128 --max-new-tokens 8 `
  --output-dir build/profiles/nvtx_graph
```

Graph默认可能以整图显示，`--cuda-graph-trace=node`用于检查内部kernel，额外开销较大。Systems用于看CPU→CUDA API→GPU时间线的空洞和依赖。成功生成`.nsys-rep`后，用Nsight Systems GUI打开，或用`nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum <report>`统计。

确认热点后用Nsight Compute单独采一个kernel，避免全模型每个launch做counter replay：

```powershell
ncu --set basic --kernel-name 'regex:.*attention_vector128.*' `
  --launch-count 1 --profile-from-start off --target-processes all `
  --export build/profiles/pagedattn `
  .\.venv\Scripts\python.exe benchmarks\profile_native_batch.py `
  --tool nvtx --cuda-profiler-api --dtype fp16 --max-new-tokens 8 `
  --output-dir build/profiles/ncu_target
```

Compute看DRAM/SM throughput、memory访问、occupancy、寄存器和warp stall。先定位哪个kernel、再分析瓶颈；本阶段没有在本机采集Compute硬件counter，不能据此给出实测带宽/occupancy。

官方参考：[PyTorch Profiler](https://docs.pytorch.org/docs/2.9/profiler.html)、[Nsight Systems CUDA Graph与capture range](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
