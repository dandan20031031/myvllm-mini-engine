# Benchmark 目录

当前入口：

| 脚本 | 范围 |
|---|---|
| `benchmark_paged_sweep.py` | 原CUDA与D128向量化kernel配对Graph计时，5种block size；非服务延迟 |
| `benchmark_native_batch.py` | 真实scheduler eager/Graph对照，实际token统计和后端指纹 |
| `benchmark_speculative.py` | 普通greedy/投机输出一致性、接受率和单请求耗时 |
| `benchmark_http.py` | 标准库HTTP/SSE负载，可对兼容服务采集usage、TTFT/P95与吞吐；`--compare`拒绝不匹配条件 |
| `validate_system_extensions.py` | 完整模型chunk/prefix/Graph与可选真实HTTP验证，仅正确性不测速 |
| `profile_native_batch.py` | 已预热真实runtime的PyTorch Profiler/NVTX，CPU范围、GPU kernel、Chrome trace；采集时间不是benchmark |

客户端usage缺失时，不以SSE chunk数代替token数。性能主对照为Nano-vLLM和MiniVLLM，vLLM/SGLang只作架构参考。本仓库尚无三个小型引擎的同条件实测排名。

`mini_engine.benchmarks.runner`提供通用callable计时和JSON输出，适合固定工作量的局部实验；它不会自动校验实际生成token数或模型语义。完整跨引擎对照还需共同离线adapter，不能直接把配置中的`output_tokens`当实测输出。

## Nano-vLLM / MiniVLLM 对照任务

架构、采样/Graph边界与固定版本见[架构报告](../reports/ARCHITECTURE.md)，历史数据、差异和完整协议见[性能评测报告](../reports/VALIDATION_AND_BENCHMARKS.md)。本次只完成审计，尚未实现或运行共同adapter。

1. 固定模型哈希与离线token IDs；统一FP16、TP1、B=1/4/8/16、prompt128/512、输出128，分别eager/Graph。
2. Nano/Mini原版不接受greedy；共同temperature=0.6、top-k禁用/top-p=1，固定输出长度与ignore-EOS适配须披露。随机采样文本不相同不能直接判错，另做teacher-forced数值验证。
3. 使用`load/reset/enqueue/step/close`接口，统一timer只包执行循环；逐次记录actual tokens/elapsed、每请求首token/结束时间和平均TPOT，输出聚合TPS与P50/P95。
4. 首轮无共享前缀；冷/热Prefix、W4A16、HTTP服务各自单列，不把吞吐口径混为单请求延迟。
5. 记录commit、dirty diff、adapter、实际参数/KV dtype、模块`__file__`、软件栈、KV预算与GPU负载；关闭资源在计时之外。

平台情况：Nano当前本地GPU路径依赖NCCL/flash-attn，Windows未验证；Mini已有Windows修复，但editable安装仍指迁移前目录，须显式选定worktree。对照代码放独立环境，不引入本项目运行依赖；缺条件时结果标为未测。

本项目现成命令仍可独立用于内部消融：

```powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_native_batch.py --help
.\.venv\Scripts\python benchmarks\benchmark_chat_graph.py --help
```

示例：

```python
import torch
from mini_engine.benchmarks import BenchmarkConfig, benchmark_callable, save_results

result = benchmark_callable(
    lambda: torch.mm(torch.randn(512, 512, device="cuda"), torch.randn(512, 512, device="cuda")),
    BenchmarkConfig(name="example", repetitions=5, output_tokens=1),
    device="cuda",
)
save_results([result], "benchmarks/results/example.json")
```

端到端 benchmark 应固定模型版本、GPU、dtype、采样参数、prompt 数据和 `max_new_tokens`，并分别报告 TTFT、TPOT、吞吐量、显存峰值以及 P50/P95 延迟。


## 阶段 07 elementwise 对比

~~~powershell
. .\scripts\enter_project.ps1
.\.venv\Scripts\python benchmarks\benchmark_elementwise.py --dtype bf16
.\.venv\Scripts\python benchmarks\benchmark_elementwise.py --dtype fp16 --output benchmarks\results\elementwise_fp16.json
~~~

默认以 rows=[1,128]、hidden=[1024,3072] 做笛卡尔积；4 类算子各先校验输出再测量。
CUDA Graph Event timeline（32 调用/graph）降低 host 调度间隙；另保存 eager wall time，两种口径分开。
Graph 时间仍包含节点调度，不能称为纯 kernel 指令延迟。脚本不测 tokens/s 或模型吞吐，也不比较 vLLM。
