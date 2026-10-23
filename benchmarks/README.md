# Benchmark 目录

当前入口：

| 脚本 | 范围 |
|---|---|
| `benchmark_paged_sweep.py` | 原CUDA与D128向量化kernel配对Graph计时，5种block size；非服务延迟 |
| `benchmark_native_batch.py` | 真实scheduler eager/Graph对照，实际token统计和后端指纹 |
| `benchmark_speculative.py` | 普通greedy/投机输出一致性、接受率和单请求耗时 |
| `benchmark_http.py` | 标准库HTTP/SSE负载，可对兼容服务采集usage、TTFT/P95与吞吐；`--compare`拒绝不匹配条件 |
| `validate_system_extensions.py` | 完整模型chunk/prefix/Graph与可选真实HTTP验证，仅正确性不测速 |

客户端usage缺失时，不以SSE chunk数代替token数。跨引擎数据需附相同模型/精度/硬件/工作负载manifest；本仓库尚无vLLM/SGLang同条件实测结果。

`mini_engine.benchmarks.runner` 提供统一计时和 JSON 输出。它不绑定某个模型后端，因此可以给 Transformers、vLLM 和 MiniQwenEngine 各写一个 adapter，再使用相同的 prompts、warmup、重复次数和输出长度。

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
