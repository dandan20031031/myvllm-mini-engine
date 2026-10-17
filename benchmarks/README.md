# Benchmark 目录

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
