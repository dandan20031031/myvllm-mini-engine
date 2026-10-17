from mini_engine.benchmarks import BenchmarkConfig, benchmark_callable


def test_benchmark_returns_metrics():
    result = benchmark_callable(lambda: sum(range(10)), BenchmarkConfig("sum", repetitions=2, output_tokens=4))
    assert result.name == "sum"
    assert len(result.latencies_ms) == 2
    assert result.tokens_per_second > 0
