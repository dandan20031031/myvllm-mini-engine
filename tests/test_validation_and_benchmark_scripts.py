from pathlib import Path


def test_validation_and_benchmark_scripts_are_present_and_project_local():
    root = Path(__file__).parents[1]
    validation = (root / "benchmarks" / "validate_qwen3.py").read_text(encoding="utf-8")
    benchmark = (root / "benchmarks" / "benchmark_native_qwen3.py").read_text(encoding="utf-8")
    assert "status" in validation and "blocked" in validation
    assert "median_tpot_ms" in benchmark and "native_qwen3" in benchmark
    assert "C:\\" not in validation + benchmark
