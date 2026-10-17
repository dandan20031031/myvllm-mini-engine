from pathlib import Path


def test_paged_benchmark_script_is_project_local():
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_paged_attention.py"
    source = path.read_text(encoding="utf-8")
    assert "benchmarks" in source and "results" in source
    assert "C:\\" not in source
