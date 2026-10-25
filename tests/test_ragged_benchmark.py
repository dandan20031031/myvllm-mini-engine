"""Ragged prefill measurement scope, numerical checks and failure cleanup."""

import json

import pytest
import torch

import benchmarks.benchmark_ragged_prefill as entry
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config


class Tokenizer:
    def encode(self, text, **kwargs):
        return [1, 2, 3]


def tiny_runner():
    torch.manual_seed(991)
    config = Qwen3Config(31, 32, 64, 2, 4, 2, 16, max_position_embeddings=64)
    model = NativeQwen3(config, "math").eval()
    return NativeQwen3Runner(model, Tokenizer(), num_blocks=16, block_size=4, eos_token_ids=[])


def test_three_modes_isolate_logit_selection_and_packed_batch():
    runner = tiny_runner()
    head_rows = []
    hook = runner.model.lm_head.register_forward_pre_hook(lambda module, args: head_rows.append(args[0].shape[0]))
    try:
        with entry.prepared_case(runner, [[1, 2, 3, 4, 5], [2, 3, 4], [3, 4, 5, 6]]) as case:
            assert case["last_rows"] == (4, 7, 11)
            assert all(isinstance(rows, tuple) for rows in case["single_rows"])
            outputs = [entry.execute_mode(runner, case, mode).clone() for mode in entry.MODES]
            assert case["total_tokens"] == 12
            assert len(set(page for pages in case["pages"] for page in pages)) == 4
            assert head_rows == [5, 3, 4, 1, 1, 1, 3]
            for actual in outputs[1:]:
                torch.testing.assert_close(actual, outputs[0], atol=1e-4, rtol=1e-4)
    finally:
        hook.remove()
        runner.close()
    assert runner.block_manager.used_blocks == 0


def test_cpu_pilot_records_actual_input_tokens_and_correctness_without_generation(monkeypatch):
    runner = tiny_runner()
    monkeypatch.setattr(runner, "generate", lambda *a, **k: pytest.fail("forward benchmark called generation"))
    result = entry.run_case(runner, [[1, 2, 3, 4, 5], [2, 3, 4]], warmup=0, repetitions=2,
                            atol=1e-4, rtol=1e-4)
    assert result["total_prompt_tokens"] == 8
    assert result["prompt_lengths"] == [5, 3]
    assert result["correctness"]["passed"] and result["correctness"]["kv_checked"]
    assert result["comparison_allowed"]
    for mode in entry.MODES:
        assert [row["prefill_input_tokens"] for row in result["samples"][mode]] == [8, 8]
        assert result["summary"][mode]["wall_ms"]["count"] == 2
        assert result["summary"][mode]["cuda_event_timeline_ms"]["mean"] is None
    assert runner.block_manager.used_blocks == 0


def test_forward_failure_returns_only_benchmark_owned_pages(monkeypatch):
    runner = tiny_runner()
    caller_page = runner.block_manager.allocate("caller", 4)[0]
    original = runner.model.forward_prefill_batch
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("injected forward failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner.model, "forward_prefill_batch", fail_after_first)
    with pytest.raises(RuntimeError, match="injected forward"):
        entry.run_case(runner, [[1, 2, 3], [2, 3]], warmup=0, repetitions=1)
    assert runner.block_manager.used_blocks == 1
    assert runner.block_manager.refcount(caller_page) == 1


def test_numerically_wrong_packed_forward_is_rejected_before_timing(monkeypatch):
    runner = tiny_runner()
    original = runner.model.forward_prefill_batch

    def corrupt(batch, *args, **kwargs):
        result = original(batch, *args, **kwargs)
        return result + 5 if len(batch.query_lengths) > 1 else result

    monkeypatch.setattr(runner.model, "forward_prefill_batch", corrupt)
    monkeypatch.setattr(entry, "measure_forward", lambda *a, **k: pytest.fail("invalid path reached measurement"))
    with pytest.raises(AssertionError):
        entry.run_case(runner, [[1, 2, 3], [2, 3]], warmup=0, repetitions=1, atol=1e-4, rtol=1e-4)
    assert runner.block_manager.used_blocks == 0


def test_input_throughput_uses_actual_tokens_over_total_forward_time():
    def row(tokens, wall):
        return {"wall_ms": wall, "cuda_event_timeline_ms": None, "prefill_input_tokens": tokens,
                "peak_allocated_bytes": None, "additional_peak_allocated_bytes": None,
                "peak_reserved_bytes": None}
    result = entry.summarize_samples([row(2, 10), row(8, 90)])
    assert result["prefill_input_tokens_per_second"] == 100
    assert result["wall_ms"]["mean"] == 50


def main_args(output):
    return ["--model", "local-only", "--device", "cpu", "--dtype", "fp32", "--prefill-backend", "math",
            "--elementwise-backend", "torch", "--data-backend", "torch", "--linear-backend", "torch",
            "--batch-sizes", "1", "2", "--prompt-lengths", "4", "--warmup", "0", "--repetitions", "1",
            "--output", str(output)]


@pytest.mark.parametrize("source_changed", [False, True])
def test_main_runs_local_weights_only_and_disables_changed_source_ratios(monkeypatch, tmp_path, source_changed):
    runner = tiny_runner()
    loads, closes, fingerprints = [], [], []
    original_close = runner.close

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return runner

    def close():
        closes.append(True)
        original_close()

    def fingerprint():
        value = "changed" if fingerprints and source_changed else "same"
        fingerprints.append(value)
        return {"source_tree_sha256": value}

    monkeypatch.setattr(entry.NativeQwen3Runner, "from_pretrained", load)
    monkeypatch.setattr(runner, "close", close)
    monkeypatch.setattr(entry, "source_fingerprint", fingerprint)
    monkeypatch.setattr(entry, "gpu_snapshot", lambda: {"available": False})
    output = tmp_path / "prefill.json"
    result = entry.main(main_args(output))
    assert loads[0][1]["local_files_only"] is True
    assert closes == [True] and runner.block_manager.used_blocks == 0
    assert result["status"] == ("source_or_backend_changed" if source_changed else "passed")
    assert len(result["cases"]) == 2
    if source_changed:
        assert all(not case["comparison_allowed"] and case["sequential_selected_over_ragged_wall_ratio"] is None
                   for case in result["cases"])
    assert json.loads(output.read_text(encoding="utf-8"))["kind"] == "ragged_cold_prefill_forward"


def test_main_failure_closes_runner_and_saves_failed_record(monkeypatch, tmp_path):
    runner = tiny_runner()
    closes = []
    monkeypatch.setattr(entry, "load_runner", lambda args: runner)
    monkeypatch.setattr(entry, "source_fingerprint", lambda: {"source_tree_sha256": "same"})
    monkeypatch.setattr(entry, "gpu_snapshot", lambda: {})
    monkeypatch.setattr(runner, "close", lambda: closes.append(True))

    def fail(*args, **kwargs):
        raise RuntimeError("injected case failure")

    monkeypatch.setattr(entry, "run_case", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="injected case"):
        entry.main(main_args(output))
    assert closes == [True]
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed" and record["error"]["type"] == "RuntimeError"
    assert record["source_unchanged"] is True and record["backend_unchanged"] is None


def test_metadata_failure_preserves_original_execution_error_and_writes_record(monkeypatch, tmp_path):
    runner = tiny_runner()
    calls, closes = [], []

    def metadata():
        calls.append(True)
        if len(calls) > 1:
            raise OSError("metadata unavailable")
        return {"source_tree_sha256": "same"}

    def fail(*args, **kwargs):
        raise RuntimeError("original forward failure")

    monkeypatch.setattr(entry, "load_runner", lambda args: runner)
    monkeypatch.setattr(entry, "run_case", fail)
    monkeypatch.setattr(entry, "source_fingerprint", metadata)
    monkeypatch.setattr(entry, "gpu_snapshot", lambda: {})
    monkeypatch.setattr(runner, "close", lambda: closes.append(True))
    output = tmp_path / "metadata-failure.json"
    with pytest.raises(RuntimeError, match="original forward"):
        entry.main(main_args(output))
    assert closes == [True]
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["error"]["type"] == "RuntimeError"
    assert record["metadata_end_error"]["type"] == "OSError"
    assert record["source_unchanged"] is None


def test_explicit_skip_kv_is_exploratory_without_comparison_ratios(monkeypatch, tmp_path):
    runner = tiny_runner()
    monkeypatch.setattr(entry, "load_runner", lambda args: runner)
    monkeypatch.setattr(entry, "source_fingerprint", lambda: {"source_tree_sha256": "same"})
    monkeypatch.setattr(entry, "gpu_snapshot", lambda: {})
    result = entry.main(main_args(tmp_path / "exploratory.json") + ["--skip-kv-check"])
    assert result["status"] == "validation_incomplete"
    for case in result["cases"]:
        assert case["correctness"]["passed"] and not case["correctness"]["kv_checked"]
        assert not case["comparison_allowed"]
        assert case["comparison_blockers"] == ["kv_not_checked"]
        assert case["sequential_full_over_selected_wall_ratio"] is None
        assert case["sequential_selected_over_ragged_wall_ratio"] is None


def test_diagnostic_reports_per_head_scale_and_nonfinite_counts():
    from benchmarks.diagnose_ragged_prefill import clean_json, kv_comparison, stats

    reference = torch.full((1, 2, 2, 2), 4.0)
    actual = reference.clone()
    actual[0, 0, 1, 0] += 1
    report = kv_comparison(actual, reference, atol=0.1, rtol=0)
    assert report["mismatched"] == 1 and report["elements"] == 8
    assert report["layers"][0]["head_max_error_over_max_reference"] == [0, 0.25]
    assert report["layers"][0]["head_relative_rms_error"] == [0, 0.125]
    invalid = stats(torch.tensor([float("nan"), 1.]), torch.ones(2), atol=0.1, rtol=0)
    assert invalid["actual_nonfinite"] == 1 and invalid["mismatched"] == 1
    json.dumps(clean_json(invalid), allow_nan=False)


def test_diagnostic_runs_independent_greedy_paths_and_preserves_page_ownership():
    from benchmarks.diagnose_ragged_prefill import run_variant

    runner = tiny_runner()
    other = runner.block_manager.allocate("caller", 4)[0]
    report, _ = run_variant(runner, [[1, 2, 3, 4, 5], [2, 3, 4]], output_tokens=4, atol=1e-4, rtol=1e-4)
    assert report["strict_elementwise_validation_passed"]
    assert report["comparisons"]["sequential_selected_logits"]["key"]["max_abs"] == 0
    assert all(len(tokens) == 4 for mode in report["greedy_outputs"].values() for tokens in mode)
    assert runner.block_manager.used_blocks == 1 and runner.block_manager.refcount(other) == 1


def test_close_failure_invalidates_all_measured_case_ratios(monkeypatch, tmp_path):
    runner = tiny_runner()
    monkeypatch.setattr(entry, "load_runner", lambda args: runner)
    monkeypatch.setattr(entry, "source_fingerprint", lambda: {"source_tree_sha256": "same"})
    monkeypatch.setattr(entry, "gpu_snapshot", lambda: {})

    def fail_close():
        raise RuntimeError("injected close failure")

    monkeypatch.setattr(runner, "close", fail_close)
    output = tmp_path / "close-failure.json"
    with pytest.raises(RuntimeError, match="injected close"):
        entry.main(main_args(output))
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "failed" and record["source_and_backend_unchanged"]
    assert record["close_error"]["type"] == "RuntimeError"
    assert runner.block_manager.used_blocks == 0
    for case in record["cases"]:
        assert not case["comparison_allowed"] and "run_failed" in case["comparison_blockers"]
        assert case["sequential_full_over_selected_wall_ratio"] is None
        assert case["sequential_selected_over_ragged_wall_ratio"] is None
