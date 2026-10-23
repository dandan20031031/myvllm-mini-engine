"""CPU fake-SSE tests: chunks are not tokens, failures are not dropped."""

import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading
import time

import pytest


spec = importlib.util.spec_from_file_location("http_benchmark", Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_http.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture
def fake_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.received.append(payload)
            assert payload["stream_options"] == {"include_usage": True}
            kind = payload["messages"][-1]["content"]
            if kind == "http_error":
                body = b'{"error":{"message":"queue full"}}'
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            with self.server.lock:
                self.server.active += 1
                self.server.peak = max(self.server.peak, self.server.active)
            def send(event):
                text = event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)
                self.wfile.write(("data: " + text + "\n\n").encode("utf-8"))
                self.wfile.flush()
            try:
                send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]})
                time.sleep(0.01)
                if kind != "empty":
                    send({"choices": [{"index": 0, "delta": {"content": "ab"}}]})
                    time.sleep(0.01)
                    send({"choices": [{"index": 0, "delta": {"content": "c"}}]})
                if kind == "invalid_json":
                    send("this is not JSON")
                    return
                if kind == "stream_error":
                    send({"error": {"message": "injected failure"}})
                    return
                send({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
                if kind != "missing_usage":
                    send({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": True if kind == "bad_usage" else (1 if kind == "one_token" else 5), "total_tokens": 12}})
                if kind != "missing_done":
                    send("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with self.server.lock:
                    self.server.active -= 1
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.received, server.lock = [], threading.Lock()
    server.active = server.peak = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(kind="normal", index=0):
    return {"request_id": f"r-{index}", "workload_index": index, "repetition": 0, "messages": [{"role": "user", "content": kind}]}


def test_content_chunks_are_not_tokens_and_tpot_uses_usage(fake_server):
    server, url = fake_server
    row = benchmark.run_stream_request(url, "fake", request())
    assert row["status"] == "success" and row["done_received"]
    assert row["content_events"] == 2 and row["completion_tokens"] == 5
    assert row["prompt_tokens"] == 7 and row["output_text"] == "abc"
    assert row["ttft_ms"] == row["first_content_ms"] > 0
    assert row["last_content_ms"] > row["first_content_ms"]
    assert row["tpot_ms"] == pytest.approx((row["last_content_ms"] - row["first_content_ms"]) / 4)
    assert row["latency_ms"] >= row["last_content_ms"]
    assert server.received[0]["temperature"] == 0


@pytest.mark.parametrize("kind,status", [("missing_usage", "missing"), ("bad_usage", "invalid")])
def test_missing_or_invalid_usage_disables_token_metrics(fake_server, kind, status):
    _, url = fake_server
    row = benchmark.run_stream_request(url, "fake", request(kind))
    assert row["status"] == "success" and row["usage_status"] == status
    assert row["completion_tokens"] is None and row["tpot_ms"] is None
    assert row["content_events"] == 2 and row["ttft_ms"] is not None
    summary = benchmark.summarize([row], 1)
    assert summary["total_completion_tokens"] is None
    assert summary["total_tokens_per_second"] is None


@pytest.mark.parametrize("kind", ["one_token", "empty"])
def test_tpot_unavailable_without_two_tokens_or_visible_content(fake_server, kind):
    _, url = fake_server
    row = benchmark.run_stream_request(url, "fake", request(kind))
    assert row["status"] == "success" and row["tpot_ms"] is None
    if kind == "empty":
        assert row["ttft_ms"] is None and row["content_events"] == 0


@pytest.mark.parametrize("kind,error_type", [("http_error", "http_error"), ("missing_done", "protocol_error"), ("invalid_json", "protocol_error"), ("stream_error", "protocol_error")])
def test_failures_keep_raw_record_and_partial_output(fake_server, kind, error_type):
    _, url = fake_server
    row = benchmark.run_stream_request(url, "fake", request(kind))
    assert row["status"] == "error" and row["error"]["type"] == error_type
    assert row["latency_ms"] > 0 and row["tpot_ms"] is None
    if kind == "http_error":
        assert row["http_status"] == 429 and "queue full" in row["error"]["message"]
    else:
        assert row["http_status"] == 200 and row["output_text"] == "abc"


def test_transport_failure_is_retained():
    row = benchmark.run_stream_request("http://127.0.0.1:1/v1/chat/completions", "fake", request(), timeout=0.1)
    assert row["status"] == "error" and row["error"]["type"] == "transport_error"
    assert row["http_status"] is None and row["completion_tokens"] is None


def test_concurrency_and_errors_are_counted_over_aggregate_wall(fake_server):
    server, url = fake_server
    requests = [request(index=index) for index in range(5)] + [request("http_error", 5)]
    rows, wall = benchmark.run_phase(url, "fake", requests, concurrency=3, max_tokens=8, timeout=2)
    summary = benchmark.summarize(rows, wall)
    assert len(rows) == 6 and summary["successful_requests"] == 5 and summary["failed_requests"] == 1
    assert summary["http_or_transport_errors"] == {"http_error:429": 1}
    assert summary["total_completion_tokens"] == 25
    assert summary["total_tokens_per_second"] == pytest.approx(25 / wall)
    assert 2 <= server.peak <= 3
    assert summary["successful_request_latency_ms"]["count"] == 5
    assert summary["all_terminal_latency_ms"]["count"] == 6


def test_percentiles_use_request_samples_and_are_defined_for_single_sample():
    assert benchmark.percentiles([]) == {"count": 0, "p50": None, "p95": None}
    assert benchmark.percentiles([None, 7]) == {"count": 1, "p50": 7, "p95": 7}
    assert benchmark.percentiles([10, 30, 20]) == {"count": 3, "p50": 20, "p95": 29}


def test_sse_comment_multiline_data_and_final_unterminated_frame():
    framed = io.BytesIO(b': keepalive\n\ndata: {"choices":\ndata: []}\n\ndata: [DONE]')
    assert list(benchmark.stream_events(framed)) == ['{"choices":\n[]}', '[DONE]']


def test_workload_normalization_hash_and_validation(tmp_path):
    path = tmp_path / "workload.json"
    path.write_text(json.dumps({"prompts": ["hello", "你好"]}), encoding="utf-8")
    prompts = benchmark.load_workload(path)
    path.write_text(json.dumps({"requests": prompts}), encoding="utf-8")
    messages = benchmark.load_workload(path)
    assert messages == prompts and benchmark.digest(messages) == benchmark.digest(prompts)
    path.write_text(json.dumps({"requests": prompts}), encoding="utf-8-sig")
    assert benchmark.load_workload(path) == prompts  # Windows PowerShell UTF-8 BOM
    path.write_text('{"prompts": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.load_workload(path)


def test_cli_warmup_is_separate_and_never_writes_api_key(fake_server, tmp_path, monkeypatch, capsys):
    server, url = fake_server
    workload = tmp_path / "workload.json"
    workload.write_text('{"prompts":["normal"]}', encoding="utf-8")
    output = tmp_path / "result.json"
    monkeypatch.setenv("TEST_BENCHMARK_API_KEY", "secret-value-do-not-record")
    monkeypatch.setattr(sys, "argv", ["benchmark_http.py", "--base-url", url.removesuffix("/chat/completions"), "--model", "fake", "--workload", str(workload), "--warmup-requests", "2", "--repetitions", "3", "--concurrency", "2", "--api-key-env", "TEST_BENCHMARK_API_KEY", "--output", str(output)])
    benchmark.main()
    printed = capsys.readouterr().out
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["summary"]["requests"] == 3
    assert document["warmup"]["summary"]["requests"] == 2
    assert all(row["request_id"].startswith("measured-") for row in document["records"])
    assert len(server.received) == 5
    assert "secret-value-do-not-record" not in output.read_text(encoding="utf-8") + printed


def comparison_report(label="left"):
    conditions = {key: "same" for key in ("checkpoint_id", "tokenizer_id", "chat_template_sha256", "dtype", "quantization", "kv_cache_dtype", "max_model_len", "kv_cache_bytes", "prefix_cache_policy")}
    return {
        "engine_label": label, "model": "same-model", "workload_sha256": "same-workload",
        "load": {"concurrency": 2}, "sampling": {"temperature": 0},
        "server_environment": {"os": "Windows 11", "gpu": "RTX 5080", "source": "user_supplied"},
        "backend_manifest": {"comparison": conditions},
        "records": [{"request_id": "same", "completion_tokens": 5, "prompt_tokens": 7, "output_sha256": "same-output"}],
        "summary": {"failed_requests": 0, "successful_requests": 1, "total_tokens_per_second": 100},
    }


def test_comparison_checks_actual_conditions_not_engine_labels():
    left, right = comparison_report(), comparison_report("vllm")
    right["summary"]["total_tokens_per_second"] = 200
    result = benchmark.compare_reports(left, right)
    assert result["right_over_left_throughput_ratio"] == 2
    assert result["remote_conditions_independently_verified"] is False
    right["load"]["concurrency"] = 4
    refused = benchmark.compare_reports(left, right)
    assert refused["status"] == "comparison_refused" and refused["right_over_left_throughput_ratio"] is None


@pytest.mark.parametrize("field", ["output", "usage", "manifest", "environment", "http_error", "missing_prompt_usage", "missing_raw_records"])
def test_comparison_refuses_incomplete_or_different_work(field):
    left, right = comparison_report(), comparison_report()
    if field == "output":
        right["records"][0]["output_sha256"] = "different"
    elif field == "usage":
        right["summary"]["total_tokens_per_second"] = None
    elif field == "manifest":
        right["backend_manifest"] = {}
    elif field == "environment":
        right["server_environment"]["gpu"] = "unspecified"
    elif field == "missing_prompt_usage":
        left["records"][0]["prompt_tokens"] = right["records"][0]["prompt_tokens"] = None
    elif field == "missing_raw_records":
        left["records"] = right["records"] = []
    else:
        right["summary"]["failed_requests"] = 1
    assert benchmark.compare_reports(left, right)["status"] == "comparison_refused"
