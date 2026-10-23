"""Real loopback HTTP/SSE tests; fake compute never loads a model or GPU."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import http.client
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from mini_engine.cache import BlockManager
from mini_engine.engine.request import RequestState
from mini_engine.server import APIError, ChatHTTPServer, InferenceService


class FakeTokenizer:
    def __init__(self, runner):
        self.runner = runner

    def apply_chat_template(self, messages, **kwargs):
        self.runner.record()
        text = messages[-1]["content"]
        if text == "bad-template":
            raise ValueError("template rejected")
        return [99 if text == "explode" else 7] * len(text)

    def decode(self, tokens, **kwargs):
        self.runner.record()
        return "".join(chr(96 + token) for token in tokens)


class FakeScheduler:
    def __init__(self, runner):
        self.runner = runner
        self.block_manager = BlockManager(64, 4)
        self.waiting, self.running = deque(), {}

    def add_request(self, request):
        self.runner.record()
        self.waiting.append(request)

    def abort(self, request_id):
        self.runner.record()
        request = self.running.pop(request_id, None)
        if request is None:
            request = next((item for item in self.waiting if item.request_id == request_id), None)
            if request is not None:
                self.waiting.remove(request)
        if request is not None:
            request.state = RequestState.ABORTED
        self.block_manager.release(request_id)


class FakeRuntime:
    def __init__(self, runner):
        self.runner = runner
        self.scheduler = FakeScheduler(runner)
        self.prefix_cache = SimpleNamespace(clear=runner.record)
        self.enable_prefix_cache = True

    def step(self):
        self.runner.record()
        while self.scheduler.waiting:
            request = self.scheduler.waiting.popleft()
            request.state = RequestState.RUNNING
            request.block_ids = self.scheduler.block_manager.allocate(request.request_id, request.prompt_len)
            self.scheduler.running[request.request_id] = request
        current = list(self.scheduler.running.values())
        self.runner.max_batch = max(self.runner.max_batch, len(current))
        self.runner.step_started.set()
        if self.runner.step_gate is not None:
            self.runner.step_gate.wait(3)
        time.sleep(self.runner.step_delay)
        if any(request.prompt_token_ids[0] == 99 for request in current):
            raise RuntimeError("injected compute failure")
        for request in current:
            request.append_token(request.generated_tokens % 26 + 1)
            if request.generated_tokens >= request.max_new_tokens:
                request.finish("length")
                self.scheduler.running.pop(request.request_id)
                self.scheduler.block_manager.release(request.request_id)
            else:
                request.block_ids = self.scheduler.block_manager.grow(request.request_id, request.total_len)
        return {request.request_id: request.output_token_ids[-1] for request in current}


class FakeRunner:
    def __init__(self, step_delay=0.005, step_gate=None):
        self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=256))
        self.enable_thinking = False
        self.thread_ids, self.max_batch = set(), 0
        self.step_started = threading.Event()
        self.step_delay, self.step_gate = step_delay, step_gate
        self.tokenizer = FakeTokenizer(self)
        self.runtime = None

    def record(self):
        self.thread_ids.add(threading.get_ident())

    def create_batch_runtime(self, **kwargs):
        self.record()
        self.runtime = FakeRuntime(self)
        self.runtime.enable_prefix_cache = kwargs.get("enable_prefix_cache", True)
        return self.runtime


@contextmanager
def running_server(runner=None, **service_kwargs):
    runner = runner or FakeRunner()
    service = InferenceService(runner, model_name="tiny", **service_kwargs).start()
    server = ChatHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server, service, runner
    finally:
        if runner.step_gate is not None:
            runner.step_gate.set()
        server.shutdown()
        service.close()
        server.server_close()
        thread.join(2)


def payload(**updates):
    return {"model": "tiny", "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3, **updates}


def request(server, body=None, *, path="/v1/chat/completions", method="POST"):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    raw = None if body is None else json.dumps(body)
    connection.request(method, path, body=raw, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < end
        time.sleep(0.01)


def test_health_models_nonstream_and_single_worker_ownership():
    with running_server() as (server, service, runner):
        status, raw = request(server, path="/health", method="GET")
        assert status == 200 and json.loads(raw)["status"] == "ok"
        status, raw = request(server, path="/v1/models", method="GET")
        assert status == 200 and json.loads(raw)["data"][0]["id"] == "tiny"
        status, raw = request(server, payload())
        result = json.loads(raw)
        assert status == 200
        assert result["choices"][0]["message"] == {"role": "assistant", "content": "abc"}
        assert result["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        assert len(runner.thread_ids) == 1
        assert threading.get_ident() not in runner.thread_ids
        assert runner.runtime.enable_prefix_cache


def test_sse_role_content_finish_and_done():
    with running_server() as (server, _, _):
        status, raw = request(server, payload(stream=True))
        assert status == 200
        events = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(item) for item in events[:-1]]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert "".join(item["choices"][0]["delta"].get("content", "") for item in chunks) == "abc"
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_sse_optional_usage_is_final_chunk_before_done():
    with running_server() as (server, _, _):
        status, raw = request(server, payload(stream=True, stream_options={"include_usage": True}))
        events = [line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")]
        assert status == 200 and events[-1] == "[DONE]"
        usage_chunk = json.loads(events[-2])
        assert usage_chunk["choices"] == []
        assert usage_chunk["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        assert json.loads(events[-3])["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("updates,expected_status", [
    ({"model": "missing"}, 404), ({"temperature": 0.5}, 400),
    ({"temperature": True}, 400), ({"stream": "true"}, 400),
    ({"max_tokens": True}, 400), ({"max_tokens": 0}, 400),
    ({"max_tokens": 513}, 400), ({"messages": []}, 400),
    ({"messages": [{"role": "tool", "content": "hi"}]}, 400),
    ({"messages": [{"role": "user", "content": [{"text": "hi"}]}]}, 400),
    ({"n": 2}, 400), ({"tools": []}, 400),
    ({"stream_options": {"include_usage": True}}, 400),
    ({"stream": True, "stream_options": {"include_usage": "true"}}, 400),
    ({"stream": True, "stream_options": {"extra": True}}, 400),
])
def test_strict_input_validation(updates, expected_status):
    with running_server() as (server, _, runner):
        status, raw = request(server, payload(**updates))
        assert status == expected_status
        assert "error" in json.loads(raw)
        assert runner.max_batch == 0


def test_context_capacity_and_tokenizer_errors_do_not_break_later_requests():
    with running_server(max_context=8) as (server, _, _):
        status, raw = request(server, payload(messages=[{"role": "user", "content": "long-prompt"}]))
        assert status == 400 and json.loads(raw)["error"]["code"] == "context_length_exceeded"
        status, _ = request(server, payload(messages=[{"role": "user", "content": "bad-template"}]))
        assert status == 400
        status, _ = request(server, payload())
        assert status == 200
    # A valid model context can still exceed the independently bounded KV pool.
    runner = FakeRunner()
    runner.model.config.max_position_embeddings = 1024
    with running_server(runner) as (server, _, _):
        status, raw = request(server, payload(messages=[{"role": "user", "content": "x" * 254}], max_tokens=3))
        assert status == 400 and json.loads(raw)["error"]["code"] == "kv_capacity_exceeded"


def test_independent_http_requests_share_continuous_batch():
    runner = FakeRunner(step_delay=0.015)
    with running_server(runner) as (server, _, _):
        with ThreadPoolExecutor(max_workers=4) as clients:
            futures = [clients.submit(request, server, payload(max_tokens=12)) for _ in range(4)]
            assert all(future.result()[0] == 200 for future in futures)
        assert runner.max_batch >= 2
        assert len(runner.thread_ids) == 1
        assert runner.runtime.scheduler.block_manager.used_blocks == 0


def test_bounded_queue_rejects_overload_and_recovers():
    gate = threading.Event()
    runner = FakeRunner(step_gate=gate)
    with running_server(runner, max_pending=1) as (server, _, _):
        with ThreadPoolExecutor(max_workers=1) as clients:
            first = clients.submit(request, server, payload())
            assert runner.step_started.wait(2)
            status, raw = request(server, payload())
            assert status == 429 and json.loads(raw)["error"]["code"] == "queue_full"
            gate.set()
            assert first.result()[0] == 200
        assert request(server, payload())[0] == 200


def test_compute_error_releases_batch_and_worker_accepts_next_request():
    with running_server() as (server, _, runner):
        status, raw = request(server, payload(messages=[{"role": "user", "content": "explode"}]))
        assert status == 500 and json.loads(raw)["error"]["code"] == "inference_error"
        assert request(server, payload())[0] == 200
        assert runner.runtime.scheduler.block_manager.used_blocks == 0


def test_disconnected_sse_client_cancels_live_request_and_releases_kv():
    runner = FakeRunner(step_delay=0.02)
    with running_server(runner) as (server, _, _):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("POST", "/v1/chat/completions", json.dumps(payload(stream=True, max_tokens=128)),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert response.readline().startswith(b"data: ")
        response.close()
        connection.close()
        wait_for(lambda: not runner.runtime.scheduler.running and not runner.runtime.scheduler.waiting)
        assert runner.runtime.scheduler.block_manager.used_blocks == 0
        assert request(server, payload())[0] == 200


def test_shutdown_cancels_requests_and_releases_kv():
    runner = FakeRunner(step_delay=0.02)
    with running_server(runner) as (server, service, _):
        with ThreadPoolExecutor(max_workers=1) as clients:
            pending = clients.submit(request, server, payload(max_tokens=128))
            assert runner.step_started.wait(2)
            service.close()
            assert pending.result()[0] == 503
        assert runner.runtime.scheduler.block_manager.used_blocks == 0
        assert not service.healthy


def test_bad_http_framing_and_json_are_rejected():
    with running_server() as (server, _, _):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("POST", "/v1/chat/completions", b"{", {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
        assert request(server, payload(temperature=float("nan")))[0] == 400
        assert request(server, path="/unknown", method="GET")[0] == 404


def test_streaming_waits_for_complete_unicode_bytes():
    runner = FakeRunner()
    def unicode_decode(tokens, **kwargs):
        runner.record()
        return {0: "", 1: "\ufffd", 2: "你", 3: "你好"}[len(tokens)]
    runner.tokenizer.decode = unicode_decode
    with running_server(runner) as (server, _, _):
        status, raw = request(server, payload(stream=True))
        chunks = [json.loads(line[6:]) for line in raw.decode().splitlines()
                  if line.startswith("data: {")]
        assert status == 200
        assert "".join(item["choices"][0]["delta"].get("content", "") for item in chunks) == "你好"


def test_prefix_cache_can_be_explicitly_disabled():
    with running_server(enable_prefix_cache=False) as (server, _, runner):
        assert request(server, payload())[0] == 200
        assert not runner.runtime.enable_prefix_cache


def test_raw_http_body_limits_duplicate_fields_and_media_type():
    with running_server() as (server, _, _):
        for body, headers, expected in [
            (b'{"model":"tiny","model":"tiny"}', {"Content-Type": "application/json"}, 400),
            (b'{}', {"Content-Type": "text/plain"}, 415),
            (b'{}', {"Content-Type": "application/json", "Content-Length": "99999999"}, 413),
            (b'{}', {"Content-Type": "application/json", "Transfer-Encoding": "chunked"}, 400),
        ]:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request("POST", "/v1/chat/completions", body, headers)
            response = connection.getresponse()
            assert response.status == expected
            response.read()
            connection.close()


def test_slow_event_consumer_is_bounded_and_cancelled():
    runner = FakeRunner()
    service = InferenceService(runner, model_name="tiny", output_queue_size=1).start()
    try:
        job = service.submit(payload(stream=True, max_tokens=128))
        # Deliberately do not drain the one-element output buffer.
        wait_for(lambda: job.released)
        assert job.cancelled.is_set()
        kind, error = job.events.get_nowait()
        assert kind == "error" and error.code == "slow_consumer"
        assert runner.runtime.scheduler.block_manager.used_blocks == 0
    finally:
        service.close()
