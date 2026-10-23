"""Small Windows-native OpenAI chat subset over one batching worker.

HTTP threads only validate/serialize I/O. Tokenization, runtime construction,
scheduler mutation, model execution and KV cleanup belong to the worker.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import queue
import select
import socket
import threading
import time
import uuid

from .engine.request import GenerationRequest, RequestState

LOG = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status, message, *, code="invalid_request", param=None):
        super().__init__(message)
        self.status, self.message, self.code, self.param = status, message, code, param

    def payload(self):
        return {"error": {"message": self.message, "type": "server_error" if self.status >= 500
                          else "invalid_request_error", "param": self.param, "code": self.code}}


@dataclass
class _Job:
    messages: list
    max_tokens: int
    stream: bool
    events: queue.Queue
    deadline: float
    include_usage: bool = False
    request_id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    cancelled: threading.Event = field(default_factory=threading.Event)
    request: GenerationRequest | None = None
    reserved_blocks: int = 0
    published_text: str = ""
    decoded_tokens: int = 0
    released: bool = False


class InferenceService:
    """Bounded submission queue feeding one NativeBatchRuntime continuously.

    Reserve each admitted request's complete KV budget. The runtime may evict
    retained prefix entries while preserving references owned by live requests.
    Decode batching combines independently arriving requests.
    """

    def __init__(self, runner, *, model_name="qwen3-0.6b", max_pending=32,
                 max_context=None, max_new_tokens=512, output_queue_size=32,
                 request_timeout=120.0, use_cuda_graph=False, enable_prefix_cache=True,
                 prefill_chunk_size=128, max_num_batched_tokens=4096):
        if (not model_name or max_pending <= 0 or max_new_tokens <= 0
                or output_queue_size <= 0 or not math.isfinite(request_timeout) or request_timeout <= 0):
            raise ValueError("service limits must be positive")
        if max_num_batched_tokens <= 0 or (prefill_chunk_size is not None and prefill_chunk_size <= 0):
            raise ValueError("batch and chunk token limits must be positive")
        model_context = runner.model.config.max_position_embeddings
        self.max_context = model_context if max_context is None else min(max_context, model_context)
        if self.max_context <= 0:
            raise ValueError("max_context must be positive")
        self.runner, self.model_name = runner, model_name
        self.max_new_tokens, self.request_timeout = max_new_tokens, request_timeout
        self.output_queue_size, self.use_cuda_graph = output_queue_size, use_cuda_graph
        self.enable_prefix_cache = enable_prefix_cache
        self.prefill_chunk_size, self.max_num_batched_tokens = prefill_chunk_size, max_num_batched_tokens
        self._queue = queue.Queue(max_pending)
        self._slots = threading.BoundedSemaphore(max_pending)
        self._stop, self._ready = threading.Event(), threading.Event()
        self._submission_lock = threading.Lock()
        self._startup_error = None
        self._thread = threading.Thread(target=self._run, name="mini-engine-worker", daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(30):
            self._stop.set()
            raise RuntimeError("inference worker initialization timed out")
        if self._startup_error is not None:
            raise RuntimeError("inference worker initialization failed") from self._startup_error
        return self

    @property
    def healthy(self):
        return self._ready.is_set() and self._thread.is_alive() and not self._stop.is_set()

    def close(self, timeout=10.0):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout)
        if self._thread.is_alive():
            raise RuntimeError("inference worker is still executing; shutdown cannot interrupt a GPU kernel")

    def submit(self, payload):
        messages, max_tokens, stream, include_usage = self._validate(payload)
        with self._submission_lock:
            if not self.healthy:
                raise APIError(503, "Inference service is unavailable", code="unavailable")
            if not self._slots.acquire(blocking=False):
                raise APIError(429, "Request queue is full", code="queue_full")
            job = _Job(messages, max_tokens, stream, queue.Queue(self.output_queue_size),
                       time.monotonic() + self.request_timeout, include_usage=include_usage)
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self._slots.release()
                raise APIError(429, "Request queue is full", code="queue_full") from None
        return job

    def _validate(self, payload):
        if not isinstance(payload, dict):
            raise APIError(400, "JSON body must be an object")
        allowed = {"model", "messages", "max_tokens", "stream", "temperature", "stream_options"}
        if set(payload) - allowed:
            raise APIError(400, "Unsupported fields: " + ", ".join(sorted(set(payload) - allowed)))
        if payload.get("model") != self.model_name:
            raise APIError(404, "Unknown model", code="model_not_found", param="model")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
            raise APIError(400, "messages must contain 1 to 128 text messages", param="messages")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str) or not message["content"]):
                raise APIError(400, "Each message needs a supported role and nonempty string content", param="messages")
        tokens = payload.get("max_tokens", min(128, self.max_new_tokens))
        if type(tokens) is not int or not 1 <= tokens <= self.max_new_tokens:
            raise APIError(400, f"max_tokens must be an integer in [1, {self.max_new_tokens}]", param="max_tokens")
        stream = payload.get("stream", False)
        if type(stream) is not bool:
            raise APIError(400, "stream must be a boolean", param="stream")
        stream_options = payload.get("stream_options", {})
        if (not isinstance(stream_options, dict) or set(stream_options) - {"include_usage"}
                or type(stream_options.get("include_usage", False)) is not bool
                or ("stream_options" in payload and not stream)):
            raise APIError(400, "stream_options only supports include_usage:boolean with stream=true", param="stream_options")
        temperature = payload.get("temperature", 0)
        if type(temperature) not in (int, float) or temperature != 0:
            raise APIError(400, "Only greedy decoding is supported; temperature must be 0", param="temperature")
        return [dict(message) for message in messages], tokens, stream, stream_options.get("include_usage", False)

    def _publish(self, job, kind, value):
        try:
            job.events.put_nowait((kind, value))
            return True
        except queue.Full:
            # A slow SSE reader cannot block the GPU worker or grow RAM.
            job.cancelled.set()
            while True:
                try:
                    job.events.get_nowait()
                except queue.Empty:
                    break
            job.events.put_nowait(("error", APIError(503, "Response consumer is too slow", code="slow_consumer")))
            return False

    def _release(self, job):
        if not job.released:
            job.released = True
            self._slots.release()

    def _decode_text(self, request):
        return self.runner.tokenizer.decode(request.output_token_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

    def _run(self):
        pending, active = deque(), {}
        runtime = None
        try:
            runtime = self.runner.create_batch_runtime(use_cuda_graph=self.use_cuda_graph,
                enable_prefix_cache=self.enable_prefix_cache, prefill_chunk_size=self.prefill_chunk_size,
                max_num_batched_tokens=self.max_num_batched_tokens)
            manager = runtime.scheduler.block_manager
            capacity = manager.num_blocks - manager.used_blocks
            graphs = getattr(runtime, "graph_manager", None)
            max_active = max(graphs.batch_sizes) if graphs is not None else self._queue.maxsize
            graph_context = max(graphs.block_buckets) * manager.block_size if graphs is not None else self.max_context
            if capacity <= 0:
                raise ValueError("No KV blocks remain after runtime initialization")
            reserved = 0
            self._ready.set()
            while not self._stop.is_set():
                while True:
                    try:
                        job = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if job.cancelled.is_set():
                        self._release(job)
                        continue
                    try:
                        ids = self.runner.tokenizer.apply_chat_template(
                            job.messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=self.runner.enable_thinking)
                        if hasattr(ids, "tolist"):
                            ids = ids.tolist()
                        if not ids or any(type(token) is not int for token in ids):
                            raise ValueError("tokenizer returned invalid token IDs")
                        if len(ids) + job.max_tokens > self.max_context:
                            raise APIError(400, "Prompt plus max_tokens exceeds the context limit", code="context_length_exceeded")
                        if len(ids) + job.max_tokens > graph_context:
                            raise APIError(400, "Prompt plus max_tokens exceeds CUDA Graph bucket coverage", code="graph_context_exceeded")
                        budget = manager.blocks_required(len(ids) + job.max_tokens)
                        if budget > capacity:
                            raise APIError(400, "Request exceeds available KV capacity", code="kv_capacity_exceeded")
                        job.request = GenerationRequest(job.request_id, ids, max_new_tokens=job.max_tokens)
                        job.reserved_blocks = budget
                        pending.append(job)
                        self._publish(job, "accepted", None)
                    except Exception as exc:
                        error = exc if isinstance(exc, APIError) else APIError(400, "Unable to encode messages", code="invalid_messages")
                        self._publish(job, "error", error)
                        self._release(job)
                for job in list(pending):
                    if job.cancelled.is_set() or time.monotonic() >= job.deadline:
                        pending.remove(job)
                        self._publish(job, "error", APIError(504, "Request timed out or was cancelled", code="cancelled"))
                        self._release(job)
                    elif len(active) < max_active and reserved + job.reserved_blocks <= capacity:
                        pending.remove(job)
                        try:
                            runtime.scheduler.add_request(job.request)
                            active[job.request_id] = job
                            reserved += job.reserved_blocks
                        except Exception:
                            LOG.exception("Unable to admit request")
                            runtime.scheduler.abort(job.request_id)
                            self._publish(job, "error", APIError(500, "Unable to admit request", code="inference_error"))
                            self._release(job)
                for request_id, job in list(active.items()):
                    if job.cancelled.is_set() or time.monotonic() >= job.deadline:
                        runtime.scheduler.abort(request_id)
                        active.pop(request_id)
                        reserved -= job.reserved_blocks
                        if not job.cancelled.is_set():
                            self._publish(job, "error", APIError(504, "Request timed out", code="timeout"))
                        self._release(job)
                if active:
                    try:
                        runtime.step()
                        for request_id, job in list(active.items()):
                            request = job.request
                            if request.state is RequestState.ABORTED:
                                self._publish(job, "error", APIError(500, "Inference request failed", code="inference_error"))
                                active.pop(request_id)
                                reserved -= job.reserved_blocks
                                self._release(job)
                                continue
                            if not request.is_finished and (not job.stream or
                                    len(request.output_token_ids) == job.decoded_tokens):
                                continue
                            text = self._decode_text(request)
                            job.decoded_tokens = len(request.output_token_ids)
                            if job.stream and text.startswith(job.published_text):
                                # Incomplete UTF-8 byte tokens may decode to a
                                # trailing replacement character; wait for the
                                # remaining bytes before exposing that suffix.
                                stable = text if request.is_finished else text.rstrip("\ufffd")
                                delta = stable[len(job.published_text):]
                                if delta:
                                    self._publish(job, "delta", delta)
                                    job.published_text = stable
                            if request.is_finished:
                                self._publish(job, "complete", {"text": text,
                                    "finish_reason": request.finished_reason or "stop",
                                    "prompt_tokens": request.prompt_len,
                                    "completion_tokens": len(request.output_token_ids)})
                                active.pop(request_id)
                                reserved -= job.reserved_blocks
                                self._release(job)
                    except Exception:
                        # A batch kernel failure cannot reliably be attributed
                        # to one row. Fail its active requests, release all KV,
                        # and keep the worker available for later submissions.
                        LOG.exception("Inference batch failed")
                        for request_id, job in list(active.items()):
                            runtime.scheduler.abort(request_id)
                            self._publish(job, "error", APIError(500, "Inference batch failed", code="inference_error"))
                            self._release(job)
                        active.clear()
                        reserved = 0
                else:
                    self._stop.wait(0.005)
        except BaseException as exc:
            self._startup_error = exc
            LOG.exception("Inference worker stopped")
        finally:
            with self._submission_lock:
                self._stop.set()
            self._ready.set()
            if runtime is not None:
                for request_id in list(active):
                    runtime.scheduler.abort(request_id)
                runtime.prefix_cache.clear()
                runtime.scheduler.block_manager.release("__graph_padding__")
            jobs = list(pending) + list(active.values())
            while True:
                try:
                    jobs.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for job in jobs:
                self._publish(job, "error", APIError(503, "Inference service is shutting down", code="unavailable"))
                self._release(job)


class ChatHTTPServer(ThreadingHTTPServer):
    """Bound connection handlers too, including idle/slow HTTP clients."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, service, *, max_connections=64, max_body_bytes=1_048_576,
                 socket_timeout=5.0):
        if max_connections <= 0 or max_body_bytes <= 0 or socket_timeout <= 0:
            raise ValueError("HTTP limits must be positive")
        self.service = service
        self.max_body_bytes, self.socket_timeout = max_body_bytes, socket_timeout
        self._connections = threading.BoundedSemaphore(max_connections)
        super().__init__(address, _ChatHandler)

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            try:
                request.settimeout(self.socket_timeout)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.socket_timeout)

    def log_message(self, format, *args):
        LOG.info("%s %s", self.client_address[0], format % args)

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200 if self.server.service.healthy else 503,
                       {"status": "ok" if self.server.service.healthy else "unavailable"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.server.service.model_name,
                "object": "model", "created": 0, "owned_by": "mini-engine"}]})
        else:
            self._json(404, APIError(404, "Unknown endpoint").payload())

    def _disconnected(self):
        readable, _, _ = select.select([self.connection], [], [], 0)
        return bool(readable and self.connection.recv(1, socket.MSG_PEEK) == b"")

    def _next(self, job):
        while True:
            if time.monotonic() >= job.deadline:
                raise APIError(504, "Request timed out", code="timeout")
            if self._disconnected():
                raise ConnectionError("client disconnected")
            try:
                return job.events.get(timeout=0.1)
            except queue.Empty:
                if not self.server.service.healthy:
                    raise APIError(503, "Inference service is unavailable", code="unavailable")

    def _sse(self, payload):
        data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _chunk(self, job, delta, finish_reason=None):
        return {"id": job.request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": self.server.service.model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    @staticmethod
    def _usage(value):
        return {"prompt_tokens": value["prompt_tokens"], "completion_tokens": value["completion_tokens"],
                "total_tokens": value["prompt_tokens"] + value["completion_tokens"]}

    @staticmethod
    def _finite_json(raw):
        def reject_constant(value):
            raise ValueError(f"nonfinite number: {value}")

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate field: {key}")
                result[key] = value
            return result

        return json.loads(raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)

    def do_POST(self):
        job, streaming = None, False
        try:
            if self.path != "/v1/chat/completions":
                raise APIError(404, "Unknown endpoint")
            if self.headers.get("Transfer-Encoding"):
                raise APIError(400, "Chunked request bodies are unsupported")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                raise APIError(411, "Exactly one Content-Length header is required")
            try:
                length = int(lengths[0])
            except ValueError:
                raise APIError(400, "Invalid Content-Length") from None
            if not 0 < length <= self.server.max_body_bytes:
                raise APIError(413, "Request body exceeds the size limit")
            if self.headers.get_content_type() != "application/json":
                raise APIError(415, "Content-Type must be application/json")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise APIError(400, "Incomplete request body")
            try:
                payload = self._finite_json(raw)
            except (UnicodeDecodeError, ValueError):
                raise APIError(400, "Body must be valid finite JSON") from None
            job = self.server.service.submit(payload)
            kind, value = self._next(job)
            if kind == "error":
                raise value
            if job.stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                streaming = True
                self._sse(self._chunk(job, {"role": "assistant"}))
            while True:
                kind, value = self._next(job)
                if kind == "error":
                    raise value
                if kind == "delta" and job.stream:
                    self._sse(self._chunk(job, {"content": value}))
                if kind == "complete":
                    if job.stream:
                        self._sse(self._chunk(job, {}, value["finish_reason"]))
                        if job.include_usage:
                            chunk = self._chunk(job, {})
                            chunk.update(choices=[], usage=self._usage(value))
                            self._sse(chunk)
                        self._sse("[DONE]")
                    else:
                        self._json(200, {"id": job.request_id, "object": "chat.completion",
                            "created": int(time.time()), "model": self.server.service.model_name,
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": value["text"]},
                                         "finish_reason": value["finish_reason"]}],
                            "usage": self._usage(value)})
                    return
        except APIError as exc:
            try:
                if streaming:
                    self._sse(exc.payload())
                    self._sse("[DONE]")
                else:
                    self._json(exc.status, exc.payload())
            except OSError:
                pass
        except (OSError, ConnectionError):
            pass
        finally:
            if job is not None:
                job.cancelled.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="qwen3-0.6b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--prefill-backend", choices=("auto", "math", "cuda", "dandan"), default="auto")
    parser.add_argument("--decode-backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--elementwise-backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--data-backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--linear-backend", choices=("torch", "cuda", "dandan", "dandan_cuda"), default="torch")
    parser.add_argument("--sampling-backend", choices=("torch", "cuda"), default="torch")
    parser.add_argument("--quantization", choices=("none", "int4"), default="none")
    parser.add_argument("--quant-backend", choices=("reference", "dandan"), default="reference")
    parser.add_argument("--quant-group-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-pending", type=int, default=32)
    parser.add_argument("--max-context", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.cuda_graph and (not args.device.startswith("cuda") or args.decode_backend != "cuda"
                            or args.data_backend != "cuda" or args.dtype == "fp32"):
        parser.error("--cuda-graph requires CUDA FP16/BF16, --decode-backend cuda and --data-backend cuda")
    import torch
    from .config import QuantConfig
    from .model import NativeQwen3Runner
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    runner = NativeQwen3Runner.from_pretrained(args.model, device=args.device, dtype=dtype,
        prefill_backend=args.prefill_backend, decode_backend=args.decode_backend,
        elementwise_backend=args.elementwise_backend, data_backend=args.data_backend,
        linear_backend=args.linear_backend, num_blocks=args.num_blocks, block_size=args.block_size,
        sampling_backend=args.sampling_backend,
        local_files_only=args.local_files_only, fuse_projections=args.fuse_projections,
        quant_backend=args.quant_backend, quantization=None if args.quantization == "none" else
        QuantConfig("int4", activation_dtype=args.dtype, group_size=args.quant_group_size))
    service = InferenceService(runner, model_name=args.served_model_name, max_pending=args.max_pending,
        max_context=args.max_context, max_new_tokens=args.max_new_tokens,
        request_timeout=args.request_timeout, use_cuda_graph=args.cuda_graph,
        enable_prefix_cache=not args.disable_prefix_cache, prefill_chunk_size=args.prefill_chunk_size,
        max_num_batched_tokens=args.max_num_batched_tokens).start()
    server = None
    try:
        server = ChatHTTPServer((args.host, args.port), service)
        print(f"MiniQwenEngine listening on http://{args.host}:{server.server_port}; greedy decoding")
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        service.close()


if __name__ == "__main__":
    main()
