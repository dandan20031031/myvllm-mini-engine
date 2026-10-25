"""One bounded inference worker owns tokenizer, scheduler, and GPU runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import math
import queue
import threading
import time
import uuid

from ..engine.request import GenerationRequest, RequestState
from .protocol import APIError, validate_chat_request

LOG = logging.getLogger(__name__)


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
        return validate_chat_request(payload, model_name=self.model_name,
                                     max_new_tokens=self.max_new_tokens)

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
                try:
                    runtime.close()
                except Exception:
                    # Runtime owns its resources; HTTP waiters must still be
                    # released when device cleanup reports an error.
                    LOG.exception("Unable to close inference runtime")
            jobs = list(pending) + list(active.values())
            while True:
                try:
                    jobs.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            for job in jobs:
                self._publish(job, "error", APIError(503, "Inference service is shutting down", code="unavailable"))
                self._release(job)
