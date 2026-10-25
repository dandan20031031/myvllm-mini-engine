"""Bounded HTTP connections and OpenAI-compatible JSON/SSE serialization."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import queue
import select
import socket
import threading
import time

from .protocol import APIError

LOG = logging.getLogger(__name__)


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
