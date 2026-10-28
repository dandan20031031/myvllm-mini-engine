"""Standard-library OpenAI SSE load client for mini-engine/vLLM/SGLang.

Measures client-observed request latency, including transport and server queueing.
SSE content chunks are never treated as tokens: token metrics require usage.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.common import percentiles, sha256_json as digest


DEFAULT_PROMPTS = ["Explain KV Cache in one sentence.", "用一句话解释 continuous batching。"]


def load_workload(path=None):
    """Accept {prompts:[str,...]} or {requests:[{messages:[...]},...]}."""
    document = {"prompts": DEFAULT_PROMPTS} if path is None else json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or len(document) != 1 or next(iter(document)) not in ("prompts", "requests"):
        raise ValueError("workload must contain exactly prompts or requests")
    rows = document.get("requests", document.get("prompts"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("workload must contain at least one request")
    normalized = []
    for row in rows:
        if "prompts" in document:
            if not isinstance(row, str) or not row:
                raise ValueError("prompts must be nonempty strings")
            messages = [{"role": "user", "content": row}]
        else:
            if not isinstance(row, dict) or set(row) != {"messages"}:
                raise ValueError("each workload request must contain only messages")
            messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str) or not message["content"]):
                raise ValueError("messages need a supported role and nonempty string content")
        normalized.append({"messages": [dict(message) for message in messages]})
    return normalized


class StreamProtocolError(ValueError):
    pass


def stream_events(response):
    """Decode SSE framing, including comments and multi-line data fields."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        yield "\n".join(data)


def run_stream_request(endpoint, model, request, *, max_tokens=32, timeout=120.0, api_key=None):
    """Return a raw terminal record, including partial streams and HTTP errors."""
    payload = {
        "model": model, "messages": request["messages"], "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    started = time.perf_counter()
    record = {
        "request_id": request["request_id"], "workload_index": request["workload_index"],
        "repetition": request["repetition"], "status": "error", "http_status": None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "ttft_ms": None, "tpot_ms": None, "latency_ms": None,
        "first_content_ms": None, "last_content_ms": None,
        "completion_tokens": None, "prompt_tokens": None,
        "usage_status": "missing", "usage": None, "content_events": 0,
        "finish_reason": None, "done_received": False, "error": None,
    }
    text = []
    try:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with urlopen(Request(endpoint, data=encoded, headers=headers, method="POST"), timeout=timeout) as response:
            record["http_status"] = response.status
            if not response.headers.get("Content-Type", "").lower().startswith("text/event-stream"):
                raise StreamProtocolError("expected text/event-stream response")
            for event in stream_events(response):
                if event.strip() == "[DONE]":
                    record["done_received"] = True
                    break
                chunk = json.loads(event)
                if not isinstance(chunk, dict):
                    raise StreamProtocolError("SSE JSON payload must be an object")
                if chunk.get("error"):
                    raise StreamProtocolError("server stream error: " + json.dumps(chunk["error"], ensure_ascii=False)[:1024])
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                    record["usage"] = usage
                    if isinstance(usage, dict) and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] >= 0:
                        record["completion_tokens"] = usage["completion_tokens"]
                        record["prompt_tokens"] = usage.get("prompt_tokens") if type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] >= 0 else None
                        record["usage_status"] = "reported"
                    else:
                        record["completion_tokens"] = record["prompt_tokens"] = None
                        record["usage_status"] = "invalid"
                choices = chunk.get("choices", [])
                if not isinstance(choices, list):
                    raise StreamProtocolError("choices must be a list")
                for choice in choices:
                    if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                        raise StreamProtocolError("only one choice at index 0 is supported")
                    delta = choice.get("delta", {})
                    if not isinstance(delta, dict):
                        raise StreamProtocolError("delta must be an object")
                    content = delta.get("content")
                    if content is not None and not isinstance(content, str):
                        raise StreamProtocolError("delta.content must be text or null")
                    if content:
                        observed = (time.perf_counter() - started) * 1000.0
                        if record["first_content_ms"] is None:
                            record["first_content_ms"] = record["ttft_ms"] = observed
                        record["last_content_ms"] = observed
                        record["content_events"] += 1
                        text.append(content)
                    if choice.get("finish_reason") is not None:
                        record["finish_reason"] = choice["finish_reason"]
            if not record["done_received"]:
                raise StreamProtocolError("stream ended without [DONE]")
            record["status"] = "success"
    except HTTPError as exc:
        record["http_status"] = exc.code
        try:
            message = exc.read(65536).decode("utf-8", errors="replace")[:4096]
        except (OSError, HTTPException):
            message = str(exc)
        record["error"] = {"type": "http_error", "message": message}
        exc.close()
    except (OSError, URLError, HTTPException, ValueError) as exc:
        record["error"] = {"type": "protocol_error" if isinstance(exc, (ValueError, UnicodeError)) else "transport_error", "message": str(exc)[:4096]}
    except Exception as exc:
        record["error"] = {"type": "client_error", "message": str(exc)[:4096]}
    finally:
        record["latency_ms"] = (time.perf_counter() - started) * 1000.0
        record["output_text"] = "".join(text)
        record["output_sha256"] = hashlib.sha256(record["output_text"].encode("utf-8")).hexdigest()
        count = record["completion_tokens"]
        if record["status"] == "success" and count is not None and count > 1 and record["first_content_ms"] is not None:
            record["tpot_ms"] = (record["last_content_ms"] - record["first_content_ms"]) / (count - 1)
    return record


def run_phase(endpoint, model, requests, *, concurrency, max_tokens, timeout, api_key=None):
    started = time.perf_counter()
    records = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_stream_request, endpoint, model, request, max_tokens=max_tokens, timeout=timeout, api_key=api_key) for request in requests]
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda row: row["request_id"]), time.perf_counter() - started


def summarize(records, wall_seconds):
    successful = [row for row in records if row["status"] == "success"]
    failed = [row for row in records if row["status"] != "success"]
    known = [row for row in successful if row["completion_tokens"] is not None]
    complete_usage = len(known) == len(successful)
    tokens = sum(row["completion_tokens"] for row in known)
    errors = {}
    for row in failed:
        key = row["error"]["type"] + ":" + str(row["http_status"] or "no_http_status")
        errors[key] = errors.get(key, 0) + 1
    return {
        "requests": len(records), "successful_requests": len(successful), "failed_requests": len(failed),
        "success_rate": len(successful) / len(records) if records else None,
        "http_or_transport_errors": errors, "aggregate_wall_seconds": wall_seconds,
        "successful_request_latency_ms": percentiles(row["latency_ms"] for row in successful),
        "all_terminal_latency_ms": percentiles(row["latency_ms"] for row in records),
        "ttft_ms": percentiles(row["ttft_ms"] for row in successful),
        "tpot_ms": percentiles(row["tpot_ms"] for row in successful),
        "successful_requests_with_reported_usage": len(known),
        "missing_or_invalid_completion_usage_requests": len(successful) - len(known),
        "known_successful_completion_tokens": tokens,
        "total_completion_tokens": tokens if complete_usage else None,
        "total_tokens_per_second": tokens / wall_seconds if complete_usage and wall_seconds > 0 else None,
        "throughput_scope": "Reported completion tokens from successful requests divided by the entire measured phase wall time, including time spent on failed requests. Partial/error stream tokens are excluded. Any successful request with missing/invalid usage makes total token throughput unavailable.",
    }


def compare_reports(left, right):
    """Refuse ratios unless declared conditions and actual output work agree."""
    reasons = []
    for key in ("model", "workload_sha256", "load", "sampling", "server_environment"):
        if left.get(key) != right.get(key):
            reasons.append("mismatched " + key)
    environment = left.get("server_environment", {})
    if any(not environment.get(key) or environment[key] == "unspecified" for key in ("os", "gpu")):
        reasons.append("server OS and GPU must be declared")
    required = ("checkpoint_id", "tokenizer_id", "chat_template_sha256", "dtype", "quantization", "kv_cache_dtype", "max_model_len", "kv_cache_bytes", "prefix_cache_policy")
    a = left.get("backend_manifest", {}).get("comparison", {})
    b = right.get("backend_manifest", {}).get("comparison", {})
    if not a or any(key not in a or a[key] in (None, "", "unspecified") for key in required):
        reasons.append("missing backend_manifest.comparison conditions")
    if a != b:
        reasons.append("mismatched backend comparison conditions")
    def outputs(report):
        return [(row["request_id"], row.get("completion_tokens"), row.get("prompt_tokens"), row.get("output_sha256")) for row in report.get("records", [])]
    if outputs(left) != outputs(right):
        reasons.append("actual output token counts or text differ")
    for report in (left, right):
        records = report.get("records", [])
        if not records or any(
            type(row.get("completion_tokens")) is not int or row["completion_tokens"] < 0
            or type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] < 0
            or not row.get("output_sha256") for row in records
        ):
            reasons.append("missing raw output/tokenization evidence")
            break
        summary = report.get("summary", {})
        if summary.get("failed_requests") or summary.get("total_tokens_per_second") is None or not summary.get("successful_requests"):
            reasons.append("failed requests or unavailable token throughput")
            break
    return {
        "status": "conditions_match_user_declared" if not reasons else "comparison_refused",
        "reasons": reasons,
        "left_engine": left.get("engine_label"), "right_engine": right.get("engine_label"),
        "right_over_left_throughput_ratio": right["summary"]["total_tokens_per_second"] / left["summary"]["total_tokens_per_second"] if not reasons and left["summary"]["total_tokens_per_second"] > 0 else None,
        "remote_conditions_independently_verified": False,
        "note": "Engine labels establish no equivalence. Matching user-supplied manifests and observed outputs permit an observed ratio, not proof that remote hardware, weights or cache state matched.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--engine-label", default="mini_engine")
    parser.add_argument("--model")
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--backend-manifest", type=Path)
    parser.add_argument("--server-os", default="unspecified")
    parser.add_argument("--gpu", default="unspecified")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT_JSON", "RIGHT_JSON"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "http.json")
    args = parser.parse_args()
    if args.compare:
        report = compare_reports(*(json.loads(Path(path).read_text(encoding="utf-8-sig")) for path in args.compare))
    else:
        if not args.model:
            parser.error("--model is required for HTTP load tests")
        if min(args.concurrency, args.repetitions, args.max_tokens) <= 0 or args.warmup_requests < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("concurrency/repetitions/max-tokens/timeout must be positive and warmup-requests nonnegative")
        parsed = urlsplit(args.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            parser.error("base-url must be an HTTP(S) URL without credentials, query or fragment")
        endpoint = args.base_url.rstrip("/") + "/chat/completions"
        try:
            workload = load_workload(args.workload)
            manifest = {} if args.backend_manifest is None else json.loads(args.backend_manifest.read_text(encoding="utf-8-sig"))
            if not isinstance(manifest, dict):
                raise ValueError("backend manifest must be a JSON object")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        requests = [dict(row, request_id=f"measured-{rep:06d}-{index:06d}", workload_index=index, repetition=rep) for rep in range(args.repetitions) for index, row in enumerate(workload)]
        random.Random(args.seed).shuffle(requests)
        warmup = [dict(workload[index % len(workload)], request_id=f"warmup-{index:06d}", workload_index=index % len(workload), repetition=-1) for index in range(args.warmup_requests)]
        options = dict(concurrency=args.concurrency, max_tokens=args.max_tokens, timeout=args.timeout, api_key=os.getenv(args.api_key_env))
        warmup_records, warmup_wall = run_phase(endpoint, args.model, warmup, **options)
        records, wall = run_phase(endpoint, args.model, requests, **options)
        report = {
            "schema_version": 1, "engine_label": args.engine_label, "model": args.model,
            "endpoint": endpoint, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "client_os": platform.platform(), "python": platform.python_version(),
            "server_environment": {"os": args.server_os, "gpu": args.gpu, "source": "user_supplied"},
            "backend_manifest": manifest, "backend_manifest_sha256": digest(manifest),
            "workload": workload, "workload_sha256": digest(workload),
            "load": {"concurrency": args.concurrency, "repetitions": args.repetitions, "max_tokens": args.max_tokens, "warmup_requests": args.warmup_requests, "client_order_seed": args.seed, "timeout_seconds": args.timeout},
            "sampling": {"temperature": 0, "stream": True, "stream_options": {"include_usage": True}},
            "summary": summarize(records, wall), "records": records,
            "warmup": {"summary": summarize(warmup_records, warmup_wall), "records": warmup_records},
            "notes": [
                "Request latency begins inside the HTTP worker immediately before sending and ends at [DONE] or failure; it includes connection setup, transport, server queueing and stream parsing. Client executor queue delay is excluded from individual request latency but included in aggregate phase wall time.",
                "TTFT observes first nonempty delta.content. TPOT=(last_content_ms-first_content_ms)/(reported completion_tokens-1) only when completion_tokens>1. SSE events/chunks are diagnostic counts, never token counts.",
                "Greedy temperature is fixed at zero. Seed controls client workload ordering, not server RNG. Connections are opened separately per request; no persistent connection pool is used.",
                "All request records are retained, including HTTP errors and partial streams. Warmup is complete before the measured phase and its records are excluded from measured percentiles/throughput.",
                "OS/GPU/backend manifests describe user-supplied server conditions. An engine label does not prove equal hardware, checkpoint, tokenizer, precision, cache state or workload.",
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("status") == "comparison_refused":
        raise SystemExit(2)
    if report.get("summary", {}).get("failed_requests"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
