"""Full local-model correctness checks for chunks, prefix reuse, graphs and HTTP.

Load model weights once and give each experiment a fresh runner/KV allocator.
This is a validation script: it intentionally reports no throughput, TPOT or
HTTP TTFT, and does not treat graph capture as steady-state service latency.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys
import threading
import traceback
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default=str(ROOT / "models" / "Qwen3-0.6B"))
    result.add_argument("--device", default="cuda")
    result.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    result.add_argument("--prefill-backend", choices=("auto", "math", "cuda", "dandan"), default="cuda")
    result.add_argument("--decode-backend", choices=("torch", "cuda"), default="cuda")
    result.add_argument("--elementwise-backend", choices=("torch", "cuda"), default="cuda")
    result.add_argument("--data-backend", choices=("torch", "cuda"), default="cuda")
    result.add_argument("--linear-backend", choices=("torch", "cuda", "dandan", "dandan_cuda"), default="cuda")
    result.add_argument("--sampling-backend", choices=("torch", "cuda"), default="torch")
    result.add_argument("--quantization", choices=("none", "int4"), default="none")
    result.add_argument("--quant-backend", choices=("reference", "dandan"), default="reference")
    result.add_argument("--quant-group-size", type=int, default=128)
    result.add_argument("--fuse-projections", action="store_true")
    result.add_argument("--block-size", type=int, default=16)
    result.add_argument("--num-blocks", type=int, default=128)
    result.add_argument("--prompt-length", type=int, default=64)
    result.add_argument("--prompt", default="Explain why a paged KV cache helps a language model serve multiple requests. ")
    result.add_argument("--chunk-size", type=int, default=16)
    result.add_argument("--token-budget", type=int, default=128)
    result.add_argument("--max-new-tokens", type=int, default=8)
    result.add_argument("--skip-graphs", action="store_true", help="Run correctness checks on CPU or without graph capture")
    result.add_argument("--http-smoke", action="store_true", help="Also send concurrent JSON and SSE requests to an ephemeral loopback port")
    result.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results" / "system_extensions_validation.json")
    return result


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def counted_prefill(model):
    """Runtime batched decode bypasses Module.__call__; this counts prompt work."""
    chunks = []
    def observe(module, args, kwargs):
        chunks.append({"tokens": int(args[0].numel()), "position": int(kwargs.get("position", 0))})
    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        yield chunks
    finally:
        handle.remove()


def execute(runtime, prompts, output_limit, *, request_prefix="validation"):
    from mini_engine.engine.request import GenerationRequest, RequestState

    requests = [GenerationRequest(f"{request_prefix}-{i}", list(ids), max_new_tokens=output_limit)
                for i, ids in enumerate(prompts)]
    for request in requests:
        runtime.scheduler.add_request(request)
    steps = 0
    # Bound ticks rather than waiting indefinitely for an allocator deadlock.
    limit = sum(len(ids) + output_limit + 1 for ids in prompts) * 4
    while runtime.scheduler.waiting or runtime.scheduler.running:
        runtime.step()
        steps += 1
        require(steps <= limit, "runtime failed to finish within its progress bound")
    for request in requests:
        require(request.state is RequestState.FINISHED, f"request did not finish: {request.request_id}")
        require(len(request.output_token_ids) == output_limit, "fixed-length validation emitted the wrong token count")
        try:
            runtime.scheduler.block_manager.table(request.request_id)
        except KeyError:
            pass
        else:
            raise AssertionError("finished request still owns allocator pages")
    return requests, steps


def fresh_runner(base, args):
    from mini_engine.model import NativeQwen3Runner
    return NativeQwen3Runner(base.model, base.tokenizer, num_blocks=args.num_blocks,
                             block_size=args.block_size, enable_thinking=False,
                             eos_token_ids=[], sampling_backend=args.sampling_backend)


def create_runtime(runner, args, *, chunked, prefix=False, graphs=False, maximum_prompt=0):
    width = max(1, (maximum_prompt + args.max_new_tokens - 2) // args.block_size + 1)
    return runner.create_batch_runtime(
        use_cuda_graph=graphs, graph_batch_sizes=(1, 2, 4), graph_block_buckets=(width,),
        prefill_chunk_size=args.chunk_size if chunked else None,
        max_num_batched_tokens=args.token_budget if chunked else max(args.token_budget, maximum_prompt * 4),
        enable_prefix_cache=prefix,
    )


def cleanup(runtime):
    """Clear cache references and graph scratch, then assert there are no leaks."""
    manager = runtime.scheduler.block_manager
    require(not runtime.scheduler.running and not runtime.scheduler.waiting, "runtime still has live requests")
    cache_pins_before_clear = manager.used_blocks
    runtime.prefix_cache.clear()
    manager.release("__graph_padding__")
    require(manager.free_blocks == manager.num_blocks, "allocator page references leaked after cache/scratch cleanup")
    return {"used_blocks_before_cache_clear": cache_pins_before_clear,
            "used_blocks_after_cleanup": manager.used_blocks,
            "all_page_refcounts_zero": all(manager.refcount(i) == 0 for i in range(manager.num_blocks))}


def validate_chunks(base, args, ids):
    runs = []
    for chunked in (False, True):
        runner = fresh_runner(base, args)
        runtime = create_runtime(runner, args, chunked=chunked, maximum_prompt=len(ids))
        with counted_prefill(base.model) as chunks:
            requests, steps = execute(runtime, [ids], args.max_new_tokens)
        require(sum(row["tokens"] for row in chunks) == len(ids), "chunking recomputed or skipped prompt tokens")
        if chunked:
            require(all(row["tokens"] <= args.chunk_size for row in chunks), "prefill chunk exceeded its size limit")
        runs.append({"chunked": chunked, "output_token_ids": requests[0].output_token_ids,
                     "steps": steps, "prefill_chunks": chunks,
                     "computed_prompt_tokens": sum(row["tokens"] for row in chunks),
                     "allocator": cleanup(runtime)})
        del runtime, runner
    require(runs[0]["output_token_ids"] == runs[1]["output_token_ids"], "chunked and unchunked greedy outputs differ")
    return {"passed": True, "output_match": True, "runs": runs}


def snapshot_prefix(runtime, blocks):
    return runtime.cache.key[:, blocks].clone(), runtime.cache.value[:, blocks].clone()


def prefix_unchanged(runtime, blocks, snapshot):
    import torch
    return torch.equal(runtime.cache.key[:, blocks], snapshot[0]) and torch.equal(runtime.cache.value[:, blocks], snapshot[1])


def validate_prefix(base, args, ids, suffix):
    runner = fresh_runner(base, args)
    runtime = create_runtime(runner, args, chunked=False, prefix=True, maximum_prompt=len(ids) + len(suffix))
    with counted_prefill(base.model) as cold_chunks:
        cold, _ = execute(runtime, [ids], args.max_new_tokens, request_prefix="cold")
    require(sum(row["tokens"] for row in cold_chunks) == len(ids), "cold prefix did not compute the expected prompt")
    entry = runtime.prefix_cache.lookup(ids)
    require(entry is not None and len(entry.token_key) == len(ids), "aligned prompt was not retained as a complete prefix")
    require(entry.next_logits is not None, "aligned prefix has no boundary logits")
    pages = list(entry.block_ids)
    saved = snapshot_prefix(runtime, pages)
    for page in pages:
        require(runner.block_manager.refcount(page) >= 1, "cached prefix page lost its pin")
    with counted_prefill(base.model) as warm_chunks:
        warm, _ = execute(runtime, [ids], args.max_new_tokens, request_prefix="warm")
    require(not warm_chunks, "exact prefix hit executed a model prompt forward")
    require(warm[0].prefix_len == len(ids), "exact-hit prefix length differs from prompt length")
    require(cold[0].output_token_ids == warm[0].output_token_ids, "cold/warm prefix outputs differ")
    require(prefix_unchanged(runtime, pages, saved), "exact hit overwrote shared prefix KV")

    with counted_prefill(base.model) as partial_chunks:
        partial, _ = execute(runtime, [ids + suffix], args.max_new_tokens, request_prefix="partial")
    partial_count = sum(row["tokens"] for row in partial_chunks)
    require(partial_count == len(suffix), "partial prefix hit did not compute exactly its suffix")
    require(partial_chunks[0]["position"] == len(ids), "partial prefix started at the wrong absolute position")
    require(prefix_unchanged(runtime, pages, saved), "partial hit overwrote shared prefix KV")

    other = fresh_runner(base, args)
    cold_runtime = create_runtime(other, args, chunked=False, maximum_prompt=len(ids) + len(suffix))
    reference, _ = execute(cold_runtime, [ids + suffix], args.max_new_tokens, request_prefix="partial-reference")
    require(partial[0].output_token_ids == reference[0].output_token_ids, "partial hit and cold suffix outputs differ")
    result = {
        "passed": True, "cold_computed_prompt_tokens": sum(row["tokens"] for row in cold_chunks),
        "warm_computed_prompt_tokens": 0, "partial_computed_prompt_tokens": partial_count,
        "prefix_tokens": len(ids), "suffix_tokens": len(suffix),
        "cold_output_token_ids": cold[0].output_token_ids, "warm_output_token_ids": warm[0].output_token_ids,
        "partial_output_token_ids": partial[0].output_token_ids,
        "partial_reference_output_token_ids": reference[0].output_token_ids,
        "shared_kv_unchanged": True, "warm_chunks": warm_chunks, "partial_chunks": partial_chunks,
        "allocator": cleanup(runtime), "cold_reference_allocator": cleanup(cold_runtime),
    }
    return result


def validate_graphs(base, args, ids):
    rows = []
    for batch in (2, 4):
        prompts = [ids[:max(2, len(ids) - i * 3)] for i in range(batch)]
        modes = []
        for graphs in (False, True):
            runner = fresh_runner(base, args)
            runtime = create_runtime(runner, args, chunked=True, graphs=graphs, maximum_prompt=max(map(len, prompts)))
            with counted_prefill(base.model) as chunks:
                requests, steps = execute(runtime, prompts, args.max_new_tokens)
            require(sum(row["tokens"] for row in chunks) == sum(map(len, prompts)), "batched prompt token accounting differs")
            captured = [] if runtime.graph_manager is None else [
                {"batch_size": shape.batch_size, "max_blocks": shape.max_blocks, "captured": bucket.captured}
                for shape, bucket in runtime.graph_manager.buckets.items()]
            if graphs:
                require(captured and all(row["captured"] for row in captured), "graph mode did not capture real decode buckets")
            modes.append({"mode": "graph" if graphs else "eager", "steps": steps,
                          "output_token_ids": [request.output_token_ids for request in requests],
                          "computed_prompt_tokens": sum(row["tokens"] for row in chunks),
                          "captured_buckets": captured, "allocator": cleanup(runtime)})
            del runtime, runner
            gc.collect()
        require(modes[0]["output_token_ids"] == modes[1]["output_token_ids"], f"batch={batch} eager/graph outputs differ")
        rows.append({"batch_size": batch, "prompt_lengths": list(map(len, prompts)), "output_match": True, "modes": modes})
    return {"passed": True, "comparisons": rows}


def validate_http(base, args):
    from mini_engine.server import ChatHTTPServer, InferenceService

    runner = fresh_runner(base, args)
    service = InferenceService(runner, model_name="validation-qwen3", request_timeout=180,
                               max_new_tokens=args.max_new_tokens, use_cuda_graph=False,
                               enable_prefix_cache=True, prefill_chunk_size=args.chunk_size,
                               max_num_batched_tokens=args.token_budget).start()
    server = ChatHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, name="validation-http", daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"

    def request(stream):
        payload = {"model": "validation-qwen3", "messages": [{"role": "user", "content": "Explain what a KV cache stores in one sentence."}],
                   "max_tokens": args.max_new_tokens, "temperature": 0, "stream": stream}
        if stream:
            payload["stream_options"] = {"include_usage": True}
        wire = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(wire, timeout=240) as response:
            require(response.status == 200, "HTTP validation request was unsuccessful")
            if not stream:
                result = json.load(response)
                usage = result["usage"]
                text = result["choices"][0]["message"]["content"]
                finish_reason = result["choices"][0]["finish_reason"]
            else:
                events, done, text, usage, finish_reason = [], False, "", None, None
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    value = line[6:]
                    if value == "[DONE]":
                        done = True
                        break
                    event = json.loads(value)
                    require("error" not in event, "SSE returned an error event")
                    events.append(event)
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        text += choice.get("delta", {}).get("content", "")
                        finish_reason = choice.get("finish_reason") or finish_reason
                require(done and events, "SSE omitted its final DONE marker or content events")
                require(usage is not None, "SSE include_usage omitted token accounting")
        require(usage["completion_tokens"] == args.max_new_tokens, "HTTP completion token count differs from emitted fixed-length output")
        require(usage["prompt_tokens"] > 0 and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "invalid HTTP usage accounting")
        require(isinstance(text, str) and finish_reason == "length", "HTTP output lacks text or a normal length finish")
        return {"stream": stream, "usage": usage, "text": text, "finish_reason": finish_reason}

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(request, stream) for stream in (False, True)]
            outputs = [future.result() for future in futures]
    finally:
        server.shutdown()
        service.close(timeout=30)
        server.server_close()
        thread.join(10)
    require(not thread.is_alive() and not service.healthy, "HTTP worker/server did not shut down")
    require(runner.block_manager.used_blocks == 0, "HTTP service leaked request/cache/scratch page references")
    return {"passed": True, "requests": outputs, "used_blocks_after_service_close": 0,
            "timing_scope": "correctness-only; no HTTP TTFT or engine throughput measured"}


def main():
    args = parser().parse_args()
    if min(args.block_size, args.num_blocks, args.prompt_length, args.chunk_size, args.token_budget, args.max_new_tokens) <= 0:
        raise ValueError("page, chunk, budget and token counts must be positive")
    import torch
    from mini_engine.config import QuantConfig
    from mini_engine.model import NativeQwen3Runner
    from mini_engine.runtime import configure_local_cache

    configure_local_cache()
    payload = {"passed": False, "kind": "full_model_system_correctness", "configuration": vars(args).copy(),
               "precision": args.dtype, "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "fixed_length_ignore_eos": True,
               "timing_scope": "No performance measurements; do not compare this run to service latency benchmarks", "checks": {}}
    payload["configuration"]["output"] = str(args.output)
    try:
        base = NativeQwen3Runner.from_pretrained(
            args.model, device=args.device, dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
            prefill_backend=args.prefill_backend, decode_backend=args.decode_backend,
            elementwise_backend=args.elementwise_backend, data_backend=args.data_backend, linear_backend=args.linear_backend,
            sampling_backend=args.sampling_backend, num_blocks=args.num_blocks, block_size=args.block_size,
            local_files_only=True, fuse_projections=args.fuse_projections,
            quantization=QuantConfig("int4", activation_dtype=args.dtype, group_size=args.quant_group_size) if args.quantization == "int4" else None,
            quant_backend=args.quant_backend,
        )
        payload["backend_report_start"] = base.model.backend_report()
        source = base.tokenizer.encode(args.prompt, add_special_tokens=False)
        require(source, "validation prompt tokenized to an empty sequence")
        length = max(args.block_size, args.prompt_length // args.block_size * args.block_size)
        ids = (source * ((length + len(source) - 1) // len(source)))[:length]
        suffix_source = base.tokenizer.encode(" Add one specific example.", add_special_tokens=False)
        suffix = suffix_source[:max(2, min(5, args.block_size - 1))]
        require(suffix, "suffix tokenized to an empty sequence")
        payload["prompt_token_ids"], payload["suffix_token_ids"] = ids, suffix
        payload["checks"]["chunked_prefill"] = validate_chunks(base, args, ids)
        payload["checks"]["prefix_cache"] = validate_prefix(base, args, ids, suffix)
        if args.skip_graphs:
            payload["checks"]["graphs"] = {"skipped": True, "reason": "--skip-graphs"}
        else:
            require(str(args.device).startswith("cuda"), "graph validation requires CUDA; use --skip-graphs for CPU")
            payload["checks"]["graphs"] = validate_graphs(base, args, ids)
        if args.http_smoke:
            payload["checks"]["http_sse"] = validate_http(base, args)
        payload["backend_report_end"] = base.model.backend_report()
        require(payload["backend_report_start"] == payload["backend_report_end"], "backend/library fingerprint changed during validation")
        payload["passed"] = True
    except Exception as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"passed": payload["passed"], "output": str(args.output), "checks": list(payload["checks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
