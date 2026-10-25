"""Diagnose precision-dependent KV differences without measuring performance.

Compare all/selected B1 logits and ragged prefill, inspect per-layer/head KV
errors, and independently decode a fixed greedy continuation. The FP32 math
oracle uses exactly the FP16/BF16-rounded weights loaded for the CUDA path.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_engine.runtime import configure_local_cache

configure_local_cache()

import torch

from mini_engine.model import NativeQwen3, NativeQwen3Runner
from mini_engine.engine.batch import PrefillBatch
from benchmarks import benchmark_ragged_prefill as bench


def stats(actual, reference, atol, rtol):
    actual, reference = actual.float(), reference.float()
    error = (actual - reference).abs()
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    mismatch = ~finite | (error > atol + rtol * reference.abs())
    max_flat = int(error.argmax())
    max_index = []
    for dimension in reversed(error.shape):
        max_index.insert(0, max_flat % dimension)
        max_flat //= dimension
    ref_rms = float(reference.square().mean().sqrt())
    return {"max_abs": float(error.max()), "max_index": max_index,
            "actual_at_max": float(actual[tuple(max_index)]),
            "reference_at_max": float(reference[tuple(max_index)]),
            "reference_max_abs": float(reference.abs().max()),
            "relative_rms": float(error.square().mean().sqrt()) / max(ref_rms, 1e-12),
            "mismatched": int(mismatch.sum()), "elements": reference.numel(),
            "fraction": float(mismatch.float().mean()),
            "actual_nonfinite": int((~torch.isfinite(actual)).sum()),
            "reference_nonfinite": int((~torch.isfinite(reference)).sum())}


def kv_comparison(actual, reference, atol, rtol):
    report = stats(actual, reference, atol, rtol)
    layers = []
    for layer in range(actual.shape[0]):
        row = stats(actual[layer], reference[layer], atol, rtol)
        error = (actual[layer].float() - reference[layer].float()).abs()
        scales = reference[layer].float().abs().amax(dim=(0, 2)).clamp_min(1e-12)
        rms = reference[layer].float().square().mean(dim=(0, 2)).sqrt().clamp_min(1e-12)
        row.update({"layer": layer,
                    "head_max_error_over_max_reference": (error.amax(dim=(0, 2)) / scales).tolist(),
                    "head_relative_rms_error": (error.square().mean(dim=(0, 2)).sqrt() / rms).tolist(),
                    "head_reference_max_abs": scales.tolist()})
        layers.append(row)
    report["layers"] = layers
    return report


def clean_json(value):
    """Keep nonfinite diagnostic counts while producing standards-compliant JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def source_fingerprint():
    result = bench.source_fingerprint()
    files = dict(result["source_files_sha256"])
    files[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["source_files_sha256"] = files
    result["source_tree_sha256"] = bench.sha256_json(files)
    return result


@torch.inference_mode()
def run_variant(runner, sequences, *, output_tokens=8, atol=0.125, rtol=0.01):
    """Own temporary pages only; the caller owns and closes the loaded runner."""
    cache, manager = runner.cache, runner.block_manager
    owners, pages = [], []
    try:
        for index, ids in enumerate(sequences):
            owner = f"__ragged_diagnostic__:{index}"
            pages.append(manager.allocate(owner, len(ids) + output_tokens))
            owners.append(owner)
        common = {"block_size": manager.block_size, "num_blocks": manager.num_blocks, "device": cache.device}
        batch = PrefillBatch.from_sequences(sequences, pages, [0] * len(sequences), **common)
        singles = [PrefillBatch.from_sequences([ids], [table], [0], **common) for ids, table in zip(sequences, pages)]
        case = {"batch": batch, "singles": singles,
                "last_rows": tuple(offset - 1 for offset in batch.query_offsets[1:]),
                "single_rows": [(len(ids) - 1,) for ids in sequences], "pages": pages,
                "output": torch.empty((len(sequences), runner.model.config.vocab_size), device=cache.device, dtype=cache.dtype)}
        table_width = max(map(len, pages))
        tables = torch.tensor([row + [-1] * (table_width - len(row)) for row in pages], device=cache.device, dtype=torch.long)
        modes = {}
        for mode in bench.MODES:
            cache.clear([page for table in pages for page in table])
            logits = bench.execute_mode(runner, case, mode)
            bench.synchronize(cache.device)
            modes[mode] = {"logits": logits.detach().float().cpu().clone(),
                           "kv": bench.cache_snapshot(cache, batch.slot_mapping)}
            pending = logits.argmax(-1)
            outputs = [[int(token)] for token in pending.tolist()]
            for step in range(output_tokens - 1):
                positions = torch.tensor([len(ids) + step for ids in sequences], device=cache.device, dtype=torch.long)
                pending = runner.model.forward_decode_batch(pending, cache, tables, positions).argmax(-1)
                for row, token in zip(outputs, pending.tolist()):
                    row.append(int(token))
            modes[mode]["generated_token_ids"] = outputs
        reference = modes[bench.MODES[0]]
        comparisons = {}
        for mode in bench.MODES[1:]:
            comparisons[mode] = {
                "logits": stats(modes[mode]["logits"], reference["logits"], atol, rtol),
                "last_argmax_match": bool(torch.equal(modes[mode]["logits"].argmax(-1), reference["logits"].argmax(-1))),
                "generated_token_ids_match": modes[mode]["generated_token_ids"] == reference["generated_token_ids"],
                "key": kv_comparison(modes[mode]["kv"][0], reference["kv"][0], atol, rtol),
                "value": kv_comparison(modes[mode]["kv"][1], reference["kv"][1], atol, rtol)}
        report = {"parameter_dtypes": sorted({str(parameter.dtype) for parameter in runner.model.parameters()}),
                  "cache_dtype": str(cache.dtype), "backends": runner.model.backend_report(),
                  "atol": atol, "rtol": rtol, "comparisons": comparisons,
                  "strict_elementwise_validation_passed": all(
                      row["logits"]["mismatched"] == row["key"]["mismatched"] == row["value"]["mismatched"] == 0
                      and row["generated_token_ids_match"] for row in comparisons.values()),
                  "greedy_outputs": {mode: value["generated_token_ids"] for mode, value in modes.items()}}
        return report, modes
    finally:
        bench.synchronize(cache.device)
        for owner in owners:
            manager.release(owner)


def load_runner(args, fused):
    return NativeQwen3Runner.from_pretrained(args.model, device=args.device,
        dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype],
        prefill_backend="cuda", decode_backend="cuda", elementwise_backend="cuda", data_backend="cuda",
        linear_backend="cuda", fuse_projections=fused, local_files_only=True,
        num_blocks=args.num_blocks, block_size=args.block_size)


def fp32_oracle(source, args):
    model = NativeQwen3(source.model.config, "math", "torch", "torch", "torch", "torch").to(
        device=args.device, dtype=torch.float32)
    model.enable_projection_fusion()
    # copy_ converts the already-rounded low-precision weights to FP32; no
    # independent checkpoint rounding or random parameters remain in the oracle.
    model.load_state_dict(source.model.state_dict(), strict=True)
    return NativeQwen3Runner(model, source.tokenizer, num_blocks=args.num_blocks,
                             block_size=args.block_size, eos_token_ids=[])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    result.add_argument("--device", default="cuda")
    result.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    result.add_argument("--prompt-lengths", nargs="+", type=int, default=[32, 24])
    result.add_argument("--prompt", default="Explain why an inference engine uses a paged KV cache. ")
    result.add_argument("--output-tokens", type=int, default=8)
    result.add_argument("--block-size", type=int, default=16)
    result.add_argument("--num-blocks", type=int, default=16)
    result.add_argument("--atol", type=float, default=0.125)
    result.add_argument("--rtol", type=float, default=0.01)
    result.add_argument("--cpu-threads", type=int, default=1)
    result.add_argument("--output", type=Path, default=ROOT / "build/validation/ragged_numeric_diagnostic.json")
    return result


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if (min(args.prompt_lengths + [args.output_tokens, args.block_size, args.num_blocks, args.cpu_threads]) <= 0
            or not math.isfinite(args.atol) or not math.isfinite(args.rtol) or min(args.atol, args.rtol) < 0):
        arg_parser.error("lengths, capacities and threads must be positive; tolerances must be finite and nonnegative")
    if torch.device(args.device).type != "cuda":
        arg_parser.error("the loaded low-precision CUDA diagnostic requires a CUDA device")
    payload = {"kind": "ragged_prefill_numerical_diagnostic", "completed": False,
               "performance_measurement": False, "configuration": {**vars(args), "output": str(args.output)},
               "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
               "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
               "variants": {},
               "scope": "Diagnostic only. Completion does not mean strict elementwise validation passed. FP32 oracle weights are exact upcasts of the loaded low-precision weights; TF32 is disabled for that oracle. Greedy continuations have a fixed output budget and ignore EOS."}
    old_threads = torch.get_num_threads()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    failure = None
    try:
        torch.set_num_threads(args.cpu_threads)
        payload["source_start"] = source_fingerprint()
        with closing(load_runner(args, True)) as fused:
            sequences = bench.prompt_tokens(fused, args.prompt_lengths, args.prompt)
            payload["prompt_token_sha256"] = bench.sha256_json(sequences)
            payload["variants"]["low_precision_fused"], fused_data = run_variant(
                fused, sequences, output_tokens=args.output_tokens, atol=args.atol, rtol=args.rtol)
            torch.backends.cuda.matmul.allow_tf32 = False
            with closing(fp32_oracle(fused, args)) as oracle:
                payload["variants"]["fp32_math_oracle_from_rounded_weights"], oracle_data = run_variant(
                    oracle, sequences, output_tokens=args.output_tokens, atol=1e-4, rtol=1e-4)
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
            payload["low_precision_vs_fp32_oracle"] = {}
            for mode in bench.MODES:
                payload["low_precision_vs_fp32_oracle"][mode] = {
                    "logits": stats(fused_data[mode]["logits"], oracle_data[mode]["logits"], args.atol, args.rtol),
                    "last_argmax_match": bool(torch.equal(fused_data[mode]["logits"].argmax(-1), oracle_data[mode]["logits"].argmax(-1))),
                    "key": kv_comparison(fused_data[mode]["kv"][0], oracle_data[mode]["kv"][0], args.atol, args.rtol),
                    "value": kv_comparison(fused_data[mode]["kv"][1], oracle_data[mode]["kv"][1], args.atol, args.rtol)}
        del oracle, fused, oracle_data, fused_data
        gc.collect()
        torch.cuda.empty_cache()
        with closing(load_runner(args, False)) as unfused:
            payload["variants"]["low_precision_unfused"], _ = run_variant(
                unfused, sequences, output_tokens=args.output_tokens, atol=args.atol, rtol=args.rtol)
        payload["completed"] = True
    except BaseException as error:
        failure = error
        payload["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.set_num_threads(old_threads)
        try:
            payload["source_end"] = source_fingerprint()
            payload["source_unchanged"] = (payload.get("source_start", {}).get("source_tree_sha256")
                                           == payload["source_end"]["source_tree_sha256"])
        except Exception as error:
            payload["source_unchanged"] = None
            payload["metadata_error"] = str(error)
            failure = failure or error
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clean_json(payload), ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"completed": payload["completed"], "output": str(args.output)}))
    if failure is not None:
        raise failure
    return payload


if __name__ == "__main__":
    main()
