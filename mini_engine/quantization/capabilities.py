"""Report project implementations separately from GPU/dtype capability candidates."""

import argparse
import json
from pathlib import Path
import platform

import torch


def precision_capabilities(device="cpu", *, probe=False):
    requested = torch.device(device)
    cuda = requested.type == "cuda" and torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(requested) if cuda else None
    sm = (properties.major, properties.minor) if properties else None
    formats = {}
    for name, attr in [("fp8_e4m3", "float8_e4m3fn"), ("fp8_e5m2", "float8_e5m2"),
                       ("mxfp8", None), ("nvfp4", "float4_e2m1fn_x2")]:
        dtype_present = attr is not None and hasattr(torch, attr)
        conversion, error = None, None
        if probe and name.startswith("fp8_") and dtype_present and (requested.type == "cpu" or cuda):
            try:
                sample = torch.tensor([0.0, 0.5, -1.0], device=requested)
                restored = sample.to(getattr(torch, attr)).float()
                conversion = bool(torch.isfinite(restored).all() and torch.equal(sample, restored))
            except (RuntimeError, NotImplementedError, TypeError) as exc:
                conversion, error = False, str(exc)
        reference = name.startswith("fp8_")
        formats[name] = {
            "dtype_present": dtype_present,
            "reference_backend_implemented": reference and dtype_present,
            "native_linear_integration": reference and dtype_present,
            "accelerated_quant_gemm_implemented": False,
            "hardware_candidate_only": bool(sm and sm >= ((8, 9) if reference else (10, 0))),
            "tensor_conversion_verified": conversion,
            "probe_error": error,
            "status": "reference_only" if reference and dtype_present else "reserved",
        }
    for name in ("int8", "int4"):
        formats[name] = {
            "reference_backend_implemented": True,
            "native_linear_integration": True,
            "accelerated_quant_gemm_implemented": False,
            "status": "reference_only",
        }
    return {
        "os": platform.system(), "pytorch": torch.__version__, "pytorch_cuda": torch.version.cuda,
        "requested_device": str(requested), "cuda_available": cuda,
        "gpu": properties.name if properties else None, "compute_capability": list(sm) if sm else None,
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "formats": formats,
        "tf32": {"kind": "fp32_gemm_policy", "weight_storage_format": False,
                 "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32)},
        "note": "GPU architecture/dtype presence do not prove Windows low-precision GEMM support. "
                "FP8 reference is weight-only, not W8A8. NVFP4/MXFP8 are not implemented.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = precision_capabilities(args.device, probe=args.probe)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    print(content)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
