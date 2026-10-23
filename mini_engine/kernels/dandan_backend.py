"""Optional, version-checked operator dependency. Importing the engine stays cheap."""

from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import torch


DANDAN_LINEAR_BACKENDS = {"dandan", "dandan_cuda"}
LINEAR_BACKENDS = {"torch", "cuda", *DANDAN_LINEAR_BACKENDS}


@lru_cache(maxsize=1)
def require_dandan_ops():
    try:
        ops = importlib.import_module("dandan_ops")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "dandan_ops is optional; install its prebuilt extension for this "
            "Python/PyTorch/CUDA environment before selecting a dandan backend"
        ) from exc
    if not callable(getattr(ops, "gemm", None)) or not hasattr(ops, "QuantizedLinear"):
        raise ImportError("incompatible dandan_ops API: gemm and QuantizedLinear are required")
    try:
        version = torch.ops.dandan_ops.w4a16_contract_version()
    except (AttributeError, RuntimeError) as exc:
        raise ImportError("dandan_ops CUDA extension is missing or stale; rebuild it for this environment") from exc
    if version != 2:
        raise ImportError(f"unsupported dandan_ops contract {version}; expected version 2")
    return ops


def validate_dandan_runtime(device, dtype):
    if torch.device(device).type != "cuda" or dtype != torch.float16:
        raise ValueError("dandan backend requires a CUDA device and FP16 weights/activations")
    if not torch.cuda.is_available():
        raise ValueError("dandan backend requires an available CUDA device")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("dandan backend requires SM80 or newer")
    return require_dandan_ops()


def require_dandan_capability(capability):
    ops = require_dandan_ops()
    names = {"prefill": ("flash_attention",), "int4": ("pack_int4", "QuantizedLinear")}
    for name in names[capability]:
        if not callable(getattr(ops, name, None)):
            raise ImportError(f"dandan_ops is missing {name}; install a compatible prebuilt library")
    if capability == "prefill" and not hasattr(torch.ops.dandan_ops, "flash_inference"):
        raise ImportError("dandan_ops binary is missing flash_inference; rebuild the extension")
    return ops


def dependency_report():
    """Fingerprint local editable builds too; called outside measured decode."""
    ops = require_dandan_ops()
    root = Path(ops.__file__).resolve().parent
    files = sorted(root.rglob("*.py")) + sorted(root.rglob("*.json")) + sorted(root.glob("_C*.pyd")) + sorted(root.glob("_C*.so"))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"module": str(root), "contract_version": 2, "sha256": digest.hexdigest(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda}
