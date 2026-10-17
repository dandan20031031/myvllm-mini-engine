"""Keep generated files on the project's drive (Windows: D: for this checkout)."""

import os
from pathlib import Path
import tempfile


def configure_local_cache() -> Path:
    root = Path(__file__).resolve().parents[1]
    directories = {
        "TEMP": root / "build" / "tmp",
        "TMP": root / "build" / "tmp",
        "TMPDIR": root / "build" / "tmp",
        "CUDA_CACHE_PATH": root / ".cache" / "cuda",
        "TORCH_EXTENSIONS_DIR": root / "build" / "torch_extensions",
        "TORCH_HOME": root / ".cache" / "torch",
        "HF_HOME": root / ".cache" / "huggingface",
        "HF_HUB_CACHE": root / ".cache" / "huggingface" / "hub",
        "XDG_CACHE_HOME": root / ".cache",
        "PIP_CACHE_DIR": root / ".cache" / "pip",
        "HF_XET_CACHE": root / ".cache" / "huggingface" / "xet",
        "TORCHINDUCTOR_CACHE_DIR": root / ".cache" / "inductor",
    }
    for name, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    # tempfile caches the first chosen directory, possibly before this call.
    tempfile.tempdir = str(directories["TEMP"])
    return root
