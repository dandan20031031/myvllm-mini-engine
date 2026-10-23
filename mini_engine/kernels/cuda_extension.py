"""Lazy Windows CUDA extension build. Importing this file never runs nvcc."""

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import sys
import warnings

from ..runtime import configure_local_cache


@lru_cache(maxsize=1)
def load_cuda_extension(verbose=False):
    root = configure_local_cache()
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requires a CUDA-enabled PyTorch and GPU")
    if CUDA_HOME is None:
        raise RuntimeError("CUDA Toolkit not found; set CUDA_HOME to your nvcc Toolkit")
    # A venv's Scripts directory is not necessarily on PATH without activation.
    scripts = Path(sys.executable).parent
    os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
    if shutil.which("ninja") is None:
        raise RuntimeError("Install Windows Ninja in this environment: python -m pip install ninja")
    if sys.platform == "win32" and shutil.which("cl") is None:
        raise RuntimeError("Start an x64 MSVC Developer PowerShell (cl.exe must be on PATH)")
    os.environ.setdefault("MAX_JOBS", "2")
    # Record the Toolkit/runtime mismatch explicitly; do not modify PyTorch's
    # checks or pretend the combinations have matching version numbers.
    import subprocess
    import re
    nvcc = Path(CUDA_HOME) / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release (\d+\.\d+)", version)
    if match and match.group(1) != torch.version.cuda:
        warnings.warn(f"Toolkit {match.group(1)} differs from PyTorch CUDA {torch.version.cuda}; "
                      "this combination must pass runtime tests on the target machine.", stacklevel=2)
    source = Path(__file__).parent / "csrc"
    sources = [source / name for name in ("bindings.cpp", "paged_attention.cu", "elementwise.cu", "data.cu", "prefill.cu", "sampling.cu")]
    # Windows locks loaded .pyd files. Content-addressed builds allow an old
    # chat process to keep running while a new source revision is compiled.
    # They also prevent a failed rebuild from importing a stale binary whose
    # exported operators no longer match the Python wrapper.
    fingerprint = hashlib.sha256()
    for path in sources:
        fingerprint.update(path.name.encode())
        fingerprint.update(path.read_bytes())
    fingerprint.update(str((torch.__version__, torch.version.cuda, str(CUDA_HOME),
                            version, os.environ.get("TORCH_CUDA_ARCH_LIST"),
                            torch.cuda.get_device_capability())).encode())
    build_id = fingerprint.hexdigest()[:16]
    build = root / "build" / "torch_extensions" / "paged_attention" / build_id
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"miniqwen_paged_attention_{build_id}",
        sources=[str(path) for path in sources],
        build_directory=str(build),
        extra_cflags=["/O2", "/utf-8", "/Zc:preprocessor"] if sys.platform == "win32" else ["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo"]
                         + (["-Xcompiler", "/Zc:preprocessor", "-Xcompiler", "/utf-8"] if sys.platform == "win32" else []),
        verbose=verbose,
    )


if __name__ == "__main__":
    extension = load_cuda_extension(verbose=True)
    print(f"CUDA extension loaded: {extension.__file__}")
