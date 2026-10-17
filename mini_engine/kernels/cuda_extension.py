"""Lazy Windows CUDA extension build. Importing this file never runs nvcc."""

from functools import lru_cache
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
    build = root / "build" / "torch_extensions" / "paged_attention"
    build.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent / "csrc"
    return load(
        name="miniqwen_paged_attention",
        sources=[str(source / "bindings.cpp"), str(source / "paged_attention.cu")],
        build_directory=str(build),
        extra_cflags=["/O2", "/utf-8", "/Zc:preprocessor"] if sys.platform == "win32" else ["-O3"],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-lineinfo"]
                         + (["-Xcompiler", "/Zc:preprocessor", "-Xcompiler", "/utf-8"] if sys.platform == "win32" else []),
        verbose=verbose,
    )


if __name__ == "__main__":
    extension = load_cuda_extension(verbose=True)
    print(f"CUDA extension loaded: {extension.__file__}")
