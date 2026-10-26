"""Shared measurement metadata and statistics; imports only the standard library.

Forward/generation timing and correctness checks remain in their entry points.
"""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_json(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def percentile(values, fraction):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def percentiles(values):
    values = [value for value in values if value is not None]
    return {"count": len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95)}


def run_metadata_command(command):
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=10, check=False)
        return {"available": result.returncode == 0, "stdout": result.stdout.strip(),
                "error": result.stderr.strip() or None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "stdout": "", "error": str(error)}


def source_metadata(*, extra_files=()):
    """Hash engine sources, this helper and explicitly supplied entry points."""
    files = {path for path in (PROJECT_ROOT / "mini_engine").rglob("*")
             if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".h", ".cuh")}
    files.add(Path(__file__).resolve())
    files.update(Path(path).resolve() for path in extra_files)
    hashes = {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(files)}
    commit = run_metadata_command(["git", "rev-parse", "HEAD"])
    status = run_metadata_command(["git", "status", "--porcelain"])
    return {
        "git_commit": commit["stdout"] if commit["available"] else None,
        "git_dirty": bool(status["stdout"]) if status["available"] else None,
        "git_status": status["stdout"] if status["available"] else None,
        "source_files_sha256": hashes, "source_tree_sha256": sha256_json(hashes),
    }


def gpu_snapshot():
    fields = ["index", "uuid", "name", "driver_version", "pstate", "utilization.gpu",
              "utilization.memory", "memory.used", "clocks.current.graphics",
              "clocks.current.memory", "power.draw", "temperature.gpu"]
    gpus = run_metadata_command(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                 "--format=csv,noheader,nounits"])
    processes = run_metadata_command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                      "--format=csv,noheader,nounits"])
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "benchmark_pid": os.getpid(),
        "gpu_query_available": gpus["available"],
        "gpus": [dict(zip(fields, row)) for row in csv.reader(io.StringIO(gpus["stdout"]), skipinitialspace=True)]
                if gpus["available"] else [],
        "gpu_query_error": gpus["error"],
        "compute_process_query_available": processes["available"],
        "compute_processes": [dict(zip(("pid", "process_name", "memory_mib"), row))
                              for row in csv.reader(io.StringIO(processes["stdout"]), skipinitialspace=True)]
                             if processes["available"] else [],
        "compute_process_query_error": processes["error"],
        "note": "Snapshots occur outside measured requests. WDDM can omit process/memory data; a snapshot cannot prove that the GPU remained idle or exclusively owned throughout the experiment.",
    }
