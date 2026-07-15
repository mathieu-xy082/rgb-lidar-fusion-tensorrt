"""Safe TensorRT runtime discovery and benchmark schema helpers.

This module deliberately avoids importing TensorRT at module import time. Milestone 5
must be able to document and test the deployment path on hosts that do not have
CUDA/TensorRT installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import importlib.util
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TensorRTEnvironment:
    """Observable TensorRT/CUDA runtime capabilities on the current host."""

    trtexec_path: str | None
    python_bindings_available: bool
    nvidia_smi_available: bool
    nvcc_available: bool

    @property
    def is_build_ready(self) -> bool:
        """Return true when an ONNX->engine build path is available."""

        return self.trtexec_path is not None or self.python_bindings_available


def detect_tensorrt_environment() -> TensorRTEnvironment:
    """Detect TensorRT/CUDA tools without failing on lightweight dev hosts."""

    return TensorRTEnvironment(
        trtexec_path=shutil.which("trtexec"),
        python_bindings_available=importlib.util.find_spec("tensorrt") is not None,
        nvidia_smi_available=shutil.which("nvidia-smi") is not None,
        nvcc_available=shutil.which("nvcc") is not None,
    )


CommandProbe = Callable[[list[str]], str]


def _default_command_probe(command: list[str]) -> str:
    result = run_command(command)
    if result.returncode != 0:
        return ""
    return result.stdout.strip() or result.stderr.strip()


def _installed_python_package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _first_non_empty_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _nvidia_smi_inventory(text: str) -> list[dict[str, str]]:
    gpus = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            gpus.append({"name": parts[0], "driver_version": parts[1], "cuda_version": parts[2]})
    return gpus


def runtime_diagnostic(
    env: TensorRTEnvironment | None = None,
    *,
    command_probe: CommandProbe = _default_command_probe,
    python_tensorrt_version: str | None = None,
) -> dict[str, Any]:
    """Return a CPU-safe structured readiness report for TensorRT/CUDA."""

    detected = env or detect_tensorrt_environment()
    trtexec_version = None
    if detected.trtexec_path is not None:
        trtexec_version = _first_non_empty_line(command_probe([detected.trtexec_path, "--version"]))

    if python_tensorrt_version is None and detected.python_bindings_available:
        python_tensorrt_version = _installed_python_package_version("tensorrt")

    nvidia_smi_output = ""
    if detected.nvidia_smi_available:
        nvidia_smi_output = command_probe(
            ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"]
        )
    gpus = _nvidia_smi_inventory(nvidia_smi_output)
    gpu_detected = bool(gpus) if detected.nvidia_smi_available else False

    nvcc_version = None
    if detected.nvcc_available:
        nvcc_version = _first_non_empty_line(command_probe(["nvcc", "--version"]))

    recommended_actions: list[str] = []
    if not gpu_detected:
        recommended_actions.append(
            "Run this project on a host/container with an NVIDIA GPU and matching driver."
        )
    if not detected.is_build_ready:
        recommended_actions.append(
            "Install TensorRT runtime tools (`trtexec`) or Python bindings in the TensorRT target environment."
        )
    recommended_actions.append(
        "Build the FP16 engine in this environment or equivalent NVIDIA container once ONNX export is available."
        if detected.is_build_ready and gpu_detected
        else "Keep CI CPU-safe: use this diagnostic as a readiness report, not as a hard dependency."
    )

    return {
        "status": "ready" if detected.is_build_ready and gpu_detected else "missing_tensorrt_runtime",
        "build_ready": detected.is_build_ready,
        "bindings": {
            "python_tensorrt": {
                "available": detected.python_bindings_available,
                "version": python_tensorrt_version,
            }
        },
        "tools": {
            "trtexec": {
                "available": detected.trtexec_path is not None,
                "path": detected.trtexec_path,
                "version": trtexec_version,
            }
        },
        "cuda": {
            "gpu_detected": gpu_detected,
            "nvidia_smi_available": detected.nvidia_smi_available,
            "nvidia_smi_version": None,
            "cuda_driver_version": gpus[0]["cuda_version"] if gpus else None,
            "nvcc_available": detected.nvcc_available,
            "nvcc_version": nvcc_version,
            "gpus": gpus,
        },
        "recommended_actions": recommended_actions,
    }


def require_tensorrt_available(env: TensorRTEnvironment | None = None) -> TensorRTEnvironment:
    """Raise a clear error when no TensorRT build/inference runtime is present."""

    detected = env or detect_tensorrt_environment()
    if not detected.is_build_ready:
        raise RuntimeError(
            "TensorRT runtime is unavailable: neither `trtexec` nor Python package "
            "`tensorrt` was found. Select a CUDA/TensorRT target first (local host "
            "or NVIDIA container), install the matching runtime there, then rerun "
            "this script with an ONNX model."
        )
    return detected


def validate_engine_output_path(path: str | Path) -> Path:
    """Validate that TensorRT engine artifacts use a non-source extension."""

    engine_path = Path(path)
    if engine_path.suffix not in {".engine", ".plan"}:
        raise ValueError("TensorRT engine output must end with .engine or .plan")
    return engine_path


def benchmark_schema() -> dict[str, Any]:
    """Return the JSON-compatible benchmark metrics schema for Milestone 5."""

    return {
        "model": "string: model or engine name",
        "precision": "string: e.g. fp16",
        "runtime": "string: trtexec or python-tensorrt",
        "device": "string: GPU name and compute capability when available",
        "batch_size": "integer",
        "input_shape": "string or list[int]",
        "warmup_runs": "integer",
        "measured_runs": "integer",
        "latency_ms": {
            "p50": "float",
            "p95": "float",
            "mean": "float",
            "min": "float",
            "max": "float",
        },
        "fps": "float: batch_size * 1000 / latency_ms.mean",
        "notes": "string: limitations, host/container, CUDA/TensorRT versions",
    }


def latency_summary_ms(latencies_ms: list[float], batch_size: int) -> dict[str, Any]:
    """Summarize measured inference latencies with the Milestone 5 metrics.

    Percentiles use the nearest-rank method so benchmark output is deterministic
    without adding a NumPy dependency to the lightweight default environment.
    """

    if not latencies_ms:
        raise ValueError("at least one latency measurement is required")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    ordered = sorted(latencies_ms)
    mean = sum(ordered) / len(ordered)

    def nearest_rank(percentile: float) -> float:
        index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
        return ordered[index]

    return {
        "latency_ms": {
            "p50": nearest_rank(50),
            "p95": nearest_rank(95),
            "mean": mean,
            "min": ordered[0],
            "max": ordered[-1],
        },
        "fps": batch_size * 1000 / mean,
    }


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command for scripts while returning captured text output."""

    return subprocess.run(command, check=False, text=True, capture_output=True)


def print_json(data: dict[str, Any]) -> None:
    """Print stable pretty JSON for diagnostics/schema output."""

    print(json.dumps(data, indent=2, sort_keys=True))
