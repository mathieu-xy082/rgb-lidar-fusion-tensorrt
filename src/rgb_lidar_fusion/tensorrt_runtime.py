"""Safe TensorRT runtime discovery and benchmark schema helpers.

This module deliberately avoids importing TensorRT at module import time. Milestone 5
must be able to document and test the deployment path on hosts that do not have
CUDA/TensorRT installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


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


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command for scripts while returning captured text output."""

    return subprocess.run(command, check=False, text=True, capture_output=True)


def print_json(data: dict[str, Any]) -> None:
    """Print stable pretty JSON for diagnostics/schema output."""

    print(json.dumps(data, indent=2, sort_keys=True))
