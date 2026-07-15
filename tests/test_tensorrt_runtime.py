import json
import subprocess
import sys

import pytest

from rgb_lidar_fusion.tensorrt_runtime import (
    TensorRTEnvironment,
    benchmark_schema,
    latency_summary_ms,
    runtime_diagnostic,
    require_tensorrt_available,
    validate_engine_output_path,
)


def test_runtime_diagnostic_is_structured_and_actionable_on_cpu_only_host():
    env = TensorRTEnvironment(
        trtexec_path=None,
        python_bindings_available=False,
        nvidia_smi_available=False,
        nvcc_available=False,
    )

    diagnostic = runtime_diagnostic(env)

    assert diagnostic["status"] == "missing_tensorrt_runtime"
    assert diagnostic["build_ready"] is False
    assert diagnostic["bindings"] == {"python_tensorrt": {"available": False, "version": None}}
    assert diagnostic["tools"]["trtexec"] == {"available": False, "path": None, "version": None}
    assert diagnostic["cuda"] == {
        "gpu_detected": False,
        "nvidia_smi_available": False,
        "nvidia_smi_version": None,
        "cuda_driver_version": None,
        "nvcc_available": False,
        "nvcc_version": None,
        "gpus": [],
    }
    assert diagnostic["recommended_actions"] == [
        "Run this project on a host/container with an NVIDIA GPU and matching driver.",
        "Install TensorRT runtime tools (`trtexec`) or Python bindings in the TensorRT target environment.",
        "Keep CI CPU-safe: use this diagnostic as a readiness report, not as a hard dependency.",
    ]


def test_runtime_diagnostic_reports_versions_and_gpu_inventory_when_tools_exist():
    env = TensorRTEnvironment(
        trtexec_path="/opt/tensorrt/bin/trtexec",
        python_bindings_available=True,
        nvidia_smi_available=True,
        nvcc_available=True,
    )

    def fake_probe(command: list[str]) -> str:
        if command == ["/opt/tensorrt/bin/trtexec", "--version"]:
            return "TensorRT v10.1.0\n"
        if command == ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"]:
            return "NVIDIA RTX 4090, 550.54.14, 12.4\n"
        if command == ["nvcc", "--version"]:
            return "Cuda compilation tools, release 12.4, V12.4.131\n"
        raise AssertionError(f"unexpected probe: {command}")

    diagnostic = runtime_diagnostic(env, command_probe=fake_probe, python_tensorrt_version="10.1.0")

    assert diagnostic["status"] == "ready"
    assert diagnostic["bindings"]["python_tensorrt"] == {"available": True, "version": "10.1.0"}
    assert diagnostic["tools"]["trtexec"] == {
        "available": True,
        "path": "/opt/tensorrt/bin/trtexec",
        "version": "TensorRT v10.1.0",
    }
    assert diagnostic["cuda"] == {
        "gpu_detected": True,
        "nvidia_smi_available": True,
        "nvidia_smi_version": None,
        "cuda_driver_version": "12.4",
        "nvcc_available": True,
        "nvcc_version": "Cuda compilation tools, release 12.4, V12.4.131",
        "gpus": [{"name": "NVIDIA RTX 4090", "driver_version": "550.54.14", "cuda_version": "12.4"}],
    }
    assert diagnostic["recommended_actions"] == [
        "Build the FP16 engine in this environment or equivalent NVIDIA container once ONNX export is available."
    ]


def test_tensorrt_benchmark_detect_cli_prints_structured_runtime_diagnostic():
    result = subprocess.run(
        [sys.executable, "scripts/tensorrt_benchmark.py", "--detect"],
        check=True,
        text=True,
        capture_output=True,
    )

    diagnostic = json.loads(result.stdout)

    assert "status" in diagnostic
    assert "bindings" in diagnostic
    assert "cuda" in diagnostic
    assert "tools" in diagnostic
    assert "recommended_actions" in diagnostic


def test_tensorrt_guard_fails_clearly_when_runtime_missing():
    env = TensorRTEnvironment(
        trtexec_path=None,
        python_bindings_available=False,
        nvidia_smi_available=False,
        nvcc_available=False,
    )

    with pytest.raises(RuntimeError, match="TensorRT runtime is unavailable"):
        require_tensorrt_available(env)


def test_tensorrt_guard_accepts_trtexec_or_python_bindings():
    assert require_tensorrt_available(
        TensorRTEnvironment(
            trtexec_path="/usr/bin/trtexec",
            python_bindings_available=False,
            nvidia_smi_available=True,
            nvcc_available=False,
        )
    ).is_build_ready
    assert require_tensorrt_available(
        TensorRTEnvironment(
            trtexec_path=None,
            python_bindings_available=True,
            nvidia_smi_available=True,
            nvcc_available=False,
        )
    ).is_build_ready


def test_engine_output_path_requires_engine_or_plan_extension():
    assert validate_engine_output_path("results/model_fp16.engine").suffix == ".engine"
    assert validate_engine_output_path("results/model_fp16.plan").suffix == ".plan"
    with pytest.raises(ValueError, match=".engine or .plan"):
        validate_engine_output_path("results/model_fp16.onnx")


def test_benchmark_schema_names_tail_latency_and_fps_metrics():
    schema = benchmark_schema()

    assert schema["precision"] == "string: e.g. fp16"
    assert "p50" in schema["latency_ms"]
    assert "p95" in schema["latency_ms"]
    assert schema["fps"].startswith("float")


def test_latency_summary_reports_p50_p95_and_fps_without_extra_dependencies():
    summary = latency_summary_ms([4.0, 1.0, 3.0, 2.0, 10.0], batch_size=2)

    assert summary["latency_ms"] == {
        "p50": 3.0,
        "p95": 10.0,
        "mean": 4.0,
        "min": 1.0,
        "max": 10.0,
    }
    assert summary["fps"] == 500.0


@pytest.mark.parametrize(
    ("latencies_ms", "batch_size", "message"),
    [([], 1, "at least one latency"), ([1.0], 0, "batch_size")],
)
def test_latency_summary_rejects_invalid_measurements(latencies_ms, batch_size, message):
    with pytest.raises(ValueError, match=message):
        latency_summary_ms(latencies_ms, batch_size=batch_size)
