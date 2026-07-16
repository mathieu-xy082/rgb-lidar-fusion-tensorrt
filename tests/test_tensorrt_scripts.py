import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_benchmark_schema_cli_outputs_metric_contract():
    result = run_script("tensorrt_benchmark.py", "--schema")

    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    assert schema["latency_ms"]["p50"] == "float"
    assert schema["latency_ms"]["p95"] == "float"
    assert schema["fps"].startswith("float")


def test_benchmark_detect_cli_is_safe_without_tensorrt():
    result = run_script("tensorrt_benchmark.py", "--detect")

    assert result.returncode == 0, result.stderr
    detected = json.loads(result.stdout)
    assert set(detected) == {
        "bindings",
        "build_ready",
        "cuda",
        "recommended_actions",
        "status",
        "tools",
    }
    assert set(detected["bindings"]) == {"python_tensorrt"}
    assert set(detected["cuda"]) == {
        "cuda_driver_version",
        "gpu_detected",
        "gpus",
        "nvcc_available",
        "nvcc_version",
        "nvidia_smi_available",
        "nvidia_smi_version",
    }
    assert set(detected["tools"]) == {"trtexec"}
    assert detected["recommended_actions"]


def test_benchmark_cli_requires_engine_for_execution():
    result = run_script("tensorrt_benchmark.py")

    assert result.returncode == 2
    assert "--engine is required" in result.stderr


def test_build_engine_cli_rejects_non_engine_artifacts_before_runtime_work():
    result = run_script(
        "tensorrt_build_engine.py",
        "--onnx",
        "results/model.onnx",
        "--engine",
        "results/model.onnx",
    )

    assert result.returncode == 2
    assert ".engine or .plan" in result.stderr


def test_infer_cli_rejects_non_engine_artifacts_before_runtime_work():
    result = run_script("tensorrt_infer.py", "--engine", "results/model.onnx")

    assert result.returncode == 2
    assert ".engine or .plan" in result.stderr
