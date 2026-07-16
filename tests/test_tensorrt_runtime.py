import pytest

from rgb_lidar_fusion.tensorrt_runtime import (
    TensorRTEnvironment,
    benchmark_schema,
    latency_summary_ms,
    require_tensorrt_available,
    validate_engine_output_path,
)


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
