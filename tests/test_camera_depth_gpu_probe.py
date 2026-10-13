import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from rgb_lidar_fusion import camera_depth_gpu_probe as probe
from rgb_lidar_fusion.training import save_checkpoint


@pytest.fixture(autouse=True)
def restore_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    torch.set_num_threads(threads)


def test_cpu_probe_runs_bounded_steps_without_changing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "nvidia_snapshot", lambda: {"test": True})
    torch.set_num_threads(1)
    model = probe.CameraDepthModel()
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path=checkpoint, model=model, optimizer=optimizer, epoch=1, step=410, config={},
    )
    before = checkpoint.read_bytes()
    events = []
    probe.run_camera_depth_probe(
        emit=events.append, steps=2, height=9, width=13, device="cpu",
        checkpoint=checkpoint,
    )
    assert checkpoint.read_bytes() == before
    assert [event["step"] for event in events if event["event"] == "step_completed"] == [1, 2]
    assert events[-1]["event"] == "completed"
    assert next(event for event in events if event["event"] == "checkpoint_loaded")["saved_step"] == 410


def test_probe_records_failed_phase_and_does_not_retry(monkeypatch):
    monkeypatch.setattr(probe, "nvidia_snapshot", lambda: {"test": True})
    calls = []

    def fail_forward(self, rgb, lidar_maps):
        calls.append(True)
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(probe.CameraDepthModel, "forward", fail_forward)
    events = []
    with pytest.raises(RuntimeError, match="simulated backend"):
        probe.run_camera_depth_probe(emit=events.append, height=9, width=13, device="cpu")
    assert len(calls) == 1
    assert events[-1]["event"] == "failed"
    assert events[-1]["phase"] == "forward"
    assert not any(event["event"] == "completed" for event in events)


def test_unhealthy_gpu_stops_probe_before_model_setup(monkeypatch):
    monkeypatch.setattr(probe, "nvidia_snapshot", lambda: {})
    from rgb_lidar_fusion.camera_depth_analysis import GpuPreflightReport

    report = GpuPreflightReport(
        ready=False, torch_version="test", cuda_available=True,
        device_name="test", driver_version="test", memory_total_mib=4096,
        memory_used_mib=1000, memory_free_mib=3096, utilization_percent=98,
        recovery_action="Reset", other_compute_processes=(), smoke_test_passed=False,
        issues=("GPU needs reset",),
    )
    monkeypatch.setattr(probe, "run_gpu_preflight", lambda **kwargs: report)
    events = []
    with pytest.raises(RuntimeError, match="GPU preflight failed"):
        probe.run_camera_depth_probe(emit=events.append)
    assert events[-1]["phase"] == "preflight"
    assert not any(event.get("phase") == "model_setup" for event in events)


def test_backward_failure_is_logged_before_any_optimizer_step(monkeypatch):
    monkeypatch.setattr(probe, "nvidia_snapshot", lambda: {})

    def fail_backward(self, *args, **kwargs):
        raise RuntimeError("simulated backward failure")

    monkeypatch.setattr(torch.Tensor, "backward", fail_backward)
    events = []
    with pytest.raises(RuntimeError, match="simulated backward"):
        probe.run_camera_depth_probe(emit=events.append, height=9, width=13, device="cpu")
    assert events[-1]["phase"] == "backward"
    assert not any(event.get("phase") == "optimizer" for event in events)
    assert not any(event["event"] == "step_completed" for event in events)


@pytest.mark.parametrize("name", ["steps", "height", "width", "batch_size"])
def test_probe_rejects_non_positive_dimensions(name):
    with pytest.raises(ValueError, match=name):
        probe.run_camera_depth_probe(emit=lambda event: None, **{name: 0})


def test_probe_cli_writes_incremental_log_and_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "probe.jsonl"
    command = [
        sys.executable, str(root / "scripts/probe_camera_depth_gpu.py"),
        "--device", "cpu", "--height", "9", "--width", "13", "--steps", "1",
        "--disable-cudnn", "--output", str(output),
    ]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"}
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert events[-1]["event"] == "completed"
    assert events[0]["cudnn_enabled"] is False
    assert events[0]["cuda_launch_blocking"] == "1"
    before = output.read_bytes()
    repeated = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0
    assert output.read_bytes() == before
