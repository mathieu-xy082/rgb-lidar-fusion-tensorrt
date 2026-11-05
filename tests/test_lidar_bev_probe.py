import json
import inspect
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion import lidar_bev_probe as probe
from rgb_lidar_fusion.camera_bev import BEVGridConfig


def _grid():
    return BEVGridConfig((0., 1.), (-0.4, .2), (-3., 3.), .2)


def _cloud():
    return np.array([[.05, -.35, -1., .2], [.08, -.32, 1., .8], [.45, .1, 0., .5]], dtype=np.float32)


def test_cpu_probe_measures_whole_encoder_without_modifying_inputs():
    cloud = _cloud()
    before = cloud.copy()
    events = []
    result = probe.run_lidar_bev_probe(cloud, grid=_grid(), emit=events.append, device="cpu", iterations=1)
    np.testing.assert_array_equal(cloud, before)
    assert events[-1]["event"] == "completed"
    assert result["minimum_free_memory_mib"] == 2048
    assert result["output_shape"] == [1, 32, 5, 3]
    assert result["retained_points"] == 3
    for value in result["measurements"].values():
        assert value["median_ms"] > 0 and value["peak_allocated_mib"] is None


def test_gpu_guard_keeps_2048_mib_local_and_stops_before_setup(monkeypatch):
    calls, events = [], []

    def refuse(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("GPU refused")

    monkeypatch.setattr(probe, "run_gpu_preflight", refuse)
    monkeypatch.setattr(probe, "build_lidar_pillars", lambda *args, **kwargs: pytest.fail("guard must precede grouping/setup"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: pytest.fail("no CUDA work after refusal"))
    with pytest.raises(RuntimeError, match="GPU refused"):
        probe.run_lidar_bev_probe(_cloud(), grid=_grid(), emit=events.append)
    assert calls == [{"minimum_free_memory_mib": 2048, "run_smoke_test": False}]
    assert events[-1]["event"] == "failed" and events[-1]["phase"] == "preflight"


def test_camera_preflight_default_is_not_lowered_by_lidar_diagnostic():
    assert inspect.signature(probe.run_gpu_preflight).parameters["minimum_free_memory_mib"].default == 2500


def test_backward_failure_is_logged_without_retry(monkeypatch):
    events, calls = [], []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("simulated backward failure")

    monkeypatch.setattr(torch.Tensor, "backward", fail)
    with pytest.raises(RuntimeError, match="simulated backward"):
        probe.run_lidar_bev_probe(_cloud(), grid=_grid(), emit=events.append, device="cpu", iterations=1)
    assert len(calls) == 1
    assert events[-1]["event"] == "failed" and events[-1]["phase"] == "backward"
    assert not any(event["event"] == "completed" for event in events)


@pytest.mark.parametrize("name", ["iterations", "warmup", "minimum_free_memory_mib"])
def test_invalid_probe_limits_are_rejected(name):
    with pytest.raises(ValueError, match="positive"):
        probe.run_lidar_bev_probe(_cloud(), grid=_grid(), emit=lambda event: None, device="cpu", **{name: 0})


def test_cli_writes_incremental_log_and_refuses_overwrite(tmp_path):
    cloud_path = tmp_path / "000000.bin"
    _cloud().tofile(cloud_path)
    output = tmp_path / "probe.jsonl"
    command = [sys.executable, "scripts/probe_lidar_bev.py", "--velodyne", str(cloud_path),
               "--device", "cpu", "--iterations", "1", "--x-range", "0", "1",
               "--y-range", "-0.4", "0.2", "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert completed.returncode == 0, completed.stderr
    events = [json.loads(line) for line in output.read_text().splitlines()]
    assert events[-1]["event"] == "completed" and events[-1]["summary"]["batch_size"] == 1
    before = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert repeated.returncode != 0 and output.read_bytes() == before
