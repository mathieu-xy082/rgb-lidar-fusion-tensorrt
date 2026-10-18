from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import rgb_lidar_fusion.camera_depth_analysis as camera_depth_analysis
from rgb_lidar_fusion.camera_depth_analysis import (
    CameraDepthSampleAnalysis,
    GpuPreflightReport,
    analyze_camera_depth_sample,
    has_camera_depth_validation_metrics,
    load_camera_depth_checkpoint,
    load_camera_depth_metrics,
    plot_camera_depth_metrics,
    plot_camera_depth_ablation,
    plot_camera_depth_sample,
    require_gpu_ready,
    run_gpu_preflight,
)
from rgb_lidar_fusion.camera_depth_model import CameraDepthModel
from rgb_lidar_fusion.lidar_splatting import SplattingConfig


def _dataset_item() -> dict:
    image = np.zeros((3, 9, 13), dtype=np.float32)
    image[0, 2:7, 3:10] = 0.4
    sparse = np.zeros((6, 9, 13), dtype=np.float32)
    for offset, (y, x) in enumerate(((1, 1), (2, 5), (4, 8), (6, 3), (7, 11))):
        sparse[0, y, x] = 0.1 * (offset + 1)
        sparse[1, y, x] = 0.01 * offset
        sparse[2, y, x] = -0.01 * offset
        sparse[3, y, x] = 0.02 * offset
        sparse[4, y, x] = 0.8
        sparse[5, y, x] = 1.0
    return {
        "image": image,
        "lidar_maps": sparse,
        "target": {},
        "meta": {"sample_id": "analysis-sample"},
    }


def _metric_row(epoch: int) -> dict:
    return {
        "epoch": epoch,
        "step": epoch,
        "loss": 0.2 / epoch,
        "grad_norm": 0.1,
        "val_loss": 0.3 / epoch,
        "val_mae_m": 4.0,
        "val_rmse_m": 5.0,
        "val_splat_mae_m": 6.0,
        "val_splat_rmse_m": 7.0,
        "val_splat_coverage": 0.5,
        "val_pixel_count": 10,
    }


def test_load_camera_depth_metrics_sorts_rows_and_detects_validation_schema(
    tmp_path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps([_metric_row(2), _metric_row(1)])
    )

    metrics = load_camera_depth_metrics(run_dir)

    assert [row["epoch"] for row in metrics] == [1, 2]
    assert has_camera_depth_validation_metrics(metrics)
    assert not has_camera_depth_validation_metrics(
        [{"epoch": 1, "loss": 0.1, "grad_norm": 0.2}]
    )


def test_load_camera_depth_checkpoint_restores_model_and_metadata(tmp_path) -> None:
    model = CameraDepthModel()
    checkpoint_path = tmp_path / "latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"max_depth_m": 80.0},
            "epoch": 3,
            "step": 12,
        },
        checkpoint_path,
    )

    loaded = load_camera_depth_checkpoint(checkpoint_path, device="cpu")

    assert loaded.epoch == 3
    assert loaded.step == 12
    assert loaded.config["max_depth_m"] == 80.0
    assert str(loaded.device) == "cpu"
    assert not loaded.model.training


def test_load_camera_depth_checkpoint_restores_sparse_input_mode(tmp_path) -> None:
    model = CameraDepthModel(lidar_mode="sparse")
    checkpoint_path = tmp_path / "sparse.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"lidar_mode": "sparse"},
            "epoch": 1,
            "step": 2,
        },
        checkpoint_path,
    )

    loaded = load_camera_depth_checkpoint(checkpoint_path, device="cpu")

    assert loaded.model.lidar_mode == "sparse"
    assert loaded.model.lidar_channels == 6


def test_analyze_camera_depth_sample_returns_aligned_metric_maps() -> None:
    torch.manual_seed(17)
    model = CameraDepthModel()

    analysis = analyze_camera_depth_sample(
        model=model,
        dataset=[_dataset_item()],
        dataset_index=0,
        holdout_fraction=0.4,
        holdout_seed=101,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
        device=torch.device("cpu"),
        max_depth_m=80.0,
    )

    assert analysis.sample_id == "analysis-sample"
    assert analysis.rgb.shape == (9, 13, 3)
    assert analysis.prediction_m.shape == (9, 13)
    assert analysis.splatted_depth_m.shape == (9, 13)
    assert analysis.splat_confidence.shape == (9, 13)
    assert int(analysis.holdout_mask.sum()) == 2
    assert np.isfinite(analysis.prediction_m).all()
    assert np.isfinite(analysis.holdout_target_m[analysis.holdout_mask]).all()
    assert np.isnan(analysis.holdout_target_m[~analysis.holdout_mask]).all()
    assert np.isfinite(
        analysis.holdout_absolute_error_m[analysis.holdout_mask]
    ).all()


def test_camera_depth_plot_helpers_return_figures() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_figure = plot_camera_depth_metrics([_metric_row(1), _metric_row(2)])
    ablation_figure = plot_camera_depth_ablation(
        {
            "sparse": [_metric_row(1), _metric_row(2)],
            "enriched": [_metric_row(1), _metric_row(2)],
        }
    )
    mask = np.zeros((4, 6), dtype=bool)
    mask[1, 2] = True
    target = np.full((4, 6), np.nan, dtype=np.float32)
    target[mask] = 12.0
    analysis = CameraDepthSampleAnalysis(
        sample_id="plot-sample",
        dataset_index=0,
        rgb=np.zeros((4, 6, 3), dtype=np.float32),
        prediction_m=np.full((4, 6), 10.0, dtype=np.float32),
        splatted_depth_m=np.full((4, 6), 11.0, dtype=np.float32),
        splat_confidence=np.ones((4, 6), dtype=np.float32),
        holdout_target_m=target,
        holdout_absolute_error_m=np.where(mask, 2.0, np.nan).astype(np.float32),
        holdout_mask=mask,
    )
    sample_figure = plot_camera_depth_sample(analysis)

    assert len(metric_figure.axes) == 2
    assert len(ablation_figure.axes) == 2
    assert len(sample_figure.axes) == 11
    plt.close(metric_figure)
    plt.close(ablation_figure)
    plt.close(sample_figure)


def test_gpu_preflight_accepts_healthy_idle_gpu_without_running_cuda(
    monkeypatch,
) -> None:
    def fake_nvidia_smi(*arguments: str) -> str:
        if arguments[0].startswith("--query-gpu="):
            return "RTX Test, 575.64.03, 4096, 700, 3396, 2\n"
        if arguments[0].startswith("--query-compute-apps="):
            return ""
        assert arguments == ("-q",)
        return "GPU Recovery Action : None\n"

    monkeypatch.setattr(camera_depth_analysis, "_run_nvidia_smi", fake_nvidia_smi)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    report = run_gpu_preflight(
        minimum_free_memory_mib=2500,
        run_smoke_test=False,
    )

    assert report.ready
    assert report.device_name == "RTX Test"
    assert report.memory_free_mib == 3396
    assert report.recovery_action == "None"
    assert report.issues == ()


def test_gpu_preflight_guard_reports_reset_memory_and_process_issues() -> None:
    report = GpuPreflightReport(
        ready=False,
        torch_version="2.13.0+cu129",
        cuda_available=False,
        device_name="RTX Test",
        driver_version="575.64.03",
        memory_total_mib=4096,
        memory_used_mib=3000,
        memory_free_mib=900,
        utilization_percent=94,
        recovery_action="Reset",
        other_compute_processes=(),
        smoke_test_passed=False,
        issues=("NVIDIA reports recovery action 'Reset'.",),
    )

    with pytest.raises(RuntimeError, match="GPU preflight failed") as error:
        require_gpu_ready(report)

    assert "recovery_action=Reset" in str(error.value)


def test_gpu_preflight_releases_cache_before_checking_free_memory(monkeypatch) -> None:
    events = []
    reserved = 1600 * 2**20

    def fake_nvidia_smi(*arguments: str) -> str:
        if arguments[0].startswith("--query-gpu="):
            events.append("memory_query")
            free = 1100 if reserved else 2700
            return f"RTX Test, 575.64.03, 4096, {4096 - free}, {free}, 2\n"
        if arguments[0].startswith("--query-compute-apps="):
            return ""
        return "GPU Recovery Action : None\n"

    def empty_cache() -> None:
        nonlocal reserved
        events.append("empty_cache")
        reserved = 0

    monkeypatch.setattr(camera_depth_analysis, "_run_nvidia_smi", fake_nvidia_smi)
    monkeypatch.setattr(camera_depth_analysis.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: reserved)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    report = run_gpu_preflight(run_smoke_test=False, release_cached_memory=True)

    assert report.ready
    assert events == ["memory_query", "gc", "empty_cache", "memory_query"]
    assert report.memory_free_mib == 2700
    assert report.torch_memory_allocated_mib == 0.0
    assert report.torch_memory_reserved_mib == 0.0
    assert report.torch_cache_released_mib == 1600.0
    assert "released_cache=1600.0 MiB" in report.format()


def test_gpu_preflight_keeps_guard_when_live_allocations_remain(monkeypatch) -> None:
    def fake_nvidia_smi(*arguments: str) -> str:
        if arguments[0].startswith("--query-gpu="):
            return "RTX Test, 575.64.03, 4096, 2996, 1100, 2\n"
        if arguments[0].startswith("--query-compute-apps="):
            return ""
        return "GPU Recovery Action : None\n"

    monkeypatch.setattr(camera_depth_analysis, "_run_nvidia_smi", fake_nvidia_smi)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1500 * 2**20)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 1600 * 2**20)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    report = run_gpu_preflight(run_smoke_test=False, release_cached_memory=True)

    assert not report.ready
    assert report.torch_memory_allocated_mib == 1500.0
    assert report.torch_cache_released_mib == 0.0
    assert any("at least 2500 MiB" in issue for issue in report.issues)


@pytest.mark.parametrize("recovery_action", ["Reset", "None"])
def test_gpu_preflight_skips_cache_operations_when_reset_or_uninitialized(
    monkeypatch, recovery_action,
) -> None:
    def fake_nvidia_smi(*arguments: str) -> str:
        if arguments[0].startswith("--query-gpu="):
            return "RTX Test, 575.64.03, 4096, 700, 3396, 2\n"
        if arguments[0].startswith("--query-compute-apps="):
            return ""
        return f"GPU Recovery Action : {recovery_action}\n"

    def unexpected_cuda_call(*args, **kwargs):
        pytest.fail("CUDA memory operations must be skipped")

    monkeypatch.setattr(camera_depth_analysis, "_run_nvidia_smi", fake_nvidia_smi)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: recovery_action == "Reset")
    monkeypatch.setattr(torch.cuda, "empty_cache", unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "memory_reserved", unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "memory_allocated", unexpected_cuda_call)

    report = run_gpu_preflight(run_smoke_test=False, release_cached_memory=True)

    assert report.ready == (recovery_action == "None")
    assert report.torch_cache_released_mib is None


def test_gpu_preflight_stops_after_cache_cleanup_failure(monkeypatch) -> None:
    def fake_nvidia_smi(*arguments: str) -> str:
        if arguments[0].startswith("--query-gpu="):
            return "RTX Test, 575.64.03, 4096, 700, 3396, 2\n"
        if arguments[0].startswith("--query-compute-apps="):
            return ""
        return "GPU Recovery Action : None\n"

    def fail_empty_cache():
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr(camera_depth_analysis, "_run_nvidia_smi", fake_nvidia_smi)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 1600 * 2**20)
    monkeypatch.setattr(torch.cuda, "empty_cache", fail_empty_cache)
    monkeypatch.setattr(torch, "arange", lambda *args, **kwargs: pytest.fail("no CUDA retry"))

    report = run_gpu_preflight(release_cached_memory=True)

    assert not report.ready
    assert any("simulated CUDA failure" in issue for issue in report.issues)
