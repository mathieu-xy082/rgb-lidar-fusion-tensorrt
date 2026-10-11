from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_depth_analysis import (
    CameraDepthSampleAnalysis,
    analyze_camera_depth_sample,
    has_camera_depth_validation_metrics,
    load_camera_depth_checkpoint,
    load_camera_depth_metrics,
    plot_camera_depth_metrics,
    plot_camera_depth_sample,
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
    assert len(sample_figure.axes) == 11
    plt.close(metric_figure)
    plt.close(sample_figure)
