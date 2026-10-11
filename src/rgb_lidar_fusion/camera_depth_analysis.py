"""Reusable metric and visualization helpers for camera-depth experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .camera_depth_model import CameraDepthModel
from .lidar_splatting import SplattingConfig
from .training import (
    DEPTH_EXPANDED_CHANNEL_INDEX,
    SPLAT_CONFIDENCE_CHANNEL_INDEX,
    prepare_camera_depth_evaluation_batch,
    select_device,
)


VALIDATION_METRIC_NAMES = (
    "val_loss",
    "val_mae_m",
    "val_rmse_m",
    "val_splat_mae_m",
    "val_splat_rmse_m",
    "val_splat_coverage",
    "val_pixel_count",
)


@dataclass(frozen=True)
class LoadedCameraDepthCheckpoint:
    """Camera-depth checkpoint ready for inference."""

    model: Any
    config: dict[str, Any]
    epoch: int
    step: int
    device: Any


@dataclass(frozen=True)
class CameraDepthSampleAnalysis:
    """Image-aligned arrays needed to inspect one validation prediction."""

    sample_id: str
    dataset_index: int
    rgb: np.ndarray
    prediction_m: np.ndarray
    splatted_depth_m: np.ndarray
    splat_confidence: np.ndarray
    holdout_target_m: np.ndarray
    holdout_absolute_error_m: np.ndarray
    holdout_mask: np.ndarray


def load_camera_depth_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Load epoch metrics from a run directory or a metrics JSON path."""

    metrics_path = Path(path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"camera-depth metrics do not exist: {metrics_path}")

    rows = json.loads(metrics_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("camera-depth metrics must contain a non-empty list.")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"metric row {index} must be a JSON object.")
        missing = {"epoch", "loss"} - row.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"metric row {index} is missing required fields: {names}.")
    return sorted(rows, key=lambda row: int(row["epoch"]))


def has_camera_depth_validation_metrics(metrics: list[dict[str, Any]]) -> bool:
    """Return whether every row contains the current validation metric schema."""

    return bool(metrics) and all(
        all(row.get(name) is not None for name in VALIDATION_METRIC_NAMES)
        for row in metrics
    )


def load_camera_depth_checkpoint(
    path: str | Path,
    *,
    device: str = "auto",
) -> LoadedCameraDepthCheckpoint:
    """Load a local training checkpoint into an inference-ready model."""

    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"camera-depth checkpoint does not exist: {checkpoint_path}"
        )
    selected_device = select_device(device)
    payload = torch.load(
        checkpoint_path,
        map_location=selected_device,
        weights_only=False,
    )
    if "model_state_dict" not in payload:
        raise ValueError("checkpoint does not contain model_state_dict.")

    model = CameraDepthModel(lidar_mode="enriched")
    model.load_state_dict(payload["model_state_dict"])
    model.to(selected_device)
    model.eval()
    return LoadedCameraDepthCheckpoint(
        model=model,
        config=dict(payload.get("config", {})),
        epoch=int(payload.get("epoch", 0)),
        step=int(payload.get("step", 0)),
        device=selected_device,
    )


def analyze_camera_depth_sample(
    *,
    model,
    dataset,
    dataset_index: int,
    holdout_fraction: float,
    holdout_seed: int,
    splatting_config: SplattingConfig,
    device,
    max_depth_m: float,
) -> CameraDepthSampleAnalysis:
    """Predict and collect image-space diagnostics for one held-out sample."""

    import torch

    if max_depth_m <= 0.0:
        raise ValueError("max_depth_m must be positive.")

    item = dataset[int(dataset_index)]
    depth_batch = prepare_camera_depth_evaluation_batch(
        dataset=[item],
        indices=[0],
        holdout_fraction=holdout_fraction,
        holdout_seed=holdout_seed + int(dataset_index),
        splatting_config=splatting_config,
    )
    rgb = torch.from_numpy(depth_batch["rgb"]).to(device)
    lidar_maps = torch.from_numpy(depth_batch["lidar_maps"]).to(device)
    was_training = model.training
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(rgb, lidar_maps)
    finally:
        model.train(was_training)
    if not torch.isfinite(prediction).all():
        raise ValueError("camera-depth sample prediction must be finite.")

    prediction_m = prediction[0, 0].detach().cpu().numpy() * max_depth_m
    target_m = depth_batch["depth_target"][0, 0] * max_depth_m
    holdout_mask = depth_batch["loss_mask"][0, 0].astype(bool, copy=True)
    holdout_target_m = np.full(target_m.shape, np.nan, dtype=np.float32)
    holdout_target_m[holdout_mask] = target_m[holdout_mask]
    holdout_absolute_error_m = np.full(target_m.shape, np.nan, dtype=np.float32)
    holdout_absolute_error_m[holdout_mask] = np.abs(
        prediction_m[holdout_mask] - target_m[holdout_mask]
    )

    lidar_numpy = depth_batch["lidar_maps"][0]
    sample_id = str(item.get("meta", {}).get("sample_id", dataset_index))
    return CameraDepthSampleAnalysis(
        sample_id=sample_id,
        dataset_index=int(dataset_index),
        rgb=np.moveaxis(depth_batch["rgb"][0], 0, 2),
        prediction_m=prediction_m.astype(np.float32, copy=False),
        splatted_depth_m=(
            lidar_numpy[DEPTH_EXPANDED_CHANNEL_INDEX] * max_depth_m
        ).astype(np.float32, copy=False),
        splat_confidence=lidar_numpy[SPLAT_CONFIDENCE_CHANNEL_INDEX].copy(),
        holdout_target_m=holdout_target_m,
        holdout_absolute_error_m=holdout_absolute_error_m,
        holdout_mask=holdout_mask,
    )


def plot_camera_depth_metrics(metrics: list[dict[str, Any]]):
    """Plot train/validation losses and model/baseline errors by epoch."""

    if not has_camera_depth_validation_metrics(metrics):
        raise ValueError(
            "metrics predate the validation schema; run camera-depth training again."
        )
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in metrics]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(epochs, [row["loss"] for row in metrics], marker="o", label="train")
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in metrics],
        marker="o",
        label="validation",
    )
    axes[0].set(title="Masked Smooth L1", xlabel="Epoch", ylabel="Normalized loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["val_mae_m"] for row in metrics],
        marker="o",
        label="model MAE",
    )
    axes[1].plot(
        epochs,
        [row["val_rmse_m"] for row in metrics],
        marker="o",
        label="model RMSE",
    )
    axes[1].plot(
        epochs,
        [row["val_splat_mae_m"] for row in metrics],
        linestyle="--",
        label="splat MAE",
    )
    axes[1].plot(
        epochs,
        [row["val_splat_rmse_m"] for row in metrics],
        linestyle="--",
        label="splat RMSE",
    )
    axes[1].set(title="Held-out depth error", xlabel="Epoch", ylabel="Meters")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    return figure


def plot_camera_depth_sample(analysis: CameraDepthSampleAnalysis):
    """Plot RGB, splat context, prediction, targets, and held-out errors."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].imshow(np.clip(analysis.rgb, 0.0, 1.0))
    axes[0, 0].set_title(f"RGB - {analysis.sample_id}")

    splatted = np.where(
        analysis.splat_confidence > 0.0,
        analysis.splatted_depth_m,
        np.nan,
    )
    splat_image = axes[0, 1].imshow(splatted, cmap="viridis")
    axes[0, 1].set_title("Input splatted depth (m)")
    figure.colorbar(splat_image, ax=axes[0, 1], fraction=0.046)

    confidence_image = axes[0, 2].imshow(
        analysis.splat_confidence,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 2].set_title("Splat confidence")
    figure.colorbar(confidence_image, ax=axes[0, 2], fraction=0.046)

    prediction_image = axes[1, 0].imshow(analysis.prediction_m, cmap="viridis")
    axes[1, 0].set_title("Predicted dense depth (m)")
    figure.colorbar(prediction_image, ax=axes[1, 0], fraction=0.046)

    holdout_y, holdout_x = np.nonzero(analysis.holdout_mask)
    axes[1, 1].imshow(np.clip(analysis.rgb, 0.0, 1.0))
    target_image = axes[1, 1].scatter(
        holdout_x,
        holdout_y,
        c=analysis.holdout_target_m[analysis.holdout_mask],
        cmap="viridis",
        s=18,
    )
    axes[1, 1].set_title("Held-out LiDAR targets (m)")
    figure.colorbar(target_image, ax=axes[1, 1], fraction=0.046)

    axes[1, 2].imshow(np.clip(analysis.rgb, 0.0, 1.0))
    error_image = axes[1, 2].scatter(
        holdout_x,
        holdout_y,
        c=analysis.holdout_absolute_error_m[analysis.holdout_mask],
        cmap="inferno",
        s=18,
    )
    axes[1, 2].set_title("Held-out absolute error (m)")
    figure.colorbar(error_image, ax=axes[1, 2], fraction=0.046)

    for axis in axes.flat:
        axis.set_axis_off()
    return figure
