"""Reusable metric and visualization helpers for camera-depth experiments."""

from __future__ import annotations

import json
import os
import re
import subprocess
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

NVIDIA_GPU_QUERY_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
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


@dataclass(frozen=True)
class GpuProcess:
    """One CUDA compute process reported by NVIDIA SMI."""

    pid: int
    name: str
    memory_mib: int


@dataclass(frozen=True)
class GpuPreflightReport:
    """GPU readiness diagnostics collected before notebook training."""

    ready: bool
    torch_version: str
    cuda_available: bool
    device_name: str | None
    driver_version: str | None
    memory_total_mib: int | None
    memory_used_mib: int | None
    memory_free_mib: int | None
    utilization_percent: int | None
    recovery_action: str | None
    other_compute_processes: tuple[GpuProcess, ...]
    smoke_test_passed: bool
    issues: tuple[str, ...]

    def format(self) -> str:
        """Return a compact multi-line summary suitable for a notebook cell."""

        lines = [
            f"ready={self.ready}",
            f"PyTorch={self.torch_version} cuda_available={self.cuda_available}",
        ]
        if self.device_name is not None:
            lines.append(
                f"GPU={self.device_name} driver={self.driver_version} "
                f"memory={self.memory_used_mib}/{self.memory_total_mib} MiB "
                f"free={self.memory_free_mib} MiB utilization={self.utilization_percent}%"
            )
        if self.recovery_action is not None:
            lines.append(f"recovery_action={self.recovery_action}")
        if self.other_compute_processes:
            processes = ", ".join(
                f"pid={process.pid} {process.name} ({process.memory_mib} MiB)"
                for process in self.other_compute_processes
            )
            lines.append(f"other_compute_processes={processes}")
        lines.extend(f"ISSUE: {issue}" for issue in self.issues)
        return "\n".join(lines)


def _run_nvidia_smi(*arguments: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


def _parse_nvidia_gpu_row(output: str) -> dict[str, Any]:
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows.")
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(NVIDIA_GPU_QUERY_FIELDS):
        raise ValueError("nvidia-smi returned an unexpected GPU query schema.")
    return {
        "device_name": values[0],
        "driver_version": values[1],
        "memory_total_mib": int(values[2]),
        "memory_used_mib": int(values[3]),
        "memory_free_mib": int(values[4]),
        "utilization_percent": int(values[5]),
    }


def _parse_nvidia_compute_processes(output: str) -> tuple[GpuProcess, ...]:
    processes = []
    for row in output.splitlines():
        if not row.strip():
            continue
        values = [value.strip() for value in row.split(",", maxsplit=2)]
        if len(values) != 3:
            raise ValueError("nvidia-smi returned an unexpected process schema.")
        processes.append(
            GpuProcess(
                pid=int(values[0]),
                name=values[1],
                memory_mib=int(values[2]),
            )
        )
    return tuple(processes)


def _parse_nvidia_recovery_action(output: str) -> str | None:
    match = re.search(r"GPU Recovery Action\s*:\s*([^\r\n]+)", output)
    return None if match is None else match.group(1).strip()


def run_gpu_preflight(
    *,
    minimum_free_memory_mib: int = 2500,
    run_smoke_test: bool = True,
) -> GpuPreflightReport:
    """Check NVIDIA/PyTorch state and optionally run a synchronized CUDA probe."""

    import torch

    if minimum_free_memory_mib <= 0:
        raise ValueError("minimum_free_memory_mib must be positive.")

    issues: list[str] = []
    gpu_status: dict[str, Any] = {}
    recovery_action = None
    processes: tuple[GpuProcess, ...] = ()
    try:
        query_fields = ",".join(NVIDIA_GPU_QUERY_FIELDS)
        gpu_status = _parse_nvidia_gpu_row(
            _run_nvidia_smi(
                f"--query-gpu={query_fields}",
                "--format=csv,noheader,nounits",
            )
        )
        processes = _parse_nvidia_compute_processes(
            _run_nvidia_smi(
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
        )
        recovery_action = _parse_nvidia_recovery_action(_run_nvidia_smi("-q"))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        issues.append(f"NVIDIA SMI diagnostics failed: {exc}")

    other_processes = tuple(process for process in processes if process.pid != os.getpid())
    if other_processes:
        issues.append("another CUDA compute process is already using the GPU.")

    memory_free_mib = gpu_status.get("memory_free_mib")
    if memory_free_mib is not None and memory_free_mib < minimum_free_memory_mib:
        issues.append(
            f"only {memory_free_mib} MiB GPU memory is free; "
            f"at least {minimum_free_memory_mib} MiB is required."
        )

    if recovery_action is not None and recovery_action.lower() not in {
        "none",
        "n/a",
        "not required",
    }:
        issues.append(f"NVIDIA reports recovery action {recovery_action!r}.")

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on broken driver state.
        cuda_available = False
        issues.append(f"PyTorch CUDA availability check failed: {exc}")
    if not cuda_available:
        issues.append("PyTorch reports CUDA as unavailable.")

    smoke_test_passed = False
    if run_smoke_test and cuda_available and not issues:
        probe = None
        try:
            probe = torch.arange(1024, dtype=torch.float32, device="cuda")
            probe_result = probe.square().sum()
            torch.cuda.synchronize()
            if not torch.isfinite(probe_result):
                raise ValueError("CUDA probe returned a non-finite result.")
            smoke_test_passed = True
        except Exception as exc:  # pragma: no cover - depends on real CUDA failures.
            issues.append(f"synchronized CUDA smoke test failed: {exc}")
        finally:
            del probe
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    elif not run_smoke_test and cuda_available and not issues:
        smoke_test_passed = True

    return GpuPreflightReport(
        ready=not issues and smoke_test_passed,
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        device_name=gpu_status.get("device_name"),
        driver_version=gpu_status.get("driver_version"),
        memory_total_mib=gpu_status.get("memory_total_mib"),
        memory_used_mib=gpu_status.get("memory_used_mib"),
        memory_free_mib=memory_free_mib,
        utilization_percent=gpu_status.get("utilization_percent"),
        recovery_action=recovery_action,
        other_compute_processes=other_processes,
        smoke_test_passed=smoke_test_passed,
        issues=tuple(issues),
    )


def require_gpu_ready(report: GpuPreflightReport) -> None:
    """Stop notebook training when the GPU preflight did not pass."""

    if not report.ready:
        raise RuntimeError(
            "GPU preflight failed; do not start training.\n" + report.format()
        )


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
