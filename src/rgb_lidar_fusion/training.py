"""Training utilities for the RGB-LiDAR baseline model."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_model import BaselineFusionModel, LIDAR_CHANNELS_BY_MODE
from .camera_depth_model import CameraDepthModel, masked_depth_loss
from .dataset import KittiObjectDepthDataset
from .lidar_splatting import SplattingConfig
from .model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_baseline_inputs,
    model_batch_to_camera_depth_training_batch,
    model_batch_to_torch_tensors,
)
from .project_lidar import ENRICHED_LIDAR_MAP_CHANNELS


DEPTH_EXPANDED_CHANNEL_INDEX = ENRICHED_LIDAR_MAP_CHANNELS.index("depth_expanded")
SPLAT_CONFIDENCE_CHANNEL_INDEX = ENRICHED_LIDAR_MAP_CHANNELS.index("confidence")


def select_camera_depth_lidar_maps(lidar_maps, *, lidar_mode: str):
    """Select sparse or enriched channels from a prepared camera-depth batch."""

    if lidar_mode not in LIDAR_CHANNELS_BY_MODE:
        valid_modes = ", ".join(sorted(LIDAR_CHANNELS_BY_MODE))
        raise ValueError(f"lidar_mode must be one of: {valid_modes}.")
    if lidar_maps.ndim != 4 or lidar_maps.shape[1] != len(
        ENRICHED_LIDAR_MAP_CHANNELS
    ):
        raise ValueError(
            "prepared lidar_maps must have shape [B, 8, H, W] before selection."
        )
    return lidar_maps[:, : LIDAR_CHANNELS_BY_MODE[lidar_mode]]


@dataclass(frozen=True)
class StepMetrics:
    """Scalar diagnostics emitted by one optimizer step."""

    loss: float
    grad_norm: float


@dataclass(frozen=True)
class DepthValidationMetrics:
    """Held-out depth metrics accumulated over a validation split."""

    loss: float
    mae_m: float
    rmse_m: float
    splat_mae_m: float
    splat_rmse_m: float
    splat_coverage: float
    pixel_count: int
    holdout_fraction: float


@dataclass(frozen=True)
class EpochMetrics:
    """Serializable diagnostics for one completed epoch."""

    epoch: int
    step: int
    loss: float
    grad_norm: float
    val_loss: float | None = None
    val_mae_m: float | None = None
    val_rmse_m: float | None = None
    val_splat_mae_m: float | None = None
    val_splat_rmse_m: float | None = None
    val_splat_coverage: float | None = None
    val_pixel_count: int | None = None
    train_holdout_fraction: float | None = None
    val_holdout_fraction: float | None = None


@dataclass(frozen=True)
class TrainingRunResult:
    """Summary returned by the synthetic training smoke runner."""

    device: str
    device_diagnostic: str
    start_epoch: int
    epochs_completed: int
    checkpoint_path: Path
    metrics: list[EpochMetrics]


def build_loss(loss_name: str):
    """Build a small supported training loss by name."""

    import torch

    normalized = loss_name.lower().replace("-", "_")
    if normalized == "smooth_l1":
        return torch.nn.SmoothL1Loss()
    if normalized == "mse":
        return torch.nn.MSELoss()
    raise ValueError("loss_name must be one of: mse, smooth_l1.")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"\'')


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load a flat scalar YAML config without adding a PyYAML dependency."""

    config: dict[str, Any] = {}
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line {line_number}: expected 'key: value'.")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.split(" #", 1)[0].strip()
        if not key or not value:
            raise ValueError(f"invalid config line {line_number}: expected 'key: value'.")
        config[key] = _parse_scalar(value)
    return config


def validate_training_config(config: dict[str, Any]) -> None:
    """Reject smoke-training config values that would create invalid tensors/runs."""

    dataset = str(config.get("dataset", "synthetic"))
    if dataset != "synthetic":
        raise ValueError(
            "synthetic smoke runner only supports dataset='synthetic' until "
            "KITTI pseudo-target loading is implemented."
        )
    positive_int_fields = ("epochs", "batch_size", "height", "width")
    for field in positive_int_fields:
        if field in config and int(config[field]) <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    if "learning_rate" in config and float(config["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive.")


def _camera_depth_holdout_options(config: dict[str, Any]) -> dict[str, Any]:
    """Validate scalar config settings and adapt block ranges to batch options."""

    mode = config.get("holdout_mode", "random_points")
    if mode not in ("random_points", "spatial_blocks"):
        raise ValueError("holdout_mode must be 'random_points' or 'spatial_blocks'.")
    options: dict[str, Any] = {"holdout_mode": mode}
    if mode == "random_points":
        return options

    for axis, default_bounds in (("height", (16, 48)), ("width", (32, 128))):
        bounds = []
        for bound, default in zip(("min", "max"), default_bounds):
            key = f"block_{axis}_{bound}_px"
            value = config.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{key} must be a positive integer.")
            bounds.append(int(value))
        if bounds[0] > bounds[1]:
            raise ValueError(f"block_{axis}_min_px must not exceed block_{axis}_max_px.")
        options[f"block_{axis}_px"] = tuple(bounds)

    max_count = config.get("block_max_count", 8)
    if (
        isinstance(max_count, bool)
        or not isinstance(max_count, (int, np.integer))
        or max_count <= 0
    ):
        raise ValueError("block_max_count must be a positive integer.")
    options["block_max_count"] = int(max_count)
    return options


def validate_camera_depth_training_config(config: dict[str, Any]) -> None:
    """Validate the local KITTI camera-depth runner configuration."""

    dataset = str(config.get("dataset", ""))
    if dataset != "kitti_camera_depth":
        raise ValueError("camera-depth runner requires dataset='kitti_camera_depth'.")
    lidar_mode = str(config.get("lidar_mode", "enriched"))
    if lidar_mode not in LIDAR_CHANNELS_BY_MODE:
        valid_modes = ", ".join(sorted(LIDAR_CHANNELS_BY_MODE))
        raise ValueError(f"lidar_mode must be one of: {valid_modes}.")
    positive_int_fields = (
        "epochs",
        "batch_size",
        "height",
        "width",
        "sample_count",
    )
    for field in positive_int_fields:
        if field in config and int(config[field]) <= 0:
            raise ValueError(f"{field} must be a positive integer.")
    sample_count = int(config.get("sample_count", 64))
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2 for a train/validation split.")
    positive_float_fields = (
        "learning_rate",
        "max_depth_m",
        "splat_sigma_px",
        "smooth_l1_beta",
    )
    for field in positive_float_fields:
        if field in config and float(config[field]) <= 0.0:
            raise ValueError(f"{field} must be positive.")
    holdout_fraction = float(config.get("holdout_fraction", 0.2))
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    _camera_depth_holdout_options(config)
    validation_fraction = float(config.get("validation_fraction", 0.2))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    try:
        int(config.get("split_seed", config.get("seed", 0)))
    except (TypeError, ValueError):
        raise ValueError("split_seed must be an integer.") from None
    try:
        int(
            config.get(
                "validation_holdout_seed",
                config.get("split_seed", config.get("seed", 0)),
            )
        )
    except (TypeError, ValueError):
        raise ValueError("validation_holdout_seed must be an integer.") from None
    if int(config.get("splat_radius_px", 2)) < 0:
        raise ValueError("splat_radius_px must be non-negative.")
    if not str(config.get("data_root", "")).strip():
        raise ValueError("data_root is required for KITTI camera-depth training.")


def split_dataset_indices(
    sample_count: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic, disjoint train and validation index lists."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2 for a train/validation split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    shuffled = np.random.default_rng(seed).permutation(sample_count)
    validation_count = int(round(sample_count * validation_fraction))
    validation_count = min(sample_count - 1, max(1, validation_count))

    validation_indices = sorted(int(index) for index in shuffled[:validation_count])
    train_indices = sorted(int(index) for index in shuffled[validation_count:])
    return train_indices, validation_indices


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for deterministic smoke training."""

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on host GPU.
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(requested: str = "auto"):
    """Select ``cuda`` when available for auto mode, otherwise ``cpu``."""

    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested but CUDA is not available.")
    return torch.device(requested)


def describe_device_selection(requested: str = "auto") -> str:
    """Return an explicit device-selection diagnostic for logs and CLI output."""

    import torch

    device = select_device(requested)
    cuda_available = torch.cuda.is_available()
    diagnostic = f"device={device} requested={requested} cuda_available={cuda_available}"
    if device.type == "cuda":  # pragma: no cover - real CUDA depends on host GPU.
        diagnostic += f" cuda_device_name={torch.cuda.get_device_name(0)!r}"
    return diagnostic


def train_one_step(
    *,
    model,
    optimizer,
    criterion,
    rgb,
    lidar_maps,
    targets,
    device,
) -> StepMetrics:
    """Run one supervised optimizer step and return finite scalar diagnostics."""

    import torch

    model.to(device)
    model.train()
    rgb = rgb.to(device)
    lidar_maps = lidar_maps.to(device)
    targets = targets.to(device)

    optimizer.zero_grad(set_to_none=True)
    predictions = model(rgb, lidar_maps)
    loss = criterion(predictions, targets)
    if not torch.isfinite(loss):
        raise ValueError("training loss must be finite.")
    loss.backward()

    grad_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("all gradients must be finite.")
        grad_norm_sq += float(parameter.grad.detach().pow(2).sum().cpu())
    grad_norm = grad_norm_sq**0.5
    if grad_norm <= 0.0:
        raise ValueError("at least one trainable parameter must receive a non-zero gradient.")

    optimizer.step()
    return StepMetrics(loss=float(loss.detach().cpu()), grad_norm=grad_norm)


def train_camera_depth_step(
    *,
    model,
    optimizer,
    rgb,
    lidar_maps,
    depth_target,
    loss_mask,
    device,
    beta: float = 1.0,
) -> StepMetrics:
    """Run one optimizer step against held-out sparse depth measurements."""

    import torch

    model.to(device)
    model.train()
    rgb = rgb.to(device)
    lidar_maps = lidar_maps.to(device)
    depth_target = depth_target.to(device)
    loss_mask = loss_mask.to(device)

    optimizer.zero_grad(set_to_none=True)
    prediction = model(rgb, lidar_maps)
    loss = masked_depth_loss(prediction, depth_target, loss_mask, beta=beta)
    if not torch.isfinite(loss):
        raise ValueError("training loss must be finite.")
    loss.backward()

    grad_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise ValueError("all gradients must be finite.")
        grad_norm_sq += float(parameter.grad.detach().pow(2).sum().cpu())
    grad_norm = grad_norm_sq**0.5
    if grad_norm <= 0.0:
        raise ValueError("at least one trainable parameter must receive a non-zero gradient.")

    optimizer.step()
    return StepMetrics(loss=float(loss.detach().cpu()), grad_norm=grad_norm)


def evaluate_camera_depth(
    *,
    model,
    dataset,
    indices: list[int],
    batch_size: int,
    holdout_fraction: float,
    holdout_seed: int,
    splatting_config: SplattingConfig,
    device,
    max_depth_m: float,
    beta: float = 1.0,
    lidar_mode: str = "enriched",
    holdout_mode: str = "random_points",
    block_height_px: tuple[int, int] = (16, 48),
    block_width_px: tuple[int, int] = (32, 128),
    block_max_count: int = 8,
) -> DepthValidationMetrics:
    """Evaluate one holdout mode with fixed masks and its achieved fraction."""

    import torch

    if not indices:
        raise ValueError("validation indices must not be empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_depth_m <= 0.0:
        raise ValueError("max_depth_m must be positive.")

    loss_sum = 0.0
    absolute_error_m_sum = 0.0
    squared_error_m_sum = 0.0
    splat_absolute_error_m_sum = 0.0
    splat_squared_error_m_sum = 0.0
    splat_covered_count = 0
    pixel_count = 0
    valid_pixel_count = 0
    was_training = model.training
    model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start : batch_start + batch_size]
                depth_batch = prepare_camera_depth_evaluation_batch(
                    dataset=dataset,
                    indices=[int(index) for index in batch_indices],
                    holdout_fraction=holdout_fraction,
                    holdout_seed=holdout_seed,
                    splatting_config=splatting_config,
                    holdout_mode=holdout_mode,
                    block_height_px=block_height_px,
                    block_width_px=block_width_px,
                    block_max_count=block_max_count,
                )
                rgb = torch.from_numpy(depth_batch["rgb"]).to(device)
                prepared_lidar_maps = torch.from_numpy(
                    depth_batch["lidar_maps"]
                ).to(device)
                lidar_maps = select_camera_depth_lidar_maps(
                    prepared_lidar_maps,
                    lidar_mode=lidar_mode,
                )
                depth_target = torch.from_numpy(depth_batch["depth_target"]).to(device)
                loss_mask = torch.from_numpy(depth_batch["loss_mask"]).to(device)
                prediction = model(rgb, lidar_maps)
                if not torch.isfinite(prediction).all():
                    raise ValueError("validation predictions must be finite.")

                selected = loss_mask.to(dtype=torch.bool)
                batch_pixel_count = int(selected.sum().item())
                batch_loss = masked_depth_loss(
                    prediction,
                    depth_target,
                    loss_mask,
                    beta=beta,
                )
                error_m = (prediction[selected] - depth_target[selected]) * max_depth_m
                splat_depth = prepared_lidar_maps[
                    :, DEPTH_EXPANDED_CHANNEL_INDEX : DEPTH_EXPANDED_CHANNEL_INDEX + 1
                ]
                splat_error_m = (
                    splat_depth[selected] - depth_target[selected]
                ) * max_depth_m
                splat_confidence = prepared_lidar_maps[
                    :, SPLAT_CONFIDENCE_CHANNEL_INDEX : SPLAT_CONFIDENCE_CHANNEL_INDEX + 1
                ]

                loss_sum += float(batch_loss.cpu()) * batch_pixel_count
                absolute_error_m_sum += float(error_m.abs().sum().cpu())
                squared_error_m_sum += float(error_m.square().sum().cpu())
                splat_absolute_error_m_sum += float(splat_error_m.abs().sum().cpu())
                splat_squared_error_m_sum += float(splat_error_m.square().sum().cpu())
                splat_covered_count += int((splat_confidence[selected] > 0.0).sum().item())
                pixel_count += batch_pixel_count
                kept = depth_batch["kept_sparse_lidar_maps"]
                kept_pixel_count = int(
                    np.count_nonzero((kept[:, 5:6] > 0.0) & (kept[:, 0:1] > 0.0))
                )
                valid_pixel_count += batch_pixel_count + kept_pixel_count
    finally:
        model.train(was_training)

    return DepthValidationMetrics(
        loss=loss_sum / pixel_count,
        mae_m=absolute_error_m_sum / pixel_count,
        rmse_m=(squared_error_m_sum / pixel_count) ** 0.5,
        splat_mae_m=splat_absolute_error_m_sum / pixel_count,
        splat_rmse_m=(splat_squared_error_m_sum / pixel_count) ** 0.5,
        splat_coverage=splat_covered_count / pixel_count,
        pixel_count=pixel_count,
        holdout_fraction=pixel_count / valid_pixel_count,
    )


def prepare_camera_depth_evaluation_batch(
    *,
    dataset,
    indices: list[int],
    holdout_fraction: float,
    holdout_seed: int,
    splatting_config: SplattingConfig,
    holdout_mode: str = "random_points",
    block_height_px: tuple[int, int] = (16, 48),
    block_width_px: tuple[int, int] = (32, 128),
    block_max_count: int = 8,
) -> dict[str, np.ndarray]:
    """Build fixed random-point or block masks using holdout_seed + dataset index."""

    if not indices:
        raise ValueError("evaluation indices must not be empty.")

    prepared_samples = []
    for index in indices:
        model_batch = dataset_items_to_model_batch(
            [dataset[int(index)]],
            include_splatted_depth=False,
        )
        prepared_samples.append(
            model_batch_to_camera_depth_training_batch(
                model_batch,
                holdout_fraction=holdout_fraction,
                seed=holdout_seed + int(index),
                splatting_config=splatting_config,
                holdout_mode=holdout_mode,
                block_height_px=block_height_px,
                block_width_px=block_width_px,
                block_max_count=block_max_count,
            )
        )

    return {
        key: np.concatenate(
            [sample[key] for sample in prepared_samples],
            axis=0,
        )
        for key in prepared_samples[0]
    }


def _synthetic_dataset_item(index: int, *, height: int, width: int) -> dict[str, Any]:
    image = np.zeros((3, height, width), dtype=np.float32)
    offset = index % max(1, width - 3)
    image[0, 1:4, offset : offset + 3] = 0.2 + 0.1 * index
    image[1, 2:6, 1:5] = 0.4
    image[2, 3:7, 3:7] = 0.6
    lidar_maps = np.zeros((6, height, width), dtype=np.float32)
    y = min(height - 2, 2 + index)
    x = min(width - 2, 2 + index)
    depth = 0.15 + 0.1 * index
    lidar_maps[0, y, x] = depth
    lidar_maps[1, y, x] = 0.01 * (index + 1)
    lidar_maps[2, y, x] = 0.02 * (index + 1)
    lidar_maps[3, y, x] = 0.03 * (index + 1)
    lidar_maps[4, y, x] = 0.8
    lidar_maps[5, y, x] = 1.0
    return {
        "image": image,
        "lidar_maps": lidar_maps,
        "target": {"regression": depth},
        "meta": {"sample_id": f"synthetic-training-{index}"},
    }


def _training_targets(items: list[dict[str, Any]]):
    import torch

    return torch.tensor(
        [[float(item["target"]["regression"])] for item in items],
        dtype=torch.float32,
    )


def save_checkpoint(
    *,
    path: Path,
    model,
    optimizer,
    epoch: int,
    step: int,
    config: dict[str, Any],
) -> None:
    """Save model/optimizer state outside Git-controlled source paths."""

    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
        },
        path,
    )


def load_checkpoint(*, path: Path, model, optimizer, device) -> tuple[int, int]:
    """Load a checkpoint and return the next start epoch plus step count."""

    import torch

    if not path.exists():
        raise ValueError(f"resume checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["epoch"]), int(payload.get("step", 0))


def _read_existing_metric_rows(output_dir: Path) -> list[dict[str, Any]]:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    rows = json.loads(metrics_path.read_text())
    if not isinstance(rows, list):
        raise ValueError("existing metrics.json must contain a list of metric rows.")
    return rows


def write_metrics(output_dir: Path, metrics: list[EpochMetrics], *, append: bool = False) -> None:
    """Write metrics as JSON and CSV sidecars under the configured output dir."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_existing_metric_rows(output_dir) if append else []
    rows.extend(asdict(metric) for metric in metrics)
    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(asdict(EpochMetrics(0, 0, 0.0, 0.0))),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_run_metadata(
    output_dir: Path,
    *,
    config: dict[str, Any],
    device: str,
    device_diagnostic: str,
    checkpoint_path: Path,
    start_epoch: int,
    epochs_completed: int,
    split_metadata: dict[str, Any] | None = None,
) -> None:
    """Write lightweight run metadata for CPU/GPU training traceability."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": int(config.get("seed", 0)),
        "dataset": str(config.get("dataset", "synthetic")),
        "device": device,
        "device_diagnostic": device_diagnostic,
        "start_epoch": start_epoch,
        "epochs_completed": epochs_completed,
        "checkpoint_path": str(checkpoint_path),
    }
    if "data_root" in config:
        metadata["data_root"] = str(config["data_root"])
    if "lidar_mode" in config:
        metadata["lidar_mode"] = str(config["lidar_mode"])
    if "sample_count" in config:
        metadata["sample_count"] = int(config["sample_count"])
    if "height" in config and "width" in config:
        metadata["image_shape"] = [int(config["height"]), int(config["width"])]
    if config.get("dataset") == "kitti_camera_depth":
        metadata["holdout"] = {
            **_camera_depth_holdout_options(config),
            "requested_fraction": float(config.get("holdout_fraction", 0.2)),
        }
    if split_metadata is not None:
        metadata["split"] = split_metadata
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_synthetic_smoke_training(config: dict[str, Any]) -> TrainingRunResult:
    """Train the baseline on a tiny deterministic synthetic batch.

    The smoke runner intentionally uses the canonical dataset→model adapters:
    ``dataset_items_to_model_batch`` → ``model_batch_to_baseline_inputs`` →
    ``model_batch_to_torch_tensors``. It does not hand-slice the generic
    concatenated ``inputs`` tensor.
    """

    import torch

    validate_training_config(config)

    seed = int(config.get("seed", 0))
    requested_device = str(config.get("device", "auto"))
    set_deterministic_seed(seed)
    device = select_device(requested_device)
    device_diagnostic = describe_device_selection(requested_device)
    epochs = int(config.get("epochs", 1))
    batch_size = int(config.get("batch_size", 2))
    learning_rate = float(config.get("learning_rate", 1e-3))
    height = int(config.get("height", 16))
    width = int(config.get("width", 16))
    output_dir = Path(str(config.get("output_dir", "results/training/synthetic_smoke")))
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"

    model = BaselineFusionModel(output_dim=1, lidar_mode="enriched")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = build_loss(str(config.get("loss", "smooth_l1")))

    start_epoch = 0
    step = 0
    resume_from = config.get("resume_from")
    if resume_from:
        start_epoch, step = load_checkpoint(
            path=Path(str(resume_from)),
            model=model,
            optimizer=optimizer,
            device=device,
        )

    items = [_synthetic_dataset_item(i, height=height, width=width) for i in range(batch_size)]
    batch = dataset_items_to_model_batch(items)
    baseline_inputs = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
    tensors = model_batch_to_torch_tensors(baseline_inputs)
    targets = _training_targets(items)

    metrics: list[EpochMetrics] = []
    for epoch in range(start_epoch, epochs):
        step += 1
        step_metrics = train_one_step(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            rgb=tensors["rgb"],
            lidar_maps=tensors["lidar_maps"],
            targets=targets,
            device=device,
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch + 1,
                step=step,
                loss=step_metrics.loss,
                grad_norm=step_metrics.grad_norm,
            )
        )
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=step,
            config=config,
        )

    write_metrics(output_dir, metrics, append=bool(resume_from))
    write_run_metadata(
        output_dir,
        config=config,
        device=str(device),
        device_diagnostic=device_diagnostic,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch,
        epochs_completed=epochs,
    )
    return TrainingRunResult(
        device=str(device),
        device_diagnostic=device_diagnostic,
        start_epoch=start_epoch,
        epochs_completed=epochs,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )


def run_kitti_camera_depth_training(config: dict[str, Any]) -> TrainingRunResult:
    """Train dense camera depth against held-out KITTI LiDAR measurements."""

    import torch

    validate_camera_depth_training_config(config)
    seed = int(config.get("seed", 0))
    requested_device = str(config.get("device", "auto"))
    set_deterministic_seed(seed)
    device = select_device(requested_device)
    device_diagnostic = describe_device_selection(requested_device)
    epochs = int(config.get("epochs", 1))
    batch_size = int(config.get("batch_size", 2))
    learning_rate = float(config.get("learning_rate", 1e-3))
    height = int(config.get("height", 320))
    width = int(config.get("width", 1024))
    sample_count = int(config.get("sample_count", 64))
    holdout_fraction = float(config.get("holdout_fraction", 0.2))
    holdout_options = _camera_depth_holdout_options(config)
    validation_fraction = float(config.get("validation_fraction", 0.2))
    split_seed = int(config.get("split_seed", seed))
    validation_holdout_seed = int(
        config.get("validation_holdout_seed", split_seed)
    )
    lidar_mode = str(config.get("lidar_mode", "enriched"))
    max_depth_m = float(config.get("max_depth_m", 80.0))
    beta = float(config.get("smooth_l1_beta", 1.0))
    output_dir = Path(
        str(config.get("output_dir", "results/training/kitti_camera_depth"))
    )
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    splatting_config = SplattingConfig(
        radius_px=int(config.get("splat_radius_px", 2)),
        sigma_px=float(config.get("splat_sigma_px", 1.0)),
    )
    dataset = KittiObjectDepthDataset(
        str(config["data_root"]),
        image_shape=(height, width),
        max_depth_m=max_depth_m,
        sample_limit=sample_count,
    )
    train_indices, validation_indices = split_dataset_indices(
        len(dataset),
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    model = CameraDepthModel(lidar_mode=lidar_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    step = 0
    resume_from = config.get("resume_from")
    if resume_from:
        start_epoch, step = load_checkpoint(
            path=Path(str(resume_from)),
            model=model,
            optimizer=optimizer,
            device=device,
        )

    metrics: list[EpochMetrics] = []
    for epoch in range(start_epoch, epochs):
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []
        train_target_pixel_count = 0
        train_valid_pixel_count = 0
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        for batch_start in range(0, len(order), batch_size):
            indices = order[batch_start : batch_start + batch_size]
            items = [dataset[int(index)] for index in indices]
            model_batch = dataset_items_to_model_batch(
                items,
                include_splatted_depth=False,
            )
            depth_batch = model_batch_to_camera_depth_training_batch(
                model_batch,
                holdout_fraction=holdout_fraction,
                seed=seed + epoch * len(dataset) + batch_start,
                splatting_config=splatting_config,
                **holdout_options,
            )
            train_target_pixel_count += int(depth_batch["loss_mask"].sum())
            sparse = model_batch["sparse_lidar_maps"]
            train_valid_pixel_count += int(
                np.count_nonzero((sparse[:, 5:6] > 0.0) & (sparse[:, 0:1] > 0.0))
            )
            lidar_maps = select_camera_depth_lidar_maps(
                depth_batch["lidar_maps"],
                lidar_mode=lidar_mode,
            )
            step += 1
            step_metrics = train_camera_depth_step(
                model=model,
                optimizer=optimizer,
                rgb=torch.from_numpy(depth_batch["rgb"]),
                lidar_maps=torch.from_numpy(lidar_maps),
                depth_target=torch.from_numpy(depth_batch["depth_target"]),
                loss_mask=torch.from_numpy(depth_batch["loss_mask"]),
                device=device,
                beta=beta,
            )
            epoch_losses.append(step_metrics.loss)
            epoch_grad_norms.append(step_metrics.grad_norm)

        validation_metrics = evaluate_camera_depth(
            model=model,
            dataset=dataset,
            indices=validation_indices,
            batch_size=batch_size,
            holdout_fraction=holdout_fraction,
            holdout_seed=validation_holdout_seed,
            splatting_config=splatting_config,
            device=device,
            max_depth_m=max_depth_m,
            beta=beta,
            lidar_mode=lidar_mode,
            **holdout_options,
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch + 1,
                step=step,
                loss=float(np.mean(epoch_losses)),
                grad_norm=float(np.mean(epoch_grad_norms)),
                val_loss=validation_metrics.loss,
                val_mae_m=validation_metrics.mae_m,
                val_rmse_m=validation_metrics.rmse_m,
                val_splat_mae_m=validation_metrics.splat_mae_m,
                val_splat_rmse_m=validation_metrics.splat_rmse_m,
                val_splat_coverage=validation_metrics.splat_coverage,
                val_pixel_count=validation_metrics.pixel_count,
                train_holdout_fraction=(
                    train_target_pixel_count / train_valid_pixel_count
                ),
                val_holdout_fraction=validation_metrics.holdout_fraction,
            )
        )
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=step,
            config=config,
        )

    write_metrics(output_dir, metrics, append=bool(resume_from))
    write_run_metadata(
        output_dir,
        config=config,
        device=str(device),
        device_diagnostic=device_diagnostic,
        checkpoint_path=checkpoint_path,
        start_epoch=start_epoch,
        epochs_completed=epochs,
        split_metadata={
            "seed": split_seed,
            "holdout_seed": validation_holdout_seed,
            "validation_fraction": validation_fraction,
            "train_sample_count": len(train_indices),
            "validation_sample_count": len(validation_indices),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
        },
    )
    return TrainingRunResult(
        device=str(device),
        device_diagnostic=device_diagnostic,
        start_epoch=start_epoch,
        epochs_completed=epochs,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
