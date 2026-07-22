"""Training utilities for the RGB-LiDAR baseline model."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_model import BaselineFusionModel
from .model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_baseline_inputs,
    model_batch_to_torch_tensors,
)


@dataclass(frozen=True)
class StepMetrics:
    """Scalar diagnostics emitted by one optimizer step."""

    loss: float
    grad_norm: float


@dataclass(frozen=True)
class EpochMetrics:
    """Serializable diagnostics for one completed epoch."""

    epoch: int
    step: int
    loss: float
    grad_norm: float


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
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "step", "loss", "grad_norm"])
        writer.writeheader()
        writer.writerows(rows)


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
    return TrainingRunResult(
        device=str(device),
        device_diagnostic=device_diagnostic,
        start_epoch=start_epoch,
        epochs_completed=epochs,
        checkpoint_path=checkpoint_path,
        metrics=metrics,
    )
