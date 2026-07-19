from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


import numpy as np

from rgb_lidar_fusion.baseline_model import BaselineFusionModel
from rgb_lidar_fusion.model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_baseline_inputs,
    model_batch_to_torch_tensors,
)
from rgb_lidar_fusion.training import (
    build_loss,
    load_training_config,
    run_synthetic_smoke_training,
    train_one_step,
)


def _synthetic_dataset_item(target_value: float = 0.5) -> dict:
    image = np.zeros((3, 8, 8), dtype=np.float32)
    image[0, 1:4, 1:4] = 0.25
    image[1, 2:6, 2:6] = 0.5
    image[2, 3:7, 3:7] = 0.75
    lidar_maps = np.zeros((6, 8, 8), dtype=np.float32)
    lidar_maps[0, 3, 3] = 0.2
    lidar_maps[1, 3, 3] = 0.01
    lidar_maps[2, 3, 3] = 0.02
    lidar_maps[3, 3, 3] = 0.03
    lidar_maps[4, 3, 3] = 0.9
    lidar_maps[5, 3, 3] = 1.0
    return {
        "image": image,
        "lidar_maps": lidar_maps,
        "target": {"regression": target_value},
        "meta": {"sample_id": f"synthetic-{target_value}"},
    }


def test_build_loss_supports_smooth_l1_and_mse_with_clear_errors() -> None:
    assert isinstance(build_loss("smooth_l1"), torch.nn.SmoothL1Loss)
    assert isinstance(build_loss("mse"), torch.nn.MSELoss)

    with pytest.raises(ValueError, match="loss_name must be one of"):
        build_loss("focal")


def test_load_training_config_reads_flat_cpu_safe_yaml(tmp_path) -> None:
    config_path = tmp_path / "synthetic_smoke.yaml"
    config_path.write_text(
        """
        # Minimal YAML subset: flat scalar keys only.
        seed: 123
        device: auto
        epochs: 2
        batch_size: 4
        learning_rate: 0.001
        loss: smooth_l1
        output_dir: results/training/synthetic_smoke
        """
    )

    config = load_training_config(config_path)

    assert config == {
        "seed": 123,
        "device": "auto",
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
        "loss": "smooth_l1",
        "output_dir": "results/training/synthetic_smoke",
    }


def test_train_one_step_consumes_model_batch_adapter_and_updates_parameters() -> None:
    torch.manual_seed(7)
    model = BaselineFusionModel(output_dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    batch = dataset_items_to_model_batch(
        [_synthetic_dataset_item(0.25), _synthetic_dataset_item(0.75)]
    )
    baseline_inputs = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
    tensors = model_batch_to_torch_tensors(baseline_inputs)
    targets = torch.tensor([[0.25], [0.75]], dtype=torch.float32)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_one_step(
        model=model,
        optimizer=optimizer,
        criterion=build_loss("mse"),
        rgb=tensors["rgb"],
        lidar_maps=tensors["lidar_maps"],
        targets=targets,
        device=torch.device("cpu"),
    )

    after = list(model.parameters())
    assert metrics.loss > 0.0
    assert metrics.grad_norm > 0.0
    assert np.isfinite(metrics.loss)
    assert np.isfinite(metrics.grad_norm)
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_synthetic_smoke_training_writes_metrics_checkpoint_and_resumes(tmp_path) -> None:
    output_dir = tmp_path / "training"

    first = run_synthetic_smoke_training(
        {
            "seed": 11,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "loss": "mse",
            "output_dir": str(output_dir),
        }
    )

    assert first.device == "cpu"
    assert first.epochs_completed == 1
    assert first.checkpoint_path == output_dir / "checkpoints" / "latest.pt"
    assert first.checkpoint_path.exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics.csv").exists()
    assert first.metrics[-1].loss > 0.0

    resumed = run_synthetic_smoke_training(
        {
            "seed": 11,
            "device": "cpu",
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.01,
            "loss": "mse",
            "output_dir": str(output_dir),
            "resume_from": str(first.checkpoint_path),
        }
    )

    assert resumed.start_epoch == 1
    assert resumed.epochs_completed == 2
    assert resumed.checkpoint_path.exists()
    assert len(resumed.metrics) == 1
