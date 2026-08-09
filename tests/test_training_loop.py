from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    describe_device_selection,
    load_training_config,
    run_kitti_camera_depth_training,
    run_synthetic_smoke_training,
    train_one_step,
    validate_camera_depth_training_config,
    validate_training_config,
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


def test_describe_device_selection_reports_request_cuda_availability_and_choice(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    diagnostic = describe_device_selection("auto")

    assert diagnostic == "device=cpu requested=auto cuda_available=False"


def test_describe_device_selection_reports_cuda_device_name_when_selected(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: "Fake RTX")

    diagnostic = describe_device_selection("auto")

    assert diagnostic == "device=cuda requested=auto cuda_available=True cuda_device_name='Fake RTX'"


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


def test_validate_training_config_rejects_non_positive_smoke_dimensions() -> None:
    config = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.01,
        "height": 0,
        "width": 16,
        "output_dir": "results/training/synthetic_smoke",
    }

    with pytest.raises(ValueError, match="height must be a positive integer"):
        validate_training_config(config)


def test_validate_training_config_rejects_kitti_config_for_synthetic_runner() -> None:
    config = {
        "dataset": "kitti_tiny",
        "data_root": "data/kitti",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
        "height": 128,
        "width": 416,
        "output_dir": "results/training/kitti_tiny",
    }

    with pytest.raises(ValueError, match="synthetic smoke runner only supports dataset='synthetic'"):
        validate_training_config(config)


def test_validate_camera_depth_config_rejects_invalid_holdout_fraction() -> None:
    config = {
        "dataset": "kitti_camera_depth",
        "data_root": "data/kitti/training",
        "holdout_fraction": 1.0,
    }

    with pytest.raises(ValueError, match="holdout_fraction must be between 0 and 1"):
        validate_camera_depth_training_config(config)


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


def test_synthetic_smoke_training_writes_metrics_checkpoint_and_resumes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
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
    assert first.device_diagnostic == "device=cpu requested=cpu cuda_available=False"
    assert first.epochs_completed == 1
    assert first.checkpoint_path == output_dir / "checkpoints" / "latest.pt"
    assert first.checkpoint_path.exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics.csv").exists()
    import json

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["device"] == "cpu"
    assert metadata["device_diagnostic"] == "device=cpu requested=cpu cuda_available=False"
    assert metadata["seed"] == 11
    assert metadata["checkpoint_path"] == str(first.checkpoint_path)
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


def test_resumed_synthetic_training_preserves_metrics_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output_dir = tmp_path / "training"
    base_config = {
        "seed": 17,
        "device": "cpu",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.01,
        "loss": "mse",
        "output_dir": str(output_dir),
    }
    first = run_synthetic_smoke_training(base_config)

    run_synthetic_smoke_training(
        {
            **base_config,
            "epochs": 2,
            "resume_from": str(first.checkpoint_path),
        }
    )

    metrics_json = output_dir / "metrics.json"
    metrics_csv = output_dir / "metrics.csv"
    import json

    metrics = json.loads(metrics_json.read_text())
    assert [row["epoch"] for row in metrics] == [1, 2]
    assert metrics_csv.read_text().splitlines()[1:][0].startswith("1,")
    assert metrics_csv.read_text().splitlines()[1:][1].startswith("2,")


def _write_training_kitti_frame(root: Path, sample_id: str, depth_offset: float) -> None:
    Image = pytest.importorskip("PIL.Image")
    image_dir = root / "image_2"
    velodyne_dir = root / "velodyne"
    calibration_dir = root / "calib"
    image_dir.mkdir(parents=True, exist_ok=True)
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    pixels = np.zeros((12, 16, 3), dtype=np.uint8)
    pixels[..., 0] = 64
    pixels[3:9, 4:12, 1] = 160
    Image.fromarray(pixels, mode="RGB").save(image_dir / f"{sample_id}.png")
    points = np.array(
        [
            [-3.0, -1.0, 6.0 + depth_offset, 0.5],
            [-1.5, 0.0, 6.5 + depth_offset, 0.6],
            [0.0, 0.0, 7.0 + depth_offset, 0.7],
            [1.5, 0.0, 7.5 + depth_offset, 0.8],
            [3.0, 1.0, 8.0 + depth_offset, 0.9],
        ],
        dtype=np.float32,
    )
    points.tofile(velodyne_dir / f"{sample_id}.bin")
    (calibration_dir / f"{sample_id}.txt").write_text(
        "\n".join(
            (
                "P2: 8 0 8 0 0 8 6 0 0 0 1 0",
                "R0_rect: 1 0 0 0 1 0 0 0 1",
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0",
            )
        )
        + "\n"
    )


def test_kitti_camera_depth_training_writes_metrics_checkpoint_and_resumes(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data_root = tmp_path / "kitti" / "training"
    _write_training_kitti_frame(data_root, "000000", 0.0)
    _write_training_kitti_frame(data_root, "000001", 0.5)
    output_dir = tmp_path / "camera-depth-training"
    config = {
        "dataset": "kitti_camera_depth",
        "data_root": str(data_root),
        "sample_count": 2,
        "seed": 29,
        "device": "cpu",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "height": 12,
        "width": 16,
        "holdout_fraction": 0.4,
        "splat_radius_px": 1,
        "splat_sigma_px": 1.0,
        "smooth_l1_beta": 0.1,
        "output_dir": str(output_dir),
    }

    first = run_kitti_camera_depth_training(config)

    assert first.device == "cpu"
    assert first.epochs_completed == 1
    assert first.metrics[0].step == 1
    assert first.metrics[0].loss > 0.0
    assert first.checkpoint_path.exists()

    resumed = run_kitti_camera_depth_training(
        {
            **config,
            "epochs": 2,
            "resume_from": str(first.checkpoint_path),
        }
    )

    assert resumed.start_epoch == 1
    assert resumed.metrics[0].epoch == 2
    assert resumed.metrics[0].step == 2
    import json

    metrics = json.loads((output_dir / "metrics.json").read_text())
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert [row["epoch"] for row in metrics] == [1, 2]
    assert metadata["dataset"] == "kitti_camera_depth"
    assert metadata["sample_count"] == 2
    assert metadata["image_shape"] == [12, 16]


def test_train_baseline_cli_reports_config_errors_without_traceback(tmp_path) -> None:
    config_path = tmp_path / "kitti_tiny.yaml"
    config_path.write_text(
        """
        dataset: kitti_tiny
        seed: 13
        device: cpu
        epochs: 1
        batch_size: 1
        learning_rate: 0.001
        loss: smooth_l1
        height: 128
        width: 416
        data_root: data/kitti
        output_dir: results/training/kitti_tiny
        """
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "training"),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "synthetic smoke runner only supports dataset='synthetic'" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_train_baseline_cli_reports_missing_resume_checkpoint_without_traceback(tmp_path) -> None:
    missing_checkpoint = tmp_path / "missing" / "latest.pt"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline.py",
            "--config",
            "configs/training/synthetic_smoke.yaml",
            "--output-dir",
            str(tmp_path / "training"),
            "--resume-from",
            str(missing_checkpoint),
            "--device",
            "cpu",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert f"resume checkpoint does not exist: {missing_checkpoint}" in completed.stderr
    assert "Traceback" not in completed.stderr
