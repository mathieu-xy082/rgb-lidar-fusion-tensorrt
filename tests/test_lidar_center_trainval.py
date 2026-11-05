from dataclasses import dataclass, replace
import json
from pathlib import Path
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.lidar_center_training import LidarCenterTrainConfig, load_lidar_center_checkpoint
from rgb_lidar_fusion import lidar_center_trainval as training


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "data"
    for name in ("image_2", "calib", "velodyne", "label_2"):
        (root / name).mkdir(parents=True)
    for index in range(8):
        sample_id = f"{index:06d}"
        Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(root / "image_2" / f"{sample_id}.png")
        (root / "calib" / f"{sample_id}.txt").write_text(
            "P2: 10 0 10 0 0 10 10 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        )
        (root / "label_2" / f"{sample_id}.txt").write_text("Car 0 0 0 6 6 14 14 2 2 4 0 1 5 0\n")
        np.array([[5., 0., -.2, .8], [5.1, .1, .2, .5], [6., .5, .1, .9]], dtype=np.float32).tofile(
            root / "velodyne" / f"{sample_id}.bin",
        )
    return LidarCenterTrainConfig(
        data_root=str(root), output_dir=str(tmp_path / "run"), sample_count=8,
        grid=BEVGridConfig((0., 10.), (-4., 4.), (-3., 3.), 1.),
        epochs=2, channels=4, learning_rate=.01, cpu_threads=1, device="cpu",
    )


@pytest.mark.parametrize("values", [
    {"sample_count": 1}, {"sample_count": True}, {"validation_fraction": 0.},
    {"validation_fraction": 1.}, {"validation_fraction": float("nan")},
    {"split_seed": -1}, {"log_every_steps": 0}, {"eval_every": 5},
])
def test_invalid_split_config_is_rejected(values):
    with pytest.raises(ValueError):
        LidarCenterTrainConfig(**values)


def test_split_is_reproducible_disjoint_complete_and_requires_labels(config):
    first = training.build_lidar_center_split(config)
    assert first == training.build_lidar_center_split(config)
    train, val = set(first["train_ids"]), set(first["validation_ids"])
    assert not train & val and train | val == set(first["source_ids"])
    assert len(train) == 6 and len(val) == 2
    assert first["train_ids"] != training.build_lidar_center_split(replace(config, split_seed=14))["train_ids"]
    assert "not a guarantee" in first["limitation"]
    with pytest.raises(ValueError, match="need 512"):
        training.build_lidar_center_split(replace(config, sample_count=512))
    (Path(config.data_root) / "label_2/000000.txt").unlink()
    with pytest.raises(FileNotFoundError):
        training.build_lidar_center_split(config)


def test_validation_never_enters_optimizer_and_eval_preserves_parameters_and_bn(config, monkeypatch):
    updated_ids = []
    original_transfer = training._on_device
    original_evaluate = training._evaluate_split

    def training_transfer(frame, device):
        updated_ids.append(frame.sample_id)
        return original_transfer(frame, device)

    def checked_evaluate(model, *args, **kwargs):
        before = {k: v.clone() for k, v in model.state_dict().items()}
        was_training = model.training
        result = original_evaluate(model, *args, **kwargs)
        assert model.training == was_training
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        return result

    class CheckedModel(training.LidarCenterDiagnostic):
        def forward(self, *args, **kwargs):
            assert torch.is_grad_enabled() == self.training
            return super().forward(*args, **kwargs)

    monkeypatch.setattr(training, "_on_device", training_transfer)
    monkeypatch.setattr(training, "_evaluate_split", checked_evaluate)
    monkeypatch.setattr(training, "LidarCenterDiagnostic", CheckedModel)
    result = training.run_lidar_center_training(config)
    split = result["split"]
    assert set(updated_ids) == set(split["train_ids"])
    assert not set(updated_ids) & set(split["validation_ids"])
    assert len(updated_ids) == config.epochs * len(split["train_ids"])
    assert result["metrics"][-1]["step"] == len(updated_ids)
    assert all("train_eval_loss" in row and "val_loss" in row for row in result["metrics"])


def test_latest_and_best_checkpoints_and_exact_resume(config, tmp_path):
    full = training.run_lidar_center_training(config)
    short_config = replace(config, epochs=1, output_dir=str(tmp_path / "resumed"))
    short = training.run_lidar_center_training(short_config)
    resumed = training.run_lidar_center_training(replace(short_config, epochs=2), resume_from=short["checkpoint"])
    full_model, _, _ = load_lidar_center_checkpoint(full["checkpoint"])
    resumed_model, _, _ = load_lidar_center_checkpoint(resumed["checkpoint"])
    for name, value in full_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
    assert full["metrics"] == resumed["metrics"]
    payload = torch.load(full["best_checkpoint"], map_location="cpu", weights_only=True)
    best_row = min(full["metrics"], key=lambda row: row["val_loss"])
    assert payload["epoch"] == best_row["epoch"] == full["best_epoch"]
    assert payload["metrics"][-1]["val_loss"] == best_row["val_loss"]
    assert payload["purpose"] == "inference" and "optimizer_state_dict" not in payload
    best_model, saved_config, _ = load_lidar_center_checkpoint(full["best_checkpoint"])
    assert not best_model.training and isinstance(saved_config, LidarCenterTrainConfig)
    evaluated = training._evaluate_split(best_model, saved_config, full["split"], "val", torch.device("cpu"))
    assert evaluated["val_loss"] == pytest.approx(best_row["val_loss"])
    with pytest.raises(ValueError, match="latest.pt"):
        training.run_lidar_center_training(config, resume_from=full["best_checkpoint"])


def test_negative_validation_frames_remain_supervised_and_have_no_fake_recall(config):
    split = training.build_lidar_center_split(config)
    for sample_id in split["validation_ids"]:
        (Path(config.data_root) / "label_2" / f"{sample_id}.txt").write_text("")
    result = training.run_lidar_center_training(replace(config, epochs=1))
    row = result["metrics"][-1]
    assert row["val_negative_frames"] == row["val_evaluated_frames"] == 2
    assert row["val_recall"] is None and row["val_matched_mean_distance_m"] is None
    assert row["val_loss"] > 0 and row["val_fn"] == 0


def test_negative_training_frames_are_not_dropped(config):
    split = training.build_lidar_center_split(config)
    for sample_id in split["train_ids"]:
        (Path(config.data_root) / "label_2" / f"{sample_id}.txt").write_text("")
    result = training.run_lidar_center_training(replace(config, epochs=1))
    row = result["metrics"][-1]
    assert row["train_negative_update_frames"] == row["train_negative_frames"] == 6
    assert row["step"] == 6 and row["train_recall"] is None


def test_unknown_supervision_is_skipped_before_forward_and_reported(config, monkeypatch):
    split = training.build_lidar_center_split(config)
    unknown = {split["train_ids"][0], split["validation_ids"][0]}
    original = training._prepare_frame

    def prepare(*args, **kwargs):
        frame = original(*args, **kwargs)
        if frame.sample_id in unknown:
            return replace(frame, targets={k: torch.zeros_like(v) for k, v in frame.targets.items()},
                           centers_xy=np.empty((0, 2), dtype=np.float32))
        return frame

    monkeypatch.setattr(training, "_prepare_frame", prepare)
    result = training.run_lidar_center_training(replace(config, epochs=1))
    row = result["metrics"][-1]
    assert row["step"] == 5 and row["train_evaluated_frames"] == 5 and row["val_evaluated_frames"] == 1
    assert set(row["train_skipped_update_ids"]) == unknown & set(split["train_ids"])
    assert set(row["val_skipped_ids"]) == unknown & set(split["validation_ids"])


def test_streaming_does_not_retain_dataset_pillars(config, monkeypatch):
    refs = []
    maximum_alive = 0
    original = training._prepare_frame

    def prepare(*args, **kwargs):
        nonlocal maximum_alive
        frame = original(*args, **kwargs)
        refs.append(weakref.ref(frame))
        maximum_alive = max(maximum_alive, sum(ref() is not None for ref in refs))
        return frame

    monkeypatch.setattr(training, "_prepare_frame", prepare)
    training.run_lidar_center_training(config)
    assert len(refs) > config.sample_count * config.epochs
    assert maximum_alive <= 4 < config.sample_count


def test_resume_checks_dataset_and_split_and_repairs_sidecars(config):
    short = replace(config, epochs=1)
    result = training.run_lidar_center_training(short)
    output = Path(config.output_dir)
    (output / "metrics.json").write_text("[]")
    (output / "checkpoints/best.pt").write_bytes(b"interrupted sidecar")
    restored = training.run_lidar_center_training(short, resume_from=result["checkpoint"])
    assert json.loads((output / "metrics.json").read_text()) == restored["metrics"]
    load_lidar_center_checkpoint(restored["best_checkpoint"])
    with pytest.raises(ValueError, match="configuration"):
        training.run_lidar_center_training(replace(config, split_seed=99), resume_from=result["checkpoint"])
    path = Path(config.data_root) / "label_2/000000.txt"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset files"):
        training.run_lidar_center_training(config, resume_from=result["checkpoint"])
    with pytest.raises(ValueError, match="not empty"):
        training.run_lidar_center_training(config)


def test_cuda_failure_is_blocked_before_allocation_and_without_retry(config, monkeypatch):
    from rgb_lidar_fusion import camera_depth_analysis

    @dataclass
    class Blocked:
        ready: bool = False

        def format(self):
            return "blocked in test"

    monkeypatch.setattr(camera_depth_analysis, "run_gpu_preflight", lambda **kwargs: Blocked())
    def unexpected(*args, **kwargs):
        pytest.fail("CUDA touched after failed preflight")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", unexpected)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", unexpected)
    events = []
    with pytest.raises(RuntimeError, match="preflight failed"):
        training.run_lidar_center_training(replace(config, device="cuda"), emit=events.append)
    assert events[-1]["event"] == "failed"


def test_overfit_checkpoint_cannot_be_resumed_as_trainval(config):
    path = Path(config.output_dir) / "checkpoints/latest.pt"
    path.parent.mkdir(parents=True)
    torch.save({"kind": "lidar_center_overfit_v1"}, path)
    with pytest.raises(ValueError, match="expected kind"):
        training.run_lidar_center_training(config, resume_from=path)


def test_entirely_unknown_split_raises_instead_of_reporting_perfect_metrics(config, monkeypatch):
    original = training._prepare_frame

    def prepare(*args, **kwargs):
        frame = original(*args, **kwargs)
        return replace(frame, targets={k: torch.zeros_like(v) for k, v in frame.targets.items()},
                       centers_xy=np.empty((0, 2), dtype=np.float32))

    monkeypatch.setattr(training, "_prepare_frame", prepare)
    with pytest.raises(ValueError, match="no supervised frames"):
        training.run_lidar_center_training(config)
    assert not (Path(config.output_dir) / "metrics.json").exists()
