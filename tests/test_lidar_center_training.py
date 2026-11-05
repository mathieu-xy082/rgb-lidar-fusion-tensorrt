from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.lidar_center_training import (
    LidarCenterOverfitConfig, load_lidar_center_checkpoint, match_lidar_centers, run_lidar_center_overfit,
)


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "data"
    for name in ("image_2", "calib", "velodyne", "label_2"):
        (root / name).mkdir(parents=True)
    for sample_id in ("000000", "000001"):
        Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(root / "image_2" / f"{sample_id}.png")
        (root / "calib" / f"{sample_id}.txt").write_text(
            "P2: 10 0 10 0 0 10 10 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        )
        (root / "label_2" / f"{sample_id}.txt").write_text("Car 0 0 0 6 6 14 14 2 2 4 0 1 5 0\n")
        np.array([[5., 0., -.2, .8], [5.1, .1, .2, .5], [6., .5, .1, .9]], dtype=np.float32).tofile(
            root / "velodyne" / f"{sample_id}.bin",
        )
    return LidarCenterOverfitConfig(
        data_root=str(root), output_dir=str(tmp_path / "run"), sample_ids=("000000", "000001"),
        grid=BEVGridConfig((0., 10.), (-4., 4.), (-3., 3.), 1.),
        epochs=8, eval_every=1, channels=4, learning_rate=.01, cpu_threads=1,
    )


@pytest.mark.parametrize("change", [
    {"sample_ids": []}, {"sample_ids": ["000000"] * 5}, {"sample_ids": ["000000", "000000"]},
    {"sample_ids": ["../bad"]}, {"sample_ids": "000000"}, {"epochs": True}, {"epochs": 0},
    {"seed": -1}, {"learning_rate": float("nan")}, {"device": "auto"}, {"score_threshold": 0},
    {"top_k": 0}, {"match_radius_m": float("inf")},
])
def test_invalid_overfit_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        LidarCenterOverfitConfig(**change)


def test_config_json_round_trip_and_unknown_options(config):
    restored = LidarCenterOverfitConfig.from_mapping(config.serialized())
    assert restored.serialized() == config.serialized()
    with pytest.raises(ValueError, match="configuration"):
        LidarCenterOverfitConfig.from_mapping({"unknown": 1})
    with pytest.raises(ValueError, match="configuration"):
        LidarCenterOverfitConfig.from_mapping({"grid": {"bad": 1}})


def test_center_matching_is_score_ordered_one_to_one_and_counts_duplicates():
    predicted = np.array([[.8, 0., .1], [0., 0., .9], [9., 0., .8]])
    counts = match_lidar_centers(predicted, np.array([[0., 0.], [10., 0.]]), radius_m=1.)
    assert counts == {"tp": 2, "fp": 1, "fn": 0, "matched_distance_sum_m": 1.}
    empty = match_lidar_centers(np.empty((0, 3)), np.array([[1., 0.]]), radius_m=1.)
    assert empty["tp"] == 0 and empty["fn"] == 1 and empty["matched_distance_sum_m"] == 0


def test_training_writes_explicit_memorization_artifacts_and_loads_cpu_model(config):
    events = []
    threads = torch.get_num_threads()
    state = torch.random.get_rng_state().clone()
    result = run_lidar_center_overfit(config, emit=events.append)
    assert torch.get_num_threads() == threads
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    assert result["epochs_completed"] == 8 and result["device"] == "cpu"
    assert result["metrics"][0]["epoch"] == 0 and result["metrics"][0]["train_loss"] is None
    assert result["metrics"][-1]["step"] == 16
    assert result["metrics"][-1]["train_loss"] < result["metrics"][1]["train_loss"]
    assert all(event["event"] != "preflight" for event in events)
    assert events[-1]["event"] == "completed"
    model, loaded, history = load_lidar_center_checkpoint(result["checkpoint"])
    assert not model.training and next(model.parameters()).device.type == "cpu"
    assert loaded.serialized() == config.serialized() and history == result["metrics"]
    metadata = json.loads((Path(config.output_dir) / "run_metadata.json").read_text())
    assert metadata["train_ids"] == metadata["evaluation_ids"] == list(config.sample_ids)
    assert "no held-out" in metadata["evaluation_scope"]
    assert "val_loss" not in history[-1]


def test_resume_restores_adam_bn_and_epoch_order_exactly(config, tmp_path):
    whole = run_lidar_center_overfit(config)
    partial_config = replace(config, output_dir=str(tmp_path / "resumed"), epochs=4)
    partial = run_lidar_center_overfit(partial_config)
    resumed = run_lidar_center_overfit(replace(partial_config, epochs=8), resume_from=partial["checkpoint"])
    first, _, _ = load_lidar_center_checkpoint(whole["checkpoint"])
    second, _, _ = load_lidar_center_checkpoint(resumed["checkpoint"])
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert whole["metrics"] == resumed["metrics"]
    with pytest.raises(ValueError, match="configuration"):
        run_lidar_center_overfit(replace(partial_config, learning_rate=.02, epochs=9), resume_from=partial["checkpoint"])


def test_existing_runs_and_changed_dataset_cannot_be_silently_overwritten(config):
    from pathlib import Path

    result = run_lidar_center_overfit(replace(config, epochs=1))
    with pytest.raises(ValueError, match="not empty"):
        run_lidar_center_overfit(config)
    path = Path(config.data_root) / "label_2/000000.txt"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset files changed"):
        run_lidar_center_overfit(config, resume_from=result["checkpoint"])


@pytest.mark.parametrize("invalid", ["missing_label", "no_positive", "no_points"])
def test_invalid_training_frames_fail_instead_of_producing_fake_success(config, invalid):
    from pathlib import Path

    root = Path(config.data_root)
    if invalid == "missing_label":
        (root / "label_2/000000.txt").unlink()
    elif invalid == "no_positive":
        (root / "label_2/000000.txt").write_text("")
    else:
        np.array([[99., 0, 0, .5]], dtype=np.float32).tofile(root / "velodyne/000000.bin")
    with pytest.raises((FileNotFoundError, ValueError)):
        run_lidar_center_overfit(config)
    assert not Path(config.output_dir).exists()


def test_cuda_preflight_blocks_device_initialization_when_not_ready(config, monkeypatch):
    from rgb_lidar_fusion import camera_depth_analysis

    @dataclass
    class Blocked:
        ready: bool = False

        def format(self):
            return "GPU deliberately unavailable in CPU test"

    monkeypatch.setattr(camera_depth_analysis, "run_gpu_preflight", lambda **kwargs: Blocked())
    def unexpected(*args, **kwargs):
        pytest.fail("CUDA must not be touched after failed preflight")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", unexpected)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    events = []
    with pytest.raises(RuntimeError, match="preflight failed"):
        run_lidar_center_overfit(replace(config, device="cuda"), emit=events.append)
    assert events[-1]["event"] == "failed"


def test_incompatible_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"kind": "camera_depth"}, path)
    with pytest.raises(ValueError, match="not a LiDAR"):
        load_lidar_center_checkpoint(path)


def test_resume_repairs_sidecars_from_checkpoint_history_even_without_more_epochs(config):
    result = run_lidar_center_overfit(replace(config, epochs=1))
    metrics_path = Path(config.output_dir) / "metrics.json"
    metrics_path.write_text("[]")
    restored = run_lidar_center_overfit(replace(config, epochs=1), resume_from=result["checkpoint"])
    assert restored["metrics"] == json.loads(metrics_path.read_text()) == result["metrics"]
