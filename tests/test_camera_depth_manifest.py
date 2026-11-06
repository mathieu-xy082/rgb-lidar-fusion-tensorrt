import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion import training
from rgb_lidar_fusion.camera_depth_analysis import camera_depth_validation_indices, load_camera_depth_checkpoint
from rgb_lidar_fusion.camera_depth_manifest import (
    CAMERA_DEPTH_CHECKPOINT_KIND, VerifiedCameraDepthDataset, validate_camera_depth_manifest,
)
from rgb_lidar_fusion.dataset import KittiObjectDepthDataset


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "data"
    for folder in ("image_2", "velodyne", "calib"):
        (root / folder).mkdir(parents=True)
    for index in range(4):
        sample_id = f"{index:06d}"
        Image.fromarray(np.full((12, 16, 3), 120, dtype=np.uint8)).save(root / "image_2" / f"{sample_id}.png")
        np.array([[-3., -1., 6. + index, .5], [-1.5, 0., 6.5 + index, .6],
                  [0., 0., 7. + index, .7], [1.5, 0., 7.5 + index, .8],
                  [3., 1., 8. + index, .9]], dtype=np.float32).tofile(root / "velodyne" / f"{sample_id}.bin")
        (root / "calib" / f"{sample_id}.txt").write_text(
            "P2: 8 0 8 0 0 8 6 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        )
    config = {
        "dataset": "kitti_camera_depth", "data_root": str(root), "sample_count": 4,
        "height": 12, "width": 16, "device": "cpu", "seed": 13,
        "epochs": 1, "batch_size": 2, "holdout_fraction": .4, "validation_fraction": .5,
        "output_dir": str(tmp_path / "run"), "splat_radius_px": 1,
    }
    result = training.run_kitti_camera_depth_training(config)
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    return config, result, payload


def _before_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid resume must fail before device initialization")
    monkeypatch.setattr(training, "set_deterministic_seed", forbidden)
    monkeypatch.setattr(training, "select_device", forbidden)


def test_manifest_is_embedded_and_written_with_actual_ids_triplet_hashes_and_protocol(run):
    config, result, payload = run
    manifest = payload["data_manifest"]
    assert payload["kind"] == CAMERA_DEPTH_CHECKPOINT_KIND
    assert manifest == json.loads((result.checkpoint_path.parent.parent / "split.json").read_text())
    assert manifest["source_ids"] == ["000000", "000001", "000002", "000003"]
    assert set(manifest["train_ids"]).isdisjoint(manifest["validation_ids"])
    assert manifest["train_ids"] == [manifest["source_ids"][i] for i in manifest["train_indices"]]
    assert manifest["validation_ids"] == [manifest["source_ids"][i] for i in manifest["validation_indices"]]
    assert set(manifest["fingerprints"]["000000"]) == {"image_2", "velodyne", "calib"}
    assert manifest["protocol"]["batch_size"] == 2
    assert manifest["protocol"]["holdout_options"]["holdout_mode"] == "random_points"
    assert manifest["protocol"]["max_depth_m"] == 80.
    metadata = json.loads((Path(config["output_dir"]) / "run_metadata.json").read_text())
    assert metadata["split"]["train_ids"] == manifest["train_ids"]


@pytest.mark.parametrize("folder,suffix", [("image_2", "png"), ("velodyne", "bin"), ("calib", "txt")])
def test_changed_file_is_rejected_before_gpu_or_checkpoint_overwrite(run, monkeypatch, folder, suffix):
    config, result, _ = run
    checkpoint_bytes = result.checkpoint_path.read_bytes()
    path = Path(config["data_root"]) / folder / f"000000.{suffix}"
    path.write_bytes(path.read_bytes() + b"changed")
    _before_gpu(monkeypatch)
    with pytest.raises(ValueError, match="file hashes changed"):
        training.run_kitti_camera_depth_training({**config, "epochs": 2, "resume_from": str(result.checkpoint_path)})
    assert result.checkpoint_path.read_bytes() == checkpoint_bytes


@pytest.mark.parametrize("change", [{"seed": 14}, {"split_seed": 14}, {"holdout_fraction": .3},
                                    {"batch_size": 1}, {"max_depth_m": 40.}, {"splat_radius_px": 2},
                                    {"smooth_l1_beta": .1}, {"validation_holdout_seed": 1000}])
def test_changed_resume_protocol_is_rejected_before_gpu(run, monkeypatch, change):
    config, result, _ = run
    _before_gpu(monkeypatch)
    with pytest.raises(ValueError, match="protocol changed"):
        training.run_kitti_camera_depth_training({**config, **change, "epochs": 2,
                                                   "resume_from": str(result.checkpoint_path)})


def test_changed_frame_membership_is_rejected_even_with_same_frame_count(run, monkeypatch):
    config, result, _ = run
    root = Path(config["data_root"])
    for folder, suffix in (("image_2", "png"), ("velodyne", "bin"), ("calib", "txt")):
        (root / folder / f"000000.{suffix}").rename(root / folder / f"000004.{suffix}")
    _before_gpu(monkeypatch)
    with pytest.raises(ValueError, match="IDs or order changed"):
        training.run_kitti_camera_depth_training({**config, "epochs": 2, "resume_from": str(result.checkpoint_path)})


def test_resume_uses_recorded_indices_without_rerolling_split_and_matches_continuous_run(run, tmp_path, monkeypatch):
    config, result, _ = run
    complete = training.run_kitti_camera_depth_training({**config, "epochs": 2, "output_dir": str(tmp_path / "full")})
    def forbidden(*args, **kwargs):
        raise AssertionError("saved membership must not be regenerated")
    monkeypatch.setattr(training, "split_dataset_indices", forbidden)
    resumed = training.run_kitti_camera_depth_training({**config, "epochs": 2, "resume_from": str(result.checkpoint_path)})
    expected = torch.load(complete.checkpoint_path, map_location="cpu", weights_only=True)
    actual = torch.load(resumed.checkpoint_path, map_location="cpu", weights_only=True)
    assert expected["data_manifest"] == actual["data_manifest"]
    assert complete.metrics[-1] == resumed.metrics[-1]
    for name, tensor in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][name], tensor, rtol=0, atol=0)


def test_legacy_checkpoint_remains_readable_but_cannot_be_resumed_as_verified(run, monkeypatch):
    config, result, payload = run
    payload.pop("kind")
    payload.pop("data_manifest")
    torch.save(payload, result.checkpoint_path)
    loaded = load_camera_depth_checkpoint(result.checkpoint_path, device="cpu")
    assert loaded.provenance_status == "legacy_unverified" and loaded.data_manifest is None
    dataset = KittiObjectDepthDataset(config["data_root"], image_shape=(12, 16), sample_limit=4)
    with pytest.warns(RuntimeWarning, match="not verified historical"):
        assert camera_depth_validation_indices(loaded, dataset)
    _before_gpu(monkeypatch)
    with pytest.raises(ValueError, match="legacy camera checkpoint"):
        training.run_kitti_camera_depth_training({**config, "epochs": 2, "resume_from": str(result.checkpoint_path)})
    assert "kind" not in torch.load(result.checkpoint_path, weights_only=True)


def test_fresh_run_refuses_existing_output_before_gpu(run, monkeypatch):
    _before_gpu(monkeypatch)
    with pytest.raises(ValueError, match="output_dir is not empty"):
        training.run_kitti_camera_depth_training(run[0])


def test_recorded_validation_indices_verify_data_and_do_not_depend_on_splitter(run, monkeypatch):
    import rgb_lidar_fusion.camera_depth_analysis as analysis
    config, result, payload = run
    loaded = load_camera_depth_checkpoint(result.checkpoint_path, device="cpu")
    assert loaded.provenance_status == "recorded_at_training"
    dataset = KittiObjectDepthDataset(config["data_root"], image_shape=(12, 16), sample_limit=4)
    def forbidden(*args, **kwargs):
        raise AssertionError("new checkpoint validation must use saved membership")
    monkeypatch.setattr(analysis, "split_dataset_indices", forbidden)
    assert camera_depth_validation_indices(loaded, dataset) == payload["data_manifest"]["validation_indices"]
    dataset.samples.reverse()
    with pytest.raises(ValueError, match="IDs or order changed"):
        camera_depth_validation_indices(loaded, dataset)


def test_verified_dataset_catches_mutation_during_sample_read(run, monkeypatch):
    config, _, payload = run
    dataset = KittiObjectDepthDataset(config["data_root"], image_shape=(12, 16), sample_limit=4)
    original = KittiObjectDepthDataset.__getitem__
    def changed(self, index):
        item = original(self, index)
        path = self.samples[index][3]
        path.write_text(path.read_text() + "\n")
        return item
    monkeypatch.setattr(KittiObjectDepthDataset, "__getitem__", changed)
    with pytest.raises(ValueError, match="changed after reading"):
        VerifiedCameraDepthDataset(dataset, payload["data_manifest"])[0]


@pytest.mark.parametrize("change", ["overlap", "id_binding", "hash"])
def test_malformed_manifest_is_rejected(run, change):
    manifest = json.loads(json.dumps(run[2]["data_manifest"]))
    if change == "overlap":
        manifest["validation_indices"] = manifest["train_indices"]
        manifest["validation_ids"] = manifest["train_ids"]
    elif change == "id_binding":
        manifest["train_ids"].reverse()
    else:
        manifest["fingerprints"]["000000"]["calib"] = "invalid"
    with pytest.raises(ValueError):
        validate_camera_depth_manifest(manifest)
