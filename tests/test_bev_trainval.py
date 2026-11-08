from dataclasses import dataclass, replace
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_bev_pilot import config as config
from test_bev_training import config as overfit_config
from rgb_lidar_fusion import bev_trainval as training
from rgb_lidar_fusion.bev_checkpoint import load_bev_fusion_checkpoint
from rgb_lidar_fusion.bev_data import BEVFusionFrameSource
from rgb_lidar_fusion.bev_pilot import read_bev_pilot_split
from rgb_lidar_fusion.bev_training import prepare_bev_overfit_frames, read_bev_overfit_split


def _source(config):
    return BEVFusionFrameSource(config.data_root, model_config=config.model_config,
                                 split=read_bev_pilot_split(config))


def _refresh_label_hash(source, sample_id, contents):
    path = Path(source.root) / "label_2" / f"{sample_id}.txt"
    path.write_text(contents)
    source.split["fingerprints"][sample_id]["label_2"] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_streaming_inputs_and_native_targets_match_existing_overfit(config, overfit_config):
    source = _source(config)
    old_frames = prepare_bev_overfit_frames(overfit_config, read_bev_overfit_split(overfit_config))
    for old in old_frames:
        new = source.read(old.sample_id)
        for name, value in old.inputs.items():
            torch.testing.assert_close(value, new.inputs[name], rtol=0, atol=0)
        for name, value in old.targets.items():
            torch.testing.assert_close(value, new.targets[name], rtol=0, atol=0)
        np.testing.assert_array_equal(old.centers_xy, new.centers_xy)


def test_streaming_keeps_no_decoded_frame_cache(config, monkeypatch):
    source = _source(config)
    seen = []
    original = source.read
    def read(sample_id):
        value = original(sample_id)
        seen.append(weakref.ref(value))
        return value
    monkeypatch.setattr(source, "read", read)
    iterator = source.frames(source.split["source_ids"])
    assert not seen
    frame = next(iterator)
    assert len(seen) == 1
    del frame
    next(iterator)
    assert seen[0]() is None and len(seen) == 2


def test_no_car_negatives_empty_cloud_and_missing_labels(config):
    source = _source(config)
    sample_id = source.split["train_ids"][0]
    positive = source.read(sample_id)
    _refresh_label_hash(source, sample_id, "")
    negative = source.read(sample_id)
    assert not len(negative.centers_xy) and negative.targets["supervision_mask"].any()
    assert not negative.targets["center_mask"].any()
    for name, value in positive.inputs.items():
        torch.testing.assert_close(value, negative.inputs[name], rtol=0, atol=0)
    cloud = Path(config.data_root) / "velodyne" / f"{sample_id}.bin"
    cloud.write_bytes(b"")
    source.split["fingerprints"][sample_id]["velodyne"] = hashlib.sha256(b"").hexdigest()
    empty = source.read(sample_id)
    assert not empty.inputs["point_mask"].any() and not empty.inputs["point_count"].any()
    (Path(config.data_root) / "label_2" / f"{sample_id}.txt").unlink()
    with pytest.raises(FileNotFoundError):
        source.read(sample_id)


def test_in_read_mutation_is_rejected(config, monkeypatch):
    source = _source(config)
    sample_id = source.split["train_ids"][0]
    original = type(source.dataset).__getitem__
    def changed(self, index):
        item = original(self, index)
        path = Path(config.data_root) / "calib" / f"{sample_id}.txt"
        path.write_text(path.read_text() + "\n")
        return item
    monkeypatch.setattr(type(source.dataset), "__getitem__", changed)
    with pytest.raises(ValueError, match="after reading"):
        source.read(sample_id)


@pytest.mark.parametrize("variant", ["fusion", "lidar_only"])
def test_only_train_frames_enter_optimizer_and_evaluation_preserves_model(config, monkeypatch, variant):
    updated = []
    original_transfer = training._on_device
    original_evaluate = training._evaluate
    def transfer(frame, device):
        updated.append(frame.sample_id)
        return original_transfer(frame, device)
    def evaluate(model, *args, **kwargs):
        before = {name: value.clone() for name, value in model.state_dict().items()}
        was_training = model.training
        result = original_evaluate(model, *args, **kwargs)
        assert model.training == was_training and not model.pfn.training
        for name, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        return result
    monkeypatch.setattr(training, "_on_device", transfer)
    monkeypatch.setattr(training, "_evaluate", evaluate)
    events = []
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    result = training.run_bev_fusion_pilot(config, variant=variant, emit=events.append)
    model, payload = load_bev_fusion_checkpoint(result["checkpoint"])
    assert torch.get_num_threads() == threads
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    assert len(result["metrics"]) == config.epochs + 1
    split = payload["split"]
    assert len(updated) == config.epochs * len(split["train_ids"])
    assert set(updated) == set(split["train_ids"]) and not set(updated) & set(split["validation_ids"])
    assert payload["step"] == len(updated)
    assert payload["training_metadata"]["task"] == "bev_fusion_pilot"
    assert payload["training_metadata"]["variant"] == variant
    assert payload["training_metadata"]["depth_holdout"] is False
    assert result["resources"]["peak_allocated_mib"] is None
    assert model.config.camera_enabled == (variant == "fusion")
    camera = torch.load(config.camera_checkpoint, map_location="cpu", weights_only=True)
    lidar = torch.load(config.lidar_checkpoint, map_location="cpu", weights_only=True)
    for name, value in model.state_dict().items():
        if name.startswith("camera."):
            expected = camera["model_state_dict"][name.removeprefix("camera.")]
        elif name.startswith(("pfn.", "context.")):
            expected = lidar["model_state_dict"][name]
        else:
            continue
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    best_model, best = load_bev_fusion_checkpoint(result["best_checkpoint"])
    row = min(result["metrics"], key=lambda r: r["val_loss"])
    assert best["purpose"] == "inference" and "optimizer_state_dict" not in best
    assert best["epoch"] == row["epoch"] and result["best_epoch"] == row["epoch"]
    assert best["metrics"][-1]["val_loss"] == row["val_loss"]
    for name, value in best_model.state_dict().items():
        torch.testing.assert_close(value, payload["best"]["model_state_dict"][name], rtol=0, atol=0)
    assert events[-1]["event"] == "completed"
    assert all(event["event"] != "preflight" for event in events)


@pytest.mark.parametrize("variant", ["fusion", "lidar_only"])
def test_exact_resume_without_original_weights_and_best_sidecar_recovery(config, tmp_path, variant):
    whole = training.run_bev_fusion_pilot(config, variant=variant)
    partial_config = replace(config, output_dir=str(tmp_path / "resumed"), epochs=1)
    partial = training.run_bev_fusion_pilot(partial_config, variant=variant)
    Path(config.camera_checkpoint).unlink()
    Path(config.lidar_checkpoint).unlink()
    output = partial_config.output_for_variant(variant)
    (output / "metrics.json").write_text("[]")
    (output / "checkpoints/best.pt").unlink()
    resumed = training.run_bev_fusion_pilot(replace(partial_config, epochs=config.epochs),
                                           variant=variant, resume_from=partial["checkpoint"])
    first, _ = load_bev_fusion_checkpoint(whole["checkpoint"])
    second, _ = load_bev_fusion_checkpoint(resumed["checkpoint"])
    assert whole["metrics"] == resumed["metrics"]
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert json.loads((output / "metrics.json").read_text()) == resumed["metrics"]
    _, best = load_bev_fusion_checkpoint(resumed["best_checkpoint"])
    assert best["epoch"] == resumed["best_epoch"]


def test_protocol_variant_purpose_history_and_best_validation_before_cuda(config, monkeypatch):
    from rgb_lidar_fusion import camera_depth_analysis
    result = training.run_bev_fusion_pilot(replace(config, epochs=1), variant="fusion")
    path = Path(result["checkpoint"])
    original = torch.load(path, map_location="cpu", weights_only=True)
    def unexpected(**kwargs):
        pytest.fail("preflight must not run on an incompatible resume")
    monkeypatch.setattr(camera_depth_analysis, "run_gpu_preflight", unexpected)
    changes = (
        ("task", lambda p: p["training_metadata"].update(task="bev_fusion_overfit")),
        ("variant", lambda p: p["training_metadata"].update(variant="lidar_only")),
        ("configuration", lambda p: p["training_metadata"]["runner_config"].update(learning_rate=.02)),
        ("purpose", lambda p: p.update(purpose="inference")),
        ("history", lambda p: p.update(metrics=p["metrics"][:-1])),
        ("best", lambda p: p["best"].update(val_loss=p["best"]["val_loss"] + 1)),
    )
    for error, change in changes:
        payload = copy.deepcopy(original)
        change(payload)
        torch.save(payload, path)
        with pytest.raises(ValueError, match=error):
            training.run_bev_fusion_pilot(replace(config, device="cuda"), variant="fusion", resume_from=path)
    torch.save(original, path)
    with pytest.raises(ValueError, match="latest.pt"):
        training.run_bev_fusion_pilot(config, variant="fusion", resume_from=result["best_checkpoint"])
    with pytest.raises(ValueError, match="not empty"):
        training.run_bev_fusion_pilot(config, variant="fusion")


def test_changed_dataset_is_rejected_on_resume_before_preflight(config, monkeypatch):
    from rgb_lidar_fusion import camera_depth_analysis
    result = training.run_bev_fusion_pilot(replace(config, epochs=1), variant="fusion")
    label = Path(config.data_root) / "label_2/000000.txt"
    label.write_text(label.read_text() + "\n")
    def unexpected(**kwargs):
        pytest.fail("changed data must fail before preflight")
    monkeypatch.setattr(camera_depth_analysis, "run_gpu_preflight", unexpected)
    with pytest.raises(ValueError, match="file changed"):
        training.run_bev_fusion_pilot(replace(config, device="cuda"), variant="fusion", resume_from=result["checkpoint"])


def test_preflight_failure_has_no_cuda_cleanup_or_retry(config, monkeypatch):
    from rgb_lidar_fusion import camera_depth_analysis
    @dataclass
    class Blocked:
        ready: bool = False
        def format(self):
            return "GPU deliberately unavailable"
    monkeypatch.setattr(camera_depth_analysis, "run_gpu_preflight", lambda **kwargs: Blocked())
    def unexpected(*args, **kwargs):
        pytest.fail("CUDA must not be touched after preflight failure")
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, unexpected)
    with pytest.raises(RuntimeError, match="preflight failed"):
        training.run_bev_fusion_pilot(replace(config, device="cuda"), variant="fusion")
    assert not config.output_for_variant("fusion").exists()


def test_unknown_supervision_is_not_fake_success(config, monkeypatch):
    original = BEVFusionFrameSource.read
    def unknown(self, sample_id):
        frame = original(self, sample_id)
        return replace(frame, targets={key: torch.zeros_like(value) for key, value in frame.targets.items()},
                       centers_xy=np.empty((0, 2), dtype=np.float32))
    monkeypatch.setattr(BEVFusionFrameSource, "read", unknown)
    with pytest.raises(ValueError, match="no supervised frames"):
        training.run_bev_fusion_pilot(config, variant="fusion")
    assert not config.output_for_variant("fusion").exists()


def test_negative_updates_and_unknown_skips_are_counted_separately(config, monkeypatch):
    split = read_bev_pilot_split(config)
    negative_id, unknown_id = split["train_ids"][:2]
    original = BEVFusionFrameSource.read
    def prepare(self, sample_id):
        frame = original(self, sample_id)
        if sample_id in (negative_id, unknown_id):
            targets = {key: torch.zeros_like(value) for key, value in frame.targets.items()}
            if sample_id == negative_id:
                targets["supervision_mask"] = frame.targets["supervision_mask"]
            return replace(frame, targets=targets, centers_xy=np.empty((0, 2), dtype=np.float32))
        return frame
    monkeypatch.setattr(BEVFusionFrameSource, "read", prepare)
    result = training.run_bev_fusion_pilot(replace(config, epochs=1), variant="fusion")
    final = result["metrics"][-1]
    assert final["train_skipped_update_ids"] == [unknown_id]
    assert final["train_negative_update_frames"] == 1
    assert final["train_negative_frames"] == 1
    assert final["step"] == len(split["train_ids"]) - 1


def test_resume_without_more_epochs_repairs_metrics_and_best(config):
    result = training.run_bev_fusion_pilot(replace(config, epochs=1), variant="fusion")
    output = config.output_for_variant("fusion")
    (output / "metrics.json").write_text("[]")
    (output / "checkpoints/best.pt").unlink()
    resumed = training.run_bev_fusion_pilot(replace(config, epochs=1), variant="fusion",
                                           resume_from=result["checkpoint"])
    assert resumed["metrics"] == result["metrics"]
    assert json.loads((output / "metrics.json").read_text()) == result["metrics"]
    _, payload = load_bev_fusion_checkpoint(output / "checkpoints/best.pt")
    assert payload["epoch"] == result["best_epoch"]


@pytest.mark.parametrize("invalid", ["counter", "nonfinite", "missing_weights", "shape"])
def test_best_snapshot_is_validated_when_loading(config, invalid):
    result = training.run_bev_fusion_pilot(replace(config, epochs=1), variant="fusion")
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    best = payload["best"]
    if invalid == "counter":
        best["epoch"] = payload["epoch"] + 1
    elif invalid == "nonfinite":
        best["val_loss"] = float("nan")
    elif invalid == "missing_weights":
        best["model_state_dict"] = {}
    else:
        best["model_state_dict"]["head.heatmap.bias"] = torch.zeros(2)
    torch.save(payload, result["checkpoint"])
    with pytest.raises(ValueError):
        load_bev_fusion_checkpoint(result["checkpoint"])


def test_cli_streams_a_cpu_variant(config, tmp_path):
    config_path = tmp_path / "pilot.json"
    config_path.write_text(json.dumps(replace(config, epochs=1).serialized()))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/train_bev_pilot.py"), "--config", str(config_path),
         "--variant", "lidar_only"], cwd=root, env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["event"] == "completed" and events[-1]["variant"] == "lidar_only"
    assert events[-1]["epochs_completed"] == 1
