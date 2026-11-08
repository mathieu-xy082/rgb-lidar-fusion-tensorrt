from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from test_bev_training import config as overfit_config
from rgb_lidar_fusion.bev_checkpoint import save_bev_fusion_checkpoint
from rgb_lidar_fusion.bev_model import BEVFusionConfig
from rgb_lidar_fusion.bev_pilot import (
    BEVFusionPilotConfig, bev_pilot_initial_state_sha256,
    prepare_bev_fusion_pilot, prepare_bev_fusion_pilot_pair, read_bev_pilot_split,
)
from rgb_lidar_fusion.bev_training import prepare_bev_overfit_frames, read_bev_overfit_split


@pytest.fixture
def config(overfit_config):
    values = overfit_config.serialized()
    values.pop("sample_ids")
    values.pop("eval_every")
    return BEVFusionPilotConfig.from_mapping({**values, "sample_count": 6, "epochs": 3})


def test_preset_and_json_roundtrip():
    config = BEVFusionPilotConfig.from_mapping(json.loads(Path("configs/training/kitti_bev_fusion_pilot.json").read_text()))
    assert config.epochs == 10 and config.device == "cpu" and config.sample_count == 512
    assert BEVFusionPilotConfig().epochs == config.epochs
    assert config.minimum_free_memory_mib == 2048
    assert config.output_for_variant("fusion") != config.output_for_variant("lidar_only")
    assert BEVFusionPilotConfig.from_mapping(config.serialized()).serialized() == config.serialized()


@pytest.mark.parametrize("values", [
    {"device": "auto"}, {"epochs": 0}, {"sample_count": 1}, {"sample_count": True},
    {"log_every_steps": 0}, {"learning_rate": float("nan")}, {"split_path": ""},
    {"model_config": BEVFusionConfig(freeze_encoders=False)},
    {"model_config": BEVFusionConfig(camera_enabled=False)},
])
def test_invalid_protocol_is_rejected(values):
    with pytest.raises(ValueError):
        BEVFusionPilotConfig(**values)


def test_pair_has_identical_fresh_blocks_sources_and_split_without_gpu_or_outputs(config, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("pilot preparation must not touch CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    rng = torch.random.get_rng_state().clone()
    pair = prepare_bev_fusion_pilot_pair(config)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert set(pair) == {"fusion", "lidar_only"}
    first, second = pair["fusion"], pair["lidar_only"]
    assert first.split == second.split and first.pretraining == second.pretraining
    assert first.initialization["new_blocks_sha256"] == second.initialization["new_blocks_sha256"]
    assert first.model.config.camera_enabled and not second.model.config.camera_enabled
    assert not Path(config.output_dir).exists()
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[name], rtol=0, atol=0)
    assert [n for n, p in first.model.named_parameters() if p.requires_grad] == [
        n for n, p in second.model.named_parameters() if p.requires_grad
    ]
    for start in pair.values():
        start.model.train()
        for module in (start.model.camera, start.model.pfn, start.model.context):
            assert not module.training and all(not p.requires_grad for p in module.parameters())
        assert start.pretraining["camera"]["provenance"] == "recorded_at_training"


def test_lidar_only_zeroes_camera_but_preserves_raw_lidar_context(config, overfit_config, monkeypatch):
    pair = prepare_bev_fusion_pilot_pair(config)
    frames = prepare_bev_overfit_frames(overfit_config, read_bev_overfit_split(overfit_config))
    def unexpected(*args, **kwargs):
        pytest.fail("LiDAR-only control must skip the camera encoder")
    monkeypatch.setattr(pair["lidar_only"].model.camera, "forward_features", unexpected)
    with torch.no_grad():
        baseline = pair["lidar_only"].model.encode(**frames[0].inputs)
        fusion = pair["fusion"].model.encode(**frames[0].inputs)
    assert not baseline.camera_bev.features.any()
    assert not baseline.camera_bev.pixel_count.any() and not baseline.camera_bev.weight_sum.any()
    torch.testing.assert_close(baseline.lidar_bev.features, fusion.lidar_bev.features, rtol=0, atol=0)
    torch.testing.assert_close(baseline.lidar_bev.point_count, fusion.lidar_bev.point_count, rtol=0, atol=0)
    assert baseline.lidar_bev.point_count.any() and fusion.camera_bev.pixel_count.any()


def test_seed_controls_fresh_blocks_and_hash_tracks_weights(config):
    first = prepare_bev_fusion_pilot(config, variant="fusion")
    second = prepare_bev_fusion_pilot(replace(config, seed=config.seed + 1), variant="fusion")
    assert first.initialization["new_blocks_sha256"] != second.initialization["new_blocks_sha256"]
    with torch.no_grad():
        first.model.head.heatmap.bias.add_(1)
    assert bev_pilot_initial_state_sha256(first.model) != first.initialization["new_blocks_sha256"]


def test_saved_population_order_and_hashes_are_required(config):
    with pytest.raises(ValueError, match="sample_count"):
        read_bev_pilot_split(replace(config, sample_count=5))
    split = json.loads(Path(config.split_path).read_text())
    split["source_ids"].reverse()
    Path(config.split_path).write_text(json.dumps(split))
    with pytest.raises(ValueError, match="population/order"):
        read_bev_pilot_split(config)
    image = Path(config.data_root) / "calib/000000.txt"
    image.write_text(image.read_text() + "\n")
    with pytest.raises(ValueError, match="file changed"):
        read_bev_pilot_split(config)


def test_source_membership_must_equal_canonical_split_not_just_avoid_leakage(config):
    camera = torch.load(config.camera_checkpoint, map_location="cpu", weights_only=True)
    manifest = camera["data_manifest"]
    moved = manifest["train_indices"].pop()
    manifest["validation_indices"].append(moved)
    manifest["train_ids"] = [manifest["source_ids"][i] for i in manifest["train_indices"]]
    manifest["validation_ids"] = [manifest["source_ids"][i] for i in manifest["validation_indices"]]
    torch.save(camera, config.camera_checkpoint)
    with pytest.raises(ValueError, match="source train membership"):
        prepare_bev_fusion_pilot(config, variant="fusion")


def test_four_frame_fusion_checkpoint_is_not_an_encoder_source(config, tmp_path):
    start = prepare_bev_fusion_pilot(config, variant="fusion")
    path = tmp_path / "overfit.pt"
    save_bev_fusion_checkpoint(path, model=start.model, split=start.split, epoch=200, step=800,
                               training_metadata={"task": "bev_fusion_overfit"})
    with pytest.raises(ValueError, match="unsupported camera checkpoint"):
        prepare_bev_fusion_pilot(replace(config, camera_checkpoint=str(path)), variant="fusion")
    with pytest.raises(ValueError, match="LiDAR warm start"):
        prepare_bev_fusion_pilot(replace(config, lidar_checkpoint=str(path)), variant="fusion")


def test_unknown_variants_fail_before_reading_sources(config):
    with pytest.raises(ValueError, match="variant"):
        prepare_bev_fusion_pilot(replace(config, split_path="missing.json"), variant="camera_only")


def test_cpu_audit_cli_records_only_preparation(config, tmp_path):
    config_path = tmp_path / "pilot.json"
    config_path.write_text(json.dumps(config.serialized()))
    output = tmp_path / "analysis/preparation.json"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/audit_bev_pilot.py"), "--config", str(config_path),
         "--output", str(output)], cwd=root, env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["task"] == "bev_fusion_pilot_preparation"
    assert report["cpu_only"] and not report["training_started"]
    assert len(report["split"]["source_ids"]) == 6
    assert report["variants"]["fusion"]["initialization"]["new_blocks_sha256"] == (
        report["variants"]["lidar_only"]["initialization"]["new_blocks_sha256"]
    )
    assert not Path(config.output_dir).exists()
