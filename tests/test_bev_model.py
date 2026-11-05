from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion.bev_batch import prepare_bev_fusion_batch
from rgb_lidar_fusion.bev_checkpoint import (
    CHECKPOINT_KIND, initialize_bev_fusion_encoders, load_bev_fusion_checkpoint,
    make_bev_fusion_optimizer, restore_bev_fusion_training, save_bev_fusion_checkpoint,
    validate_bev_fusion_split,
)
from rgb_lidar_fusion.bev_model import BEVFusionConfig, BEVFusionDiagnostic
from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.camera_depth_model import CameraDepthModel
from rgb_lidar_fusion.dataset import KittiObjectDepthDataset
from rgb_lidar_fusion.lidar_center_model import LidarCenterDiagnostic, lidar_center_loss
from rgb_lidar_fusion.lidar_center_training import LidarCenterTrainConfig
from rgb_lidar_fusion.lidar_center_trainval import build_lidar_center_split
from rgb_lidar_fusion.project_lidar import load_velodyne_bin


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "data"
    for name in ("image_2", "calib", "velodyne", "label_2"):
        (root / name).mkdir(parents=True)
    for index in range(8):
        sample_id = f"{index:06d}"
        Image.fromarray(np.full((20, 24, 3), 120 + index, dtype=np.uint8)).save(root / "image_2" / f"{sample_id}.png")
        (root / "calib" / f"{sample_id}.txt").write_text(
            "P2: 10 0 12 0 0 10 10 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        )
        (root / "label_2" / f"{sample_id}.txt").write_text("Car 0 0 0 6 6 14 14 2 2 4 0 1 5 0\n")
        np.array([[5., 0., -.2, .8], [5.1, .1, .2, .5], [6., .5, .1, .9]], dtype=np.float32).tofile(
            root / "velodyne" / f"{sample_id}.bin",
        )
    config = BEVFusionConfig(grid=BEVGridConfig((0., 10.), (-4., 4.), (-3., 3.), 1.),
                             image_shape=(16, 24), max_points_per_pillar=4,
                             lidar_channels=2, backbone_channels=(2, 3, 4))
    dataset = KittiObjectDepthDataset(root, image_shape=config.image_shape)
    items = [dataset[0], dataset[1]]
    clouds = {item["meta"]["sample_id"]: load_velodyne_bin(item["meta"]["lidar_path"]) for item in items}
    split_config = LidarCenterTrainConfig(data_root=str(root), sample_count=8, validation_fraction=.25,
                                         grid=config.grid, channels=2, max_points_per_pillar=4, device="cpu")
    split = build_lidar_center_split(split_config)
    return config, items, clouds, split, split_config


def _batch(case):
    config, items, clouds, *_ = case
    return prepare_bev_fusion_batch(items, clouds, config=config)


def _loss(prediction):
    heatmap = torch.zeros_like(prediction.heatmap_logits)
    heatmap[:, :, 5, 4] = 1
    offsets = torch.zeros_like(prediction.offset_xy)
    offsets[:, :, 5, 4] = .25
    return lidar_center_loss(prediction, heatmap=heatmap, offset_xy=offsets,
                             center_mask=heatmap.bool(), supervision_mask=torch.ones_like(heatmap, dtype=torch.bool)).total


def _sources(case, tmp_path):
    config, _, _, split, lidar_config = case
    camera = CameraDepthModel(lidar_mode="enriched")
    camera_config = {"dataset": "kitti_camera_depth", "lidar_mode": "enriched",
                     "height": config.image_shape[0], "width": config.image_shape[1],
                     "max_depth_m": config.max_depth_m, "splat_radius_px": config.splat_radius_px,
                     "splat_sigma_px": config.splat_sigma_px, "seed": 13, "split_seed": 13,
                     "validation_fraction": .25, "sample_count": 8}
    camera_path = tmp_path / "camera.pt"
    torch.save({"model_state_dict": camera.state_dict(), "config": camera_config,
                "epoch": 5, "step": 12, "optimizer_state_dict": {"unused": True}}, camera_path)
    lidar = LidarCenterDiagnostic(channels=2)
    lidar_state = {k: v for k, v in lidar.state_dict().items() if not k.startswith("head.")}
    lidar_path = tmp_path / "lidar.pt"
    torch.save({"kind": "lidar_center_trainval_v1", "model_state_dict": lidar_state,
                "split": split, "config": lidar_config.serialized(), "epoch": 9, "step": 18}, lidar_path)
    return camera_path, lidar_path


def _initialize(model, paths, split):
    return initialize_bev_fusion_encoders(model, camera_checkpoint=paths[0], lidar_checkpoint=paths[1],
                                          split=split, camera_source_ids=split["source_ids"])


def test_default_configuration_file_and_mapping_roundtrip():
    values = json.loads(Path("configs/bev_fusion.json").read_text())
    config = BEVFusionConfig.from_mapping(values)
    assert config == BEVFusionConfig()
    assert config.serialized() == values
    assert BEVFusionConfig.from_mapping(config.serialized()).serialized() == values


@pytest.mark.parametrize("values", [
    {"image_shape": (0, 2)}, {"image_shape": (2,)}, {"image_shape": (True, 3)},
    {"backbone_channels": (1, 2)}, {"backbone_channels": (1, 2, -1)},
    {"max_points_per_pillar": 0}, {"max_points_per_pillar": True}, {"pillar_seed": -1},
    {"lidar_channels": 0}, {"splat_radius_px": -1}, {"splat_radius_px": 1.5},
    {"splat_sigma_px": 0.}, {"max_depth_m": float("nan")}, {"max_depth_m": 0.},
    {"min_splat_confidence": 1.1}, {"min_splat_confidence": float("inf")},
    {"freeze_encoders": 1}, {"camera_enabled": "false"}, {"grid": None},
])
def test_invalid_model_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        BEVFusionConfig(**values)


def test_unknown_config_keys_and_model_config_type_are_rejected():
    with pytest.raises(ValueError, match="invalid BEV fusion configuration"):
        BEVFusionConfig.from_mapping({"unexpected": 1})
    with pytest.raises(ValueError, match="config must be BEVFusionConfig"):
        BEVFusionDiagnostic(False)


def test_batch_preserves_sample_order_and_rebuilds_enriched_maps_without_targets(case):
    config, items, clouds, *_ = case
    before = [item["lidar_maps"].copy() for item in items]
    batch = prepare_bev_fusion_batch(items[::-1], clouds, config=config)
    assert batch.sample_ids == ("000001", "000000")
    assert batch.inputs["rgb"].shape == (2, 3, 16, 24)
    assert batch.inputs["lidar_maps"].shape == (2, 8, 16, 24)
    assert batch.inputs["pillar_features"].shape[1:] == (4, 9)
    assert all(value.device.type == "cpu" for value in batch.inputs.values())
    assert (batch.inputs["lidar_maps"][:, 7] > 0).sum() > batch.inputs["lidar_maps"][:, 5].sum()
    for item, saved in zip(items, before, strict=True):
        np.testing.assert_array_equal(item["lidar_maps"], saved)
        item["target"] = {"depth_target": float("nan"), "labels": ["arbitrary"]}
    other = prepare_bev_fusion_batch(items[::-1], clouds, config=config)
    for name in batch.inputs:
        torch.testing.assert_close(batch.inputs[name], other.inputs[name], rtol=0, atol=0)


@pytest.mark.parametrize("case_name", ["missing_cloud", "id", "scale", "shape", "splatted"])
def test_bad_batch_contract_is_rejected(case, case_name):
    config, items, clouds, *_ = case
    if case_name == "missing_cloud":
        clouds.pop("000000")
    elif case_name == "id":
        items[0]["meta"]["sample_id"] = "wrong"
    elif case_name == "scale":
        items[0]["max_depth_m"] = 40.
    elif case_name == "shape":
        items[0]["image"] = items[0]["image"][:, :1]
    else:
        items[0]["meta"]["lidar_representation"] = "splatted"
    with pytest.raises(ValueError):
        prepare_bev_fusion_batch(items, clouds, config=config)


def test_fractional_normalization_scale_matches_float32_dataset_metadata(case):
    config, items, clouds, *_ = case
    config = replace(config, max_depth_m=80.1)
    for item in items:
        item["max_depth_m"] = np.float32(80.1)
    assert prepare_bev_fusion_batch(items, clouds, config=config).sample_ids == ("000000", "000001")


def test_empty_cloud_and_zero_sparse_maps_keep_empty_evidence(case):
    config, items, clouds, *_ = case
    for item in items:
        item["lidar_maps"].fill(0)
        clouds[item["meta"]["sample_id"]] = np.empty((0, 4), dtype=np.float32)
    batch = prepare_bev_fusion_batch(items, clouds, config=config)
    with torch.no_grad():
        result = BEVFusionDiagnostic(config).eval().encode(**batch.inputs)
    assert not result.shared_bev.camera_support.any() and not result.shared_bev.lidar_observed_mask.any()
    assert torch.all(result.shared_bev.features == 0)


def test_raw_lidar_outside_camera_fov_is_preserved(case):
    config, items, clouds, *_ = case
    # P2 gives u = 12 - 10 * 3.8 / 1 = -26, outside the native image.
    clouds["000000"] = np.concatenate([clouds["000000"], np.array([[1., 3.8, 0., .7]], dtype=np.float32)])
    batch = prepare_bev_fusion_batch(items, clouds, config=config)
    with torch.no_grad():
        result = BEVFusionDiagnostic(config).eval().encode(**batch.inputs)
    assert result.lidar_bev.point_count[0].sum() == 4
    assert result.lidar_bev.point_count[0, 0, 1, 7] == 1


def test_frozen_encoders_and_bn_stay_unchanged_through_training_step(case):
    config = case[0]
    model = BEVFusionDiagnostic(config).train()
    assert model.training and model.fusion.training and model.backbone.training and model.head.training
    assert not model.camera.training and not model.pfn.training and not model.context.training
    batch = _batch(case)
    before = {n: value.clone() for n, value in model.state_dict().items()
              if n.startswith(("camera.", "pfn.", "context."))}
    optimizer = make_bev_fusion_optimizer(model)
    assert all(n.startswith(("fusion.", "backbone.", "head.")) for n, p in model.named_parameters() if p.requires_grad)
    loss = _loss(model(**batch.inputs))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert all(p.grad is None for module in (model.camera, model.pfn, model.context) for p in module.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fusion.parameters())
    model.eval().train()
    assert not model.pfn.training and not model.camera.training


def test_unfrozen_end_to_end_backward_and_no_depth_head_call(case):
    config = replace(case[0], freeze_encoders=False)
    model = BEVFusionDiagnostic(config).train()
    with torch.no_grad():
        for module in (model.camera, model.pfn, model.context, model.fusion, model.backbone):
            for parameter in module.parameters():
                parameter.fill_(.002)
    def forbidden(*args):
        raise AssertionError("depth_head must not be evaluated")
    handle = model.camera.depth_head.register_forward_hook(forbidden)
    try:
        batch = _batch(case)
        loss = _loss(model(**batch.inputs))
        assert torch.isfinite(loss)
        loss.backward()
    finally:
        handle.remove()
    for module in (model.camera.stem, model.camera.encoder_1, model.camera.encoder_2,
                   model.camera.decoder_1, model.camera.decoder_0, model.pfn, model.context,
                   model.fusion, model.backbone, model.head):
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0
    assert all(p.grad is None and not p.requires_grad for p in model.camera.depth_head.parameters())


def test_encode_exposes_evidence_and_does_not_call_diagnostic_head(case):
    model = BEVFusionDiagnostic(case[0]).eval()
    batch = _batch(case)
    def forbidden(*args):
        raise AssertionError("diagnostic head must not run in encode")
    handle = model.head.register_forward_hook(forbidden)
    try:
        with torch.no_grad():
            representations = model.encode(**batch.inputs)
    finally:
        handle.remove()
    assert representations.camera_bev.features.shape == (2, 32, 10, 8)
    assert representations.lidar_bev.features.shape == (2, 2, 10, 8)
    assert representations.shared_bev.features.shape == (2, 2, 10, 8)
    assert representations.camera_bev.pixel_count.sum() > 0
    assert torch.equal(representations.lidar_bev.point_count, batch.inputs["point_count"])
    assert torch.equal(representations.shared_bev.lidar_point_count, batch.inputs["point_count"])
    assert torch.equal(representations.shared_bev.camera_pixel_count, representations.camera_bev.pixel_count)


def test_lidar_only_control_skips_camera_without_removing_lidar(case):
    model = BEVFusionDiagnostic(replace(case[0], camera_enabled=False, freeze_encoders=False)).train()
    def forbidden(*args):
        raise AssertionError("camera must not run in LiDAR-only control")
    handle = model.camera.register_forward_pre_hook(forbidden)
    model.camera.forward_features = forbidden
    try:
        representations = model.encode(**_batch(case).inputs)
    finally:
        handle.remove()
    assert not model.camera.training and not representations.shared_bev.camera_support.any()
    assert representations.shared_bev.lidar_observed_mask.any()
    assert all(not p.requires_grad for p in model.camera.parameters())


@pytest.mark.parametrize("field", ["rgb", "lidar_maps", "camera_projection", "t_vehicle_to_camera"])
def test_nonfinite_inputs_are_rejected(case, field):
    batch = _batch(case)
    batch.inputs[field][(0,) * batch.inputs[field].ndim] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        BEVFusionDiagnostic(case[0])(**batch.inputs)


@pytest.mark.parametrize("field", ["rgb", "camera_projection", "point_count", "pillar_features"])
def test_input_shape_and_pillar_cap_mismatches_are_rejected(case, field):
    batch = _batch(case)
    if field == "rgb":
        batch.inputs[field] = batch.inputs[field][:, :, :1]
    elif field == "camera_projection":
        batch.inputs[field] = batch.inputs[field][:, :, :3]
    elif field == "point_count":
        batch.inputs[field] = batch.inputs[field][:1]
    else:
        batch.inputs[field] = batch.inputs[field][:, :1]
    with pytest.raises(ValueError):
        BEVFusionDiagnostic(case[0])(**batch.inputs)


def test_checkpoint_warm_start_loads_only_encoders_with_explicit_audit(case, tmp_path):
    model = BEVFusionDiagnostic(case[0])
    paths = _sources(case, tmp_path)
    untouched = {n: p.clone() for n, p in model.state_dict().items()
                 if n.startswith(("fusion.", "backbone.", "head."))}
    report = _initialize(model, paths, case[3])
    assert report["loaded"] == ["camera", "pfn", "context"]
    assert report["source_head_and_optimizer_loaded"] is False
    assert "cannot be independently proven" in report["camera"]["limitation"]
    assert not set(report["camera"]["train_ids"]) & set(case[3]["validation_ids"])
    for name in ("camera", "lidar"):
        with Path(report[name]["path"]).open("rb") as stream:
            assert report[name]["sha256"] == hashlib.file_digest(stream, "sha256").hexdigest()
    for name, value in untouched.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert not make_bev_fusion_optimizer(model).state
    assert not model.pfn.training
    torch.testing.assert_close(model.camera.stem[0].weight,
                               torch.load(paths[0], weights_only=True)["model_state_dict"]["stem.0.weight"])


@pytest.mark.parametrize("change", ["camera_split", "lidar_split", "overfit", "preprocessing", "hashes", "state"])
def test_bad_pretrained_sources_are_rejected_before_any_model_mutation(case, tmp_path, change):
    model = BEVFusionDiagnostic(case[0])
    paths = _sources(case, tmp_path)
    before = {n: t.clone() for n, t in model.state_dict().items()}
    path = paths[0] if change in ("camera_split", "preprocessing") else paths[1]
    payload = torch.load(path, weights_only=True)
    if change == "camera_split":
        payload["config"]["validation_fraction"] = .125
    elif change == "lidar_split":
        train, val = payload["split"]["train_ids"], payload["split"]["validation_ids"]
        train[0], val[0] = val[0], train[0]
    elif change == "overfit":
        payload["kind"] = "lidar_center_overfit_v1"
    elif change == "preprocessing":
        payload["config"]["max_depth_m"] = 40.
    elif change == "hashes":
        payload["split"]["fingerprints"]["000000"]["image_2"] = "0" * 64
    else:
        payload["model_state_dict"]["context.blocks.0.convs.0.weight"].fill_(float("nan"))
    torch.save(payload, path)
    with pytest.raises(ValueError):
        _initialize(model, paths, case[3])
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_wrong_or_unstated_legacy_camera_id_mapping_is_rejected(case, tmp_path):
    paths = _sources(case, tmp_path)
    with pytest.raises(ValueError, match="camera_source_ids"):
        initialize_bev_fusion_encoders(BEVFusionDiagnostic(case[0]), camera_checkpoint=paths[0],
                                       lidar_checkpoint=paths[1], split=case[3], camera_source_ids=[])


def test_new_checkpoint_cpu_roundtrip_does_not_require_dataset_files_for_inference(case, tmp_path):
    model = BEVFusionDiagnostic(case[0]).eval()
    batch = _batch(case)
    with torch.no_grad():
        expected = model(**batch.inputs)
    path = tmp_path / "best.pt"
    save_bev_fusion_checkpoint(path, model=model, split=case[3], epoch=0, step=0)
    for sample_id in case[3]["source_ids"]:
        (Path(case[3]["data_root"]) / "label_2" / f"{sample_id}.txt").unlink()
    rng = torch.random.get_rng_state().clone()
    loaded, payload = load_bev_fusion_checkpoint(path)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert not loaded.training and not loaded.pfn.training
    assert payload["kind"] == CHECKPOINT_KIND and payload["purpose"] == "inference"
    assert "optimizer_state_dict" not in payload
    assert all(p.device.type == "cpu" for p in loaded.parameters())
    with torch.no_grad():
        actual = loaded(**batch.inputs)
    torch.testing.assert_close(actual.heatmap_logits, expected.heatmap_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.offset_xy, expected.offset_xy, rtol=0, atol=0)


def test_checkpoint_config_policy_and_nonfinite_weights_are_checked(case, tmp_path):
    path = tmp_path / "bad.pt"
    model = BEVFusionDiagnostic(case[0])
    save_bev_fusion_checkpoint(path, model=model, split=case[3], epoch=0, step=0)
    payload = torch.load(path, weights_only=True)
    payload["model_state_dict"]["fusion.projection.weight"].fill_(float("nan"))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="must be finite"):
        load_bev_fusion_checkpoint(path)
    save_bev_fusion_checkpoint(path, model=model, split=case[3], epoch=0, step=0)
    payload = torch.load(path, weights_only=True)
    payload["config"]["freeze_encoders"] = False
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trainable parameter policy"):
        load_bev_fusion_checkpoint(path)


def test_exact_cpu_adam_resume_uses_only_new_model_state_and_optimizer(case, tmp_path):
    model = BEVFusionDiagnostic(case[0]).train()
    batch = _batch(case)
    optimizer = make_bev_fusion_optimizer(model, learning_rate=.002)
    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        _loss(model(**batch.inputs)).backward()
        optimizer.step()
    step(model, optimizer)
    path = tmp_path / "latest.pt"
    save_bev_fusion_checkpoint(path, model=model, optimizer=optimizer, split=case[3], epoch=1, step=1)
    step(model, optimizer)
    resumed, _ = load_bev_fusion_checkpoint(path)
    resumed.train()
    resumed_optimizer = make_bev_fusion_optimizer(resumed, learning_rate=.9)
    result = restore_bev_fusion_training(path, model=resumed, optimizer=resumed_optimizer, split=case[3])
    assert result["epoch"] == result["step"] == 1
    assert resumed_optimizer.param_groups[0]["lr"] == .002
    step(resumed, resumed_optimizer)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["inference", "config", "data", "optimizer"])
def test_unsafe_resume_is_rejected(case, tmp_path, change):
    model = BEVFusionDiagnostic(case[0])
    optimizer = make_bev_fusion_optimizer(model)
    path = tmp_path / "latest.pt"
    save_bev_fusion_checkpoint(path, model=model, optimizer=None if change == "inference" else optimizer,
                               split=case[3], epoch=0, step=0)
    if change == "config":
        model = BEVFusionDiagnostic(replace(case[0], camera_enabled=False))
        optimizer = make_bev_fusion_optimizer(model)
    elif change == "data":
        (Path(case[3]["data_root"]) / "label_2/000000.txt").write_text("changed")
    elif change == "optimizer":
        optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError):
        restore_bev_fusion_training(path, model=model, optimizer=optimizer, split=case[3])


def test_old_checkpoint_cannot_be_resumed_or_loaded_as_multimodal(case, tmp_path):
    path, _ = _sources(case, tmp_path)
    with pytest.raises(ValueError, match="not a multimodal"):
        load_bev_fusion_checkpoint(path)


def test_manifest_membership_and_hashes_are_checked(case):
    split = json.loads(json.dumps(case[3]))
    split["validation_ids"][0] = split["train_ids"][0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_bev_fusion_split(split)
