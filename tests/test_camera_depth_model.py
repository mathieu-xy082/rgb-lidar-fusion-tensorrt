from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_depth_model import CameraDepthModel, masked_depth_loss
from rgb_lidar_fusion.lidar_splatting import SplattingConfig
from rgb_lidar_fusion.model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_camera_depth_training_batch,
)
from rgb_lidar_fusion.training import (
    evaluate_camera_depth,
    prepare_camera_depth_evaluation_batch,
    train_camera_depth_step,
)


def _item(offset: int = 0) -> dict:
    image = np.zeros((3, 17, 23), dtype=np.float32)
    image[0, 2:15, 3:20] = 0.25
    image[1, 4:12, 5:18] = 0.5
    sparse = np.zeros((6, 17, 23), dtype=np.float32)
    for index, (y, x) in enumerate(((2, 2), (4, 8), (7, 14), (10, 5), (14, 20))):
        depth = 0.1 * (index + 1) + 0.01 * offset
        sparse[0, y, x] = depth
        sparse[1, y, x] = 0.01 * (index + 1)
        sparse[2, y, x] = -0.01 * index
        sparse[3, y, x] = 0.02 * index
        sparse[4, y, x] = 0.8
        sparse[5, y, x] = 1.0
    return {
        "image": image,
        "lidar_maps": sparse,
        "target": {},
        "meta": {"sample_id": f"depth-{offset}"},
    }


def test_camera_depth_model_preserves_odd_input_resolution() -> None:
    model = CameraDepthModel()
    rgb = torch.zeros((2, 3, 31, 47), dtype=torch.float32)
    lidar_maps = torch.zeros((2, 8, 31, 47), dtype=torch.float32)

    prediction = model(rgb, lidar_maps)

    assert prediction.shape == (2, 1, 31, 47)
    assert torch.isfinite(prediction).all()


def test_camera_depth_model_accepts_sparse_lidar_input() -> None:
    model = CameraDepthModel(lidar_mode="sparse")
    rgb = torch.zeros((2, 3, 15, 21), dtype=torch.float32)
    lidar_maps = torch.zeros((2, 6, 15, 21), dtype=torch.float32)

    prediction = model(rgb, lidar_maps)

    assert prediction.shape == (2, 1, 15, 21)
    assert torch.isfinite(prediction).all()


def _v1_forward(model, rgb, lidar_maps):
    """Freeze the pre-feature-API computation as an independent reference."""

    stem = model.stem(torch.cat([rgb, lidar_maps], dim=1))
    encoded_1 = model.encoder_1(stem)
    encoded_2 = model.encoder_2(encoded_1)
    decoded_1 = torch.nn.functional.interpolate(
        encoded_2, size=encoded_1.shape[-2:], mode="bilinear", align_corners=False,
    )
    decoded_1 = model.decoder_1(torch.cat([decoded_1, encoded_1], dim=1))
    decoded_0 = torch.nn.functional.interpolate(
        decoded_1, size=stem.shape[-2:], mode="bilinear", align_corners=False,
    )
    decoded_0 = model.decoder_0(torch.cat([decoded_0, stem], dim=1))
    return model.depth_head(decoded_0)


@pytest.mark.parametrize("lidar_mode,channels", [("sparse", 6), ("enriched", 8)])
@pytest.mark.parametrize("shape", [(16, 24), (15, 21), (1, 1)])
def test_camera_depth_features_preserve_full_resolution(lidar_mode, channels, shape):
    torch.manual_seed(13)
    model = CameraDepthModel(lidar_mode=lidar_mode).eval()
    rgb = torch.rand((2, 3, *shape))
    lidar = torch.rand((2, channels, *shape))
    decoder_outputs = []
    handle = model.decoder_0.register_forward_hook(
        lambda module, inputs, output: decoder_outputs.append(output),
    )
    try:
        with torch.no_grad():
            features = model.forward_features(rgb, lidar)
    finally:
        handle.remove()

    assert features.shape == (2, 32, *shape)
    assert torch.isfinite(features).all()
    assert len(decoder_outputs) == 1
    assert features is decoder_outputs[0]


@pytest.mark.parametrize("lidar_mode,channels", [("sparse", 6), ("enriched", 8)])
def test_camera_depth_feature_path_matches_v1_depth_prediction(lidar_mode, channels):
    torch.manual_seed(17)
    model = CameraDepthModel(lidar_mode=lidar_mode).eval()
    rgb = torch.rand((2, 3, 15, 21))
    lidar = torch.rand((2, channels, 15, 21))
    with torch.no_grad():
        expected = _v1_forward(model, rgb, lidar)
        features = model.forward_features(rgb, lidar)
        from_features = model.depth_head(features)
        actual = model(rgb, lidar)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(from_features, expected, rtol=0, atol=0)


@pytest.mark.parametrize("lidar_mode,channels", [("sparse", 6), ("enriched", 8)])
def test_camera_depth_features_backpropagate_without_depth_head(lidar_mode, channels):
    torch.manual_seed(13)
    model = CameraDepthModel(lidar_mode=lidar_mode)
    rgb = torch.rand((1, 3, 15, 21), requires_grad=True)
    lidar = torch.rand((1, channels, 15, 21), requires_grad=True)

    features = model.forward_features(rgb, lidar)
    assert features.requires_grad
    features.square().mean().backward()

    for value in (rgb, lidar):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    for name, parameter in model.named_parameters():
        if name.startswith("depth_head."):
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    for block in (model.stem, model.encoder_1, model.encoder_2, model.decoder_1, model.decoder_0):
        assert sum(float(p.grad.abs().sum()) for p in block.parameters()) > 0.0


@pytest.mark.parametrize("lidar_mode,channels", [("sparse", 6), ("enriched", 8)])
def test_camera_depth_feature_api_preserves_v1_checkpoint_schema(
    tmp_path, lidar_mode, channels,
):
    # The v1 checkpoint contract has these 13 convolutions, each with a bias.
    convolutions = {
        "stem.0": (32, 3 + channels, 3, 3),
        "stem.2": (32, 32, 3, 3),
        "encoder_1.0": (64, 32, 3, 3),
        "encoder_1.2.0": (64, 64, 3, 3),
        "encoder_1.2.2": (64, 64, 3, 3),
        "encoder_2.0": (96, 64, 3, 3),
        "encoder_2.2.0": (96, 96, 3, 3),
        "encoder_2.2.2": (96, 96, 3, 3),
        "decoder_1.0": (64, 160, 3, 3),
        "decoder_1.2": (64, 64, 3, 3),
        "decoder_0.0": (32, 96, 3, 3),
        "decoder_0.2": (32, 32, 3, 3),
        "depth_head": (1, 32, 1, 1),
    }
    expected_shapes = {}
    for name, shape in convolutions.items():
        expected_shapes[f"{name}.weight"] = shape
        expected_shapes[f"{name}.bias"] = (shape[0],)
    torch.manual_seed(19)
    state = CameraDepthModel(lidar_mode=lidar_mode).state_dict()
    assert {key: tuple(value.shape) for key, value in state.items()} == expected_shapes
    path = tmp_path / "v1_checkpoint.pt"
    torch.save({"model_state_dict": state}, path)

    model = CameraDepthModel(lidar_mode=lidar_mode).eval()
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"],
        strict=True,
    )
    rgb = torch.rand((1, 3, 15, 21))
    lidar = torch.rand((1, channels, 15, 21))
    with torch.no_grad():
        expected = _v1_forward(model, rgb, lidar)
        actual = model.depth_head(model.forward_features(rgb, lidar))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("lidar_mode,channels", [("sparse", 6), ("enriched", 8)])
def test_camera_depth_features_reject_invalid_input_channels(lidar_mode, channels):
    model = CameraDepthModel(lidar_mode=lidar_mode)
    with pytest.raises(ValueError, match=f"must have {channels} channels"):
        model.forward_features(torch.zeros((1, 3, 15, 21)), torch.zeros((1, channels + 1, 15, 21)))


def test_masked_depth_loss_uses_only_selected_pixels_and_rejects_empty_mask() -> None:
    prediction = torch.tensor([[[[0.0, 100.0], [0.0, 0.0]]]])
    target = torch.tensor([[[[1.0, -100.0], [0.0, 0.0]]]])
    mask = torch.tensor([[[[True, False], [False, False]]]])

    loss = masked_depth_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="at least one depth target"):
        masked_depth_loss(prediction, target, torch.zeros_like(mask))


def test_camera_depth_holdout_is_deterministic_and_removes_target_contributions() -> None:
    batch = dataset_items_to_model_batch([_item()], include_splatted_depth=False)
    config = SplattingConfig(radius_px=2, sigma_px=1.0)

    first = model_batch_to_camera_depth_training_batch(
        batch,
        holdout_fraction=0.4,
        seed=19,
        splatting_config=config,
    )
    second = model_batch_to_camera_depth_training_batch(
        batch,
        holdout_fraction=0.4,
        seed=19,
        splatting_config=config,
    )

    np.testing.assert_array_equal(first["loss_mask"], second["loss_mask"])
    np.testing.assert_array_equal(first["lidar_maps"], second["lidar_maps"])
    held_out = first["loss_mask"][0, 0]
    assert int(held_out.sum()) == 2
    assert np.all(first["kept_sparse_lidar_maps"][0, :, held_out] == 0.0)
    assert np.all(first["depth_target"][0, 0, held_out] > 0.0)

    kept = first["kept_sparse_lidar_maps"][0]
    expected_batch = dataset_items_to_model_batch(
        [{"image": _item()["image"], "lidar_maps": kept, "target": {}, "meta": {}}],
        include_splatted_depth=True,
        splatting_config=config,
    )
    np.testing.assert_allclose(first["lidar_maps"][0], expected_batch["inputs"][0, 3:])


@pytest.mark.parametrize("holdout_mode", ["random_points", "spatial_blocks"])
def test_camera_depth_training_step_has_finite_non_zero_gradients(holdout_mode: str) -> None:
    torch.manual_seed(7)
    batch = dataset_items_to_model_batch(
        [_item(0), _item(1)],
        include_splatted_depth=False,
    )
    depth_batch = model_batch_to_camera_depth_training_batch(
        batch,
        holdout_fraction=0.4,
        seed=23,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
        holdout_mode=holdout_mode,
        block_height_px=(2, 3),
        block_width_px=(3, 5),
    )
    model = CameraDepthModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = train_camera_depth_step(
        model=model,
        optimizer=optimizer,
        rgb=torch.from_numpy(depth_batch["rgb"]),
        lidar_maps=torch.from_numpy(depth_batch["lidar_maps"]),
        depth_target=torch.from_numpy(depth_batch["depth_target"]),
        loss_mask=torch.from_numpy(depth_batch["loss_mask"]),
        device=torch.device("cpu"),
    )

    assert np.isfinite(metrics.loss)
    assert np.isfinite(metrics.grad_norm)
    assert metrics.grad_norm > 0.0


@pytest.mark.parametrize("holdout_mode", ["random_points", "spatial_blocks"])
def test_camera_depth_validation_is_fixed_and_does_not_compute_gradients(holdout_mode: str) -> None:
    torch.manual_seed(11)
    model = CameraDepthModel()
    dataset = [_item(0), _item(1)]
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

    first = evaluate_camera_depth(
        model=model,
        dataset=dataset,
        indices=[0, 1],
        batch_size=2,
        holdout_fraction=0.4,
        holdout_seed=101,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
        device=torch.device("cpu"),
        max_depth_m=80.0,
        beta=0.1,
        holdout_mode=holdout_mode,
        block_height_px=(2, 3),
        block_width_px=(3, 5),
    )
    second = evaluate_camera_depth(
        model=model,
        dataset=dataset,
        indices=[0, 1],
        batch_size=1,
        holdout_fraction=0.4,
        holdout_seed=101,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
        device=torch.device("cpu"),
        max_depth_m=80.0,
        beta=0.1,
        holdout_mode=holdout_mode,
        block_height_px=(2, 3),
        block_width_px=(3, 5),
    )

    assert first.pixel_count == second.pixel_count
    assert first.loss == pytest.approx(second.loss, rel=1e-6)
    assert first.mae_m == pytest.approx(second.mae_m, rel=1e-6)
    assert first.rmse_m == pytest.approx(second.rmse_m, rel=1e-6)
    assert first.splat_mae_m == pytest.approx(second.splat_mae_m, rel=1e-6)
    assert first.splat_rmse_m == pytest.approx(second.splat_rmse_m, rel=1e-6)
    assert first.splat_coverage == pytest.approx(second.splat_coverage)
    assert first.pixel_count == 4
    assert first.holdout_fraction == second.holdout_fraction == pytest.approx(0.4)
    assert first.loss >= 0.0
    assert first.mae_m >= 0.0
    assert first.rmse_m >= first.mae_m
    assert first.splat_mae_m >= 0.0
    assert first.splat_rmse_m >= first.splat_mae_m
    assert 0.0 <= first.splat_coverage <= 1.0
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )


def test_spatial_validation_masks_are_fixed_per_index_despite_batch_order():
    dataset = [_item(0), _item(1)]
    settings = dict(
        dataset=dataset, holdout_fraction=0.4, holdout_seed=101,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
        holdout_mode="spatial_blocks", block_height_px=(2, 3), block_width_px=(3, 5),
    )

    batched = prepare_camera_depth_evaluation_batch(indices=[1, 0], **settings)
    singles = [prepare_camera_depth_evaluation_batch(indices=[index], **settings) for index in (0, 1)]

    for key in batched:
        np.testing.assert_array_equal(batched[key][0], singles[1][key][0])
        np.testing.assert_array_equal(batched[key][1], singles[0][key][0])


def test_camera_depth_validation_supports_sparse_lidar_ablation() -> None:
    model = CameraDepthModel(lidar_mode="sparse")

    metrics = evaluate_camera_depth(
        model=model,
        dataset=[_item(0), _item(1)],
        indices=[0, 1],
        batch_size=2,
        holdout_fraction=0.4,
        holdout_seed=101,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
        device=torch.device("cpu"),
        max_depth_m=80.0,
        beta=0.1,
        lidar_mode="sparse",
    )

    assert metrics.pixel_count == 4
    assert np.isfinite(metrics.mae_m)
