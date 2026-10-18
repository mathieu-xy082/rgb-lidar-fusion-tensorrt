import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_depth_support import (
    build_depth_valid_mask,
    prepare_camera_depth_support,
)
from rgb_lidar_fusion.camera_geometry import backproject_depth_to_vehicle
from rgb_lidar_fusion.model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_camera_depth_training_batch,
)
from rgb_lidar_fusion.lidar_splatting import SplattingConfig


def _maps(channels=8, batch=1):
    maps = torch.zeros(batch, channels, 3, 5)
    maps[:, 0, 1, 2] = 0.25
    maps[:, 5, 1, 2] = 1
    if channels == 8:
        maps[:, 6, 1, 1:4] = 0.25
        maps[:, 7, 1, 1:4] = torch.tensor([0.3, 1., 0.3])
    return maps


def test_valid_mask_intersects_support_finiteness_and_explicit_metric_bounds():
    depth = torch.tensor([[[[float("nan"), float("inf"), -2., 0., 1., 2., 80., 81., 10.]]]])
    support = torch.ones_like(depth, dtype=torch.bool)
    support[..., -1] = False
    before = depth.clone()
    result = build_depth_valid_mask(
        depth, support_mask=support, min_valid_depth_m=1., max_valid_depth_m=80.,
    )
    torch.testing.assert_close(result, torch.tensor([[[[False]*5 + [True, True, False, False]]]]))
    torch.testing.assert_close(depth, before, equal_nan=True)


@pytest.mark.parametrize("channels", [6, 8])
def test_sparse_source_uses_only_present_points_and_per_sample_scale(channels):
    maps = _maps(channels, batch=2)
    maps[:, 0, 0, 0] = 0.5  # Positive depth without a measurement mask.
    before = maps.clone()
    result = prepare_camera_depth_support(
        maps, source="sparse", max_depth_m=torch.tensor([80., 60.]), max_valid_depth_m=80.,
    )
    assert result.source == "sparse"
    assert result.splat_confidence is None
    assert result.valid_mask.sum() == 2
    torch.testing.assert_close(result.depth_m[:, 0, 1, 2], torch.tensor([20., 15.]))
    assert torch.all(result.depth_m[~result.valid_mask] == 0)
    torch.testing.assert_close(maps, before)


def test_selection_maximum_is_independent_of_normalization_scale():
    result = prepare_camera_depth_support(
        _maps(), source="sparse", max_depth_m=80., max_valid_depth_m=10.,
    )
    assert not result.valid_mask.any()
    assert torch.all(result.depth_m == 0)


def test_splat_support_is_distinct_from_sparse_presence_and_confidence_is_not_mutated():
    maps = _maps()
    result = prepare_camera_depth_support(
        maps, source="splatted", max_depth_m=80., max_valid_depth_m=80.,
    )
    assert result.valid_mask.sum() == 3
    assert not (maps[:, 5:6] > 0)[result.valid_mask].all()
    torch.testing.assert_close(result.splat_confidence, maps[:, 7:8])
    assert torch.all(result.depth_m[result.valid_mask] == 20.)
    strict = prepare_camera_depth_support(
        maps, source="splatted", max_depth_m=80., max_valid_depth_m=80.,
        min_splat_confidence=0.5,
    )
    assert strict.valid_mask.sum() == 1
    torch.testing.assert_close(strict.splat_confidence, result.splat_confidence)


@pytest.mark.parametrize("confidence", [0., -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_splat_confidence_cannot_establish_support(confidence):
    maps = _maps()
    maps[:, 7] = confidence
    result = prepare_camera_depth_support(
        maps, source="splatted", max_depth_m=80., max_valid_depth_m=80.,
    )
    assert not result.valid_mask.any()
    assert torch.isfinite(result.depth_m).all()
    torch.testing.assert_close(result.splat_confidence, maps[:, 7:8], equal_nan=True)


@pytest.mark.parametrize("source,channel", [("sparse", 0), ("splatted", 6), ("predicted", None)])
def test_invalid_depths_remain_excluded_even_with_support(source, channel):
    maps = _maps()
    depth = torch.tensor([[[[float("nan"), float("inf"), -0.1, 0., 2.]]]]).expand(1, 1, 3, 5)
    options = {}
    if source == "predicted":
        options = dict(depth_prediction=depth, prediction_support_mask=torch.ones_like(depth, dtype=torch.bool))
    else:
        maps[:, channel:channel+1] = depth
        maps[:, 5 if source == "sparse" else 7] = 1.
    result = prepare_camera_depth_support(
        maps, source=source, max_depth_m=80., max_valid_depth_m=80., **options,
    )
    assert not result.valid_mask.any()
    assert torch.all(result.depth_m == 0.)


def test_prediction_requires_explicit_support_and_does_not_receive_splat_uncertainty():
    maps = _maps()
    prediction = torch.full_like(maps[:, 0:1], 0.5)
    with pytest.raises(ValueError, match="prediction_support_mask"):
        prepare_camera_depth_support(
            maps, source="predicted", max_depth_m=80., max_valid_depth_m=80.,
            depth_prediction=prediction,
        )
    support = torch.zeros_like(prediction, dtype=torch.bool)
    support[..., 0, 0] = True  # Deliberate domain, outside splat support.
    result = prepare_camera_depth_support(
        maps, source="predicted", max_depth_m=80., max_valid_depth_m=80.,
        depth_prediction=prediction, prediction_support_mask=support,
    )
    assert result.valid_mask.sum() == 1
    assert result.depth_m[0, 0, 0, 0] == 40.
    assert result.splat_confidence is None


def test_invalid_prediction_placeholders_allow_finite_geometry_backward():
    maps = _maps()
    prediction = torch.full_like(maps[:, 0:1], 0.25)
    prediction[..., 0, 0] = float("nan")
    prediction.requires_grad_()
    selected = prepare_camera_depth_support(
        maps, source="predicted", max_depth_m=80., max_valid_depth_m=80.,
        depth_prediction=prediction, prediction_support_mask=torch.ones_like(prediction, dtype=torch.bool),
    )
    xyz = backproject_depth_to_vehicle(
        selected.depth_m, camera_projection=torch.eye(3, 4)[None],
        t_vehicle_to_camera=torch.eye(4)[None],
    )
    assert torch.isfinite(xyz).all()
    # The zero placeholder also has XYZ: exclude it explicitly before use.
    xyz.permute(0, 2, 3, 1)[selected.valid_mask[:, 0]].square().mean().backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 0, 0] == 0
    assert torch.all(prediction.grad[selected.valid_mask] != 0)


@pytest.mark.parametrize("mode", ["random_points", "spatial_blocks"])
def test_support_uses_only_post_holdout_maps_not_hidden_targets(mode):
    sparse = np.zeros((6, 12, 16), dtype=np.float32)
    sparse[0, 2:10, 2:14] = 0.25
    sparse[5, 2:10, 2:14] = 1.
    batch = dataset_items_to_model_batch([
        {"image": np.zeros((3, 12, 16), dtype=np.float32), "lidar_maps": sparse}
    ], include_splatted_depth=False)
    options = dict(
        holdout_mode=mode, seed=13, block_height_px=(3, 3), block_width_px=(3, 3),
        splatting_config=SplattingConfig(radius_px=1),
    )
    prepared = model_batch_to_camera_depth_training_batch(batch, **options)
    changed = {**batch, "sparse_lidar_maps": batch["sparse_lidar_maps"].copy()}
    changed["sparse_lidar_maps"][:, 0:1][prepared["loss_mask"]] = 0.9
    repeated = model_batch_to_camera_depth_training_batch(changed, **options)
    for source in ("sparse", "splatted"):
        results = [prepare_camera_depth_support(
            torch.from_numpy(value["lidar_maps"]), source=source,
            max_depth_m=80., max_valid_depth_m=80.,
        ) for value in (prepared, repeated)]
        torch.testing.assert_close(results[0].depth_m, results[1].depth_m)
        torch.testing.assert_close(results[0].valid_mask, results[1].valid_mask)
        if source == "sparse":
            assert not results[0].valid_mask[torch.from_numpy(prepared["loss_mask"])].any()
        else:
            torch.testing.assert_close(results[0].splat_confidence, results[1].splat_confidence)
    assert np.any(prepared["depth_target"] != repeated["depth_target"])


@pytest.mark.parametrize("minimum,maximum", [(-1, 80), (0, 0), (80, 1), (0, float("inf")), (float("nan"), 80)])
def test_mask_rejects_invalid_metric_bounds(minimum, maximum):
    with pytest.raises(ValueError, match="depth bounds"):
        build_depth_valid_mask(
            torch.ones(1, 1, 2, 3), support_mask=torch.ones(1, 1, 2, 3, dtype=torch.bool),
            min_valid_depth_m=minimum, max_valid_depth_m=maximum,
        )


@pytest.mark.parametrize("support", [torch.ones(1, 1, 3, 5), torch.ones(1, 1, 1, 1, dtype=torch.bool)])
def test_mask_rejects_implicit_support_conversion_or_broadcasting(support):
    with pytest.raises(ValueError, match="support_mask"):
        build_depth_valid_mask(torch.ones(1, 1, 3, 5), support_mask=support, max_valid_depth_m=80.)


@pytest.mark.parametrize("maps", [torch.zeros(1, 7, 3, 5), torch.zeros(1, 6, 3),
                                  torch.zeros(1, 6, 0, 5), torch.zeros(1, 6, 3, 5).int()])
def test_selector_rejects_invalid_lidar_layout(maps):
    with pytest.raises(ValueError, match="lidar_maps"):
        prepare_camera_depth_support(maps, source="sparse", max_depth_m=80., max_valid_depth_m=80.)


@pytest.mark.parametrize("options,match", [
    ({"source": "other"}, "source"),
    ({"min_splat_confidence": -0.1}, "min_splat_confidence"),
    ({"min_splat_confidence": float("nan")}, "min_splat_confidence"),
    ({"depth_prediction": torch.ones(1, 1, 3, 5)}, "prediction arguments"),
])
def test_selector_rejects_ambiguous_or_invalid_options(options, match):
    with pytest.raises(ValueError, match=match):
        prepare_camera_depth_support(_maps(), **{
            "source": "sparse", "max_depth_m": 80., "max_valid_depth_m": 80., **options,
        })


def test_splat_source_requires_enriched_maps():
    with pytest.raises(ValueError, match="8 LiDAR channels"):
        prepare_camera_depth_support(_maps(6), source="splatted", max_depth_m=80., max_valid_depth_m=80.)


@pytest.mark.parametrize("prediction", [torch.ones(1, 1, 1, 1), torch.ones(1, 1, 3, 5).double()])
def test_prediction_rejects_implicit_broadcast_or_dtype_conversion(prediction):
    with pytest.raises(ValueError, match="depth_prediction must share"):
        prepare_camera_depth_support(
            _maps(), source="predicted", max_depth_m=80., max_valid_depth_m=80.,
            depth_prediction=prediction, prediction_support_mask=torch.ones_like(prediction, dtype=torch.bool),
        )
