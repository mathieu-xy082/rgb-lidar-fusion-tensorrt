from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.bev_fusion import BEVFeatureFusion
from rgb_lidar_fusion.camera_bev import BEVGridConfig, CameraBEVFeatures, lift_camera_features_to_bev
from rgb_lidar_fusion.camera_depth_support import CameraDepthSupport
from rgb_lidar_fusion.lidar_bev import (
    LidarBEVContext, LidarBEVFeatures, PillarFeatureNet, scatter_pillar_features,
)


def _grid(shape=(3, 5), **overrides):
    return BEVGridConfig(**{
        "x_range_m": (1., 1. + shape[0] * .2),
        "y_range_m": (-2., -2. + shape[1] * .2),
        "z_range_m": (-3., 3.), "resolution_m": .2, **overrides,
    })


def _inputs(shape=(3, 5), *, batch=2, dtype=torch.float32, channels=(2, 3)):
    camera = torch.ones(batch, channels[0], *shape, dtype=dtype)
    lidar = torch.full((batch, channels[1], *shape), 2., dtype=dtype)
    counts = torch.ones(batch, 1, *shape, dtype=torch.int64)
    return (
        CameraBEVFeatures(camera, counts, counts.to(dtype) * .25,
                          torch.ones(batch, 1, 7, 11, dtype=torch.bool)),
        LidarBEVFeatures(lidar, counts.clone() * 90),
    )


def _fusion(**overrides):
    return BEVFeatureFusion(**{
        "camera_channels": 2, "lidar_channels": 3, "out_channels": 4, **overrides,
    })


def _run(model, camera, lidar, *, grid=None, camera_grid=None, lidar_grid=None):
    grid = grid or _grid()
    return model(camera, lidar, camera_grid=camera_grid or grid, lidar_grid=lidar_grid or grid)


def _identity_fusion():
    model = _fusion(out_channels=7)
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(7).reshape(7, 7, 1, 1))
    return model


def test_default_projection_structure_and_parameter_count():
    model = BEVFeatureFusion()
    assert list(model.children()) == [model.projection]
    projection = model.projection
    assert (projection.in_channels, projection.out_channels) == (66, 32)
    assert projection.kernel_size == projection.stride == (1, 1)
    assert projection.padding == (0, 0) and projection.bias is None
    assert sum(parameter.numel() for parameter in model.parameters()) == 66 * 32


def test_ordered_channels_and_distinct_supports_with_controlled_weights():
    camera, lidar = _inputs()
    camera.features[:, 0] = -3.
    camera.features[:, 1] = 4.
    lidar.features[:, 0] = 5.
    lidar.features[:, 1] = 6.
    lidar.features[:, 2] = 7.
    camera.pixel_count[0, :, 0, 1] = 0
    camera.weight_sum[0, :, 0, 1] = 0
    camera.features[0, :, 0, 1] = 0
    lidar.point_count[0, :, 1, 2] = 0
    result = _run(_identity_fusion(), camera, lidar)
    torch.testing.assert_close(result.features[:, :2], camera.features, rtol=0, atol=0)
    torch.testing.assert_close(result.features[:, 2:5], lidar.features, rtol=0, atol=0)
    assert torch.equal(result.features[:, 5:6], (camera.pixel_count > 0).float())
    assert torch.equal(result.features[:, 6:7], lidar.observed_mask.float())
    assert result.features[0, 0, 0, 0] == -3  # No hidden ReLU.
    assert result.camera_support.dtype == result.lidar_observed_mask.dtype == torch.bool


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (5, 8), (9, 7), (250, 250), (251, 249)])
def test_preserves_explicit_grid_shape_and_evidence_snapshots(shape):
    grid = _grid(shape)
    camera, lidar = _inputs(shape)
    result = _run(_fusion(), camera, lidar, grid=grid)
    assert result.features.shape == (2, 4, *shape) and result.grid == grid
    assert torch.isfinite(result.features).all()
    for actual, source in (
        (result.camera_pixel_count, camera.pixel_count),
        (result.camera_weight_sum, camera.weight_sum),
        (result.lidar_point_count, lidar.point_count),
    ):
        torch.testing.assert_close(actual, source, rtol=0, atol=0)
        saved = source.clone()
        actual.zero_()
        torch.testing.assert_close(source, saved, rtol=0, atol=0)
        actual.copy_(saved)
        source.zero_()
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)


def test_inputs_and_image_space_mask_are_not_modified_or_used_as_bev_channels():
    camera, lidar = _inputs()
    values = [camera.features, camera.pixel_count, camera.weight_sum, camera.pixel_mask,
              lidar.features, lidar.point_count]
    snapshots = [value.clone() for value in values]
    model = _fusion()
    first = _run(model, camera, lidar).features
    for actual, expected in zip(values, snapshots, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    second = _run(model, replace(camera, pixel_mask=torch.zeros(2, 1, 1, 1, dtype=torch.bool)), lidar)
    torch.testing.assert_close(first, second.features, rtol=0, atol=0)


def test_raw_count_and_splat_mass_magnitudes_are_not_extra_features():
    camera, lidar = _inputs()
    model = _fusion()
    first = _run(model, camera, lidar)
    second = _run(model, replace(camera, pixel_count=camera.pixel_count * 100,
                                weight_sum=camera.weight_sum * .001),
                  replace(lidar, point_count=lidar.point_count * 3))
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
    assert second.lidar_point_count[0, 0, 0, 0] == 270  # Pre-cap, not 64.


@pytest.mark.parametrize("source", ["camera", "lidar", "neither"])
def test_single_branch_and_empty_support_samples(source):
    camera, lidar = _inputs()
    if source != "camera":
        camera.features.zero_()
        camera.pixel_count.zero_()
        camera.weight_sum.zero_()
    if source != "lidar":
        lidar.features.zero_()
        lidar.point_count.zero_()
    result = _run(_identity_fusion(), camera, lidar)
    assert result.camera_support.any().item() == (source == "camera")
    assert result.lidar_observed_mask.any().item() == (source == "lidar")
    assert torch.isfinite(result.features).all()
    assert (result.features.abs().sum() > 0).item() == (source != "neither")


def test_context_outside_both_supports_is_not_erased():
    camera, lidar = _inputs()
    camera.features[:, :, 1, 2] = 0
    camera.pixel_count[:, :, 1, 2] = 0
    camera.weight_sum[:, :, 1, 2] = 0
    lidar.point_count[:, :, 1, 2] = 0
    result = _run(_identity_fusion(), camera, lidar)
    assert not result.camera_support[:, :, 1, 2].any()
    assert not result.lidar_observed_mask[:, :, 1, 2].any()
    torch.testing.assert_close(result.features[:, 2:5, 1, 2], lidar.features[:, :, 1, 2])


def test_projection_does_not_mix_cells_or_batches():
    camera, lidar = _inputs()
    camera.features.zero_()
    lidar.features.zero_()
    camera.features[0, 0, 1, 2] = 3
    model = _fusion(out_channels=1)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.weight[0, 0] = 1
    result = _run(model, camera, lidar)
    expected = torch.zeros_like(result.features)
    expected[0, 0, 1, 2] = 3
    torch.testing.assert_close(result.features, expected, rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_finite_nonzero_gradients_reach_both_feature_maps_and_projection(training):
    camera, lidar = _inputs()
    camera.features.requires_grad_()
    lidar.features.requires_grad_()
    mass = camera.weight_sum.requires_grad_()
    model = _fusion().train(training)
    with torch.no_grad():
        model.projection.weight.fill_(.1)
    _run(model, camera, lidar).features.square().mean().backward()
    for gradient in (camera.features.grad, lidar.features.grad, model.projection.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert mass.grad is None


def test_double_precision_gradcheck():
    camera, lidar = _inputs(shape=(1, 2), batch=1, dtype=torch.float64, channels=(1, 1))
    model = _fusion(camera_channels=1, lidar_channels=1, out_channels=1).double()
    def forward(camera_features, lidar_features):
        return _run(model, replace(camera, features=camera_features),
                    replace(lidar, features=lidar_features), grid=_grid((1, 2))).features
    assert torch.autograd.gradcheck(
        forward, (camera.features.requires_grad_(), lidar.features.requires_grad_()),
    )


def test_empty_support_sample_has_finite_backward_without_creating_observations():
    camera, lidar = _inputs()
    camera.features.zero_().requires_grad_()
    lidar.features.zero_().requires_grad_()
    camera.pixel_count.zero_()
    camera.weight_sum.zero_()
    lidar.point_count.zero_()
    model = _fusion()
    result = _run(model, camera, lidar)
    assert torch.all(result.features == 0)
    assert not result.camera_support.any() and not result.lidar_observed_mask.any()
    result.features.sum().backward()
    assert torch.isfinite(camera.features.grad).all() and torch.isfinite(lidar.features.grad).all()
    assert torch.all(model.projection.weight.grad == 0)


def test_tiny_positive_camera_mass_is_not_thresholded_or_used_as_a_multiplier():
    camera, lidar = _inputs()
    model = _identity_fusion()
    first = _run(model, camera, lidar)
    second = _run(model, replace(camera, weight_sum=camera.weight_sum * 1e-12), lidar)
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
    assert second.camera_support.all()


def test_existing_lift_pfn_scatter_and_context_connect_to_fusion_backward():
    grid = _grid()
    image_features = torch.ones(1, 2, 1, 2, requires_grad=True)
    depth = torch.ones(1, 1, 1, 2)
    confidence = torch.tensor([[[[.25, .75]]]])
    projection = torch.eye(3, 4)[None]
    projection[:, 0, 0] = 10
    projection[:, 0, 3] = -10.5
    projection[:, 1, 3] = 1.9
    camera = lift_camera_features_to_bev(
        image_features,
        depth_support=CameraDepthSupport("splatted", depth, depth.bool(), confidence),
        camera_projection=projection, t_vehicle_to_camera=torch.eye(4)[None],
        grid=grid, weight_by_splat_confidence=True,
    )
    point_features = torch.ones(2, 2, 9, requires_grad=True)
    point_mask = torch.ones(2, 2, dtype=torch.bool)
    indices = torch.tensor([[0, 0, 0], [0, 1, 2]])
    counts = torch.zeros(1, 1, *grid.shape, dtype=torch.int64)
    counts[0, 0, 0, 0] = counts[0, 0, 1, 2] = 2
    pfn, context = PillarFeatureNet(out_channels=3).eval(), LidarBEVContext(channels=3)
    fusion = _fusion()
    with torch.no_grad():
        for module in (pfn, context, fusion):
            for parameter in module.parameters():
                parameter.fill_(.1)
    lidar = context(scatter_pillar_features(
        pfn(point_features, point_mask), indices=indices, point_count=counts,
    ))
    result = _run(fusion, camera, lidar, grid=grid)
    assert result.camera_pixel_count.sum() == 2 and result.camera_support.sum() == 1
    assert result.lidar_point_count.sum() == 4 and result.lidar_observed_mask.sum() == 2
    result.features.square().mean().backward()
    for tensor in (image_features, point_features):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    for module in (pfn, context, fusion):
        for parameter in module.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["camera_channels", "lidar_channels", "out_channels"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_channel_counts_are_rejected(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        _fusion(**{name: value})


@pytest.mark.parametrize("overrides", [
    {"x_range_m": (2., 2.6)}, {"y_range_m": (-1., 0.)},
    {"z_range_m": (-2., 4.)},
    {"x_range_m": (1., 2.2), "y_range_m": (-2., 0.), "resolution_m": .4},
])
def test_different_metric_grids_are_rejected_even_for_equal_shapes(overrides):
    camera, lidar = _inputs()
    other = _grid(**overrides)
    assert other.shape == _grid().shape
    with pytest.raises(ValueError, match="identical metric bounds and resolution"):
        _run(_fusion(), camera, lidar, lidar_grid=other)


def test_equivalent_grid_values_with_list_bounds_are_accepted():
    camera, lidar = _inputs()
    grid = _grid()
    other = replace(grid, x_range_m=list(grid.x_range_m))
    assert _run(_fusion(), camera, lidar, lidar_grid=other).grid == grid


@pytest.mark.parametrize("name", ["camera_grid", "lidar_grid"])
def test_missing_grid_types_are_rejected(name):
    camera, lidar = _inputs()
    kwargs = {"camera_grid": _grid(), "lidar_grid": _grid(), name: None}
    with pytest.raises(ValueError, match="explicit BEVGridConfig"):
        _fusion()(camera, lidar, **kwargs)


@pytest.mark.parametrize("source", ["camera", "lidar"])
@pytest.mark.parametrize("case", ["ndim", "channels", "spatial", "empty", "batch"])
def test_malformed_features_and_batch_mismatch_are_rejected(source, case):
    camera, lidar = _inputs()
    original = camera.features if source == "camera" else lidar.features
    changed = {
        "ndim": original[0], "channels": original[:, :1],
        "spatial": original[:, :, :1], "empty": original[:0], "batch": original[:1],
    }[case]
    if source == "camera":
        camera = replace(camera, features=changed)
    else:
        lidar = replace(lidar, features=changed)
    with pytest.raises(ValueError, match="batch size" if case == "batch" else "shape"):
        _run(_fusion(), camera, lidar)


@pytest.mark.parametrize("source", ["camera", "lidar"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.int64, torch.float64])
def test_feature_dtype_mismatch_is_rejected(source, dtype):
    camera, lidar = _inputs()
    if source == "camera":
        camera = replace(camera, features=camera.features.to(dtype))
    else:
        lidar = replace(lidar, features=lidar.features.to(dtype))
    with pytest.raises(ValueError, match="dtype"):
        _run(_fusion(), camera, lidar)


@pytest.mark.parametrize("source", ["camera", "lidar"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_features_are_rejected(source, value):
    camera, lidar = _inputs()
    features = camera.features if source == "camera" else lidar.features
    features[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        _run(_fusion(), camera, lidar)


@pytest.mark.parametrize("source", ["camera", "lidar"])
@pytest.mark.parametrize("case", ["dtype", "shape", "negative", "device"])
def test_invalid_count_evidence_is_rejected(source, case):
    camera, lidar = _inputs()
    original = camera.pixel_count if source == "camera" else lidar.point_count
    changed = {
        "dtype": original.float(), "shape": original[:, :, :1],
        "negative": -original, "device": original.to("meta"),
    }[case]
    if source == "camera":
        camera = replace(camera, pixel_count=changed)
    else:
        lidar = replace(lidar, point_count=changed)
    with pytest.raises(ValueError, match="non-negative" if case == "negative" else
                       "device" if case == "device" else "int64 shape"):
        _run(_fusion(), camera, lidar)


@pytest.mark.parametrize("case", ["shape", "dtype", "device", "negative", "nan", "inf", "zero"])
def test_invalid_camera_mass_is_rejected(case):
    camera, lidar = _inputs()
    original = camera.weight_sum
    changed = {
        "shape": original[:, :, :1], "dtype": original.double(),
        "device": original.to("meta"), "negative": -original,
        "nan": original * float("nan"), "inf": original * float("inf"),
        "zero": torch.zeros_like(original),
    }[case]
    with pytest.raises(ValueError, match="same positive support" if case == "zero" else "weight_sum"):
        _run(_fusion(), replace(camera, weight_sum=changed), lidar)


def test_positive_mass_with_zero_camera_count_is_rejected():
    camera, lidar = _inputs()
    with pytest.raises(ValueError, match="same positive support"):
        _run(_fusion(), replace(camera, pixel_count=torch.zeros_like(camera.pixel_count)), lidar)


def test_nonzero_camera_features_without_support_are_rejected_not_silently_masked():
    camera, lidar = _inputs()
    camera.pixel_count[0, 0, 0, 0] = 0
    camera.weight_sum[0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match="zero outside camera support"):
        _run(_fusion(), camera, lidar)


@pytest.mark.parametrize("source", ["camera", "lidar"])
def test_feature_device_mismatch_is_rejected_without_gpu(source):
    camera, lidar = _inputs()
    if source == "camera":
        camera = replace(camera, features=camera.features.to("meta"))
    else:
        lidar = replace(lidar, features=lidar.features.to("meta"))
    with pytest.raises(ValueError, match="module's device and dtype"):
        _run(_fusion(), camera, lidar)
