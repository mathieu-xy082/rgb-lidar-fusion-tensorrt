import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_bev import BEVGridConfig, lift_camera_features_to_bev
from rgb_lidar_fusion.camera_depth_model import CameraDepthModel
from rgb_lidar_fusion.camera_depth_support import CameraDepthSupport, prepare_camera_depth_support


def _grid(**overrides):
    return BEVGridConfig(**{
        "x_range_m": (0., 4.), "y_range_m": (-2., 2.),
        "z_range_m": (-1., 3.), "resolution_m": 1., **overrides,
    })


def _simple_support(depth, *, source="sparse", mask=None, confidence=None):
    return CameraDepthSupport(
        source, depth, torch.ones_like(depth, dtype=torch.bool) if mask is None else mask, confidence,
    )


def _simple_lift(features, support, *, grid=None, p=None, transform=None, weighted=False):
    batch = features.shape[0]
    return lift_camera_features_to_bev(
        features, depth_support=support,
        camera_projection=torch.eye(3, 4, dtype=features.dtype).repeat(batch, 1, 1) if p is None else p,
        t_vehicle_to_camera=torch.eye(4, dtype=features.dtype).repeat(batch, 1, 1) if transform is None else transform,
        grid=grid or _grid(), weight_by_splat_confidence=weighted,
    )


def _point_lift(point, *, value=9., grid=None):
    """Choose P2 so pixel (0,0), at depth z, maps to known XYZ."""

    x, y, z = point
    p = torch.eye(3, 4)[None]
    p[0, 0, 3] = -x
    p[0, 1, 3] = -y
    return _simple_lift(
        torch.full((1, 1, 1, 1), value),
        _simple_support(torch.full((1, 1, 1, 1), z)), p=p, grid=grid,
    )


def test_grid_shape_and_nonzero_origin_cell_placement():
    grid = _grid(x_range_m=(-2., 2.), y_range_m=(10., 13.), resolution_m=0.5)
    assert grid.shape == (8, 6)
    result = _point_lift([-1.25, 12.25, 1.], value=7., grid=grid)
    assert result.features.shape == (1, 1, 8, 6)
    assert result.features[0, 0, 1, 4] == 7
    assert result.pixel_count.sum() == 1


@pytest.mark.parametrize("point,accepted,cell", [
    ([0., -2., 1.], True, (0, 0)),
    ([1., 0., 1.], True, (1, 2)),
    ([3.99, 1.99, 1.], True, (3, 3)),
    ([4., 0., 1.], False, None),
    ([-0.01, 0., 1.], False, None),
    ([1., 2., 1.], False, None),
    ([1., -2.01, 1.], False, None),
    ([1., 0., 3.], False, None),
])
def test_half_open_bounds_and_expected_cells(point, accepted, cell):
    result = _point_lift(point)
    assert result.pixel_mask.item() == accepted
    assert result.pixel_count.sum() == int(accepted)
    if accepted:
        assert result.features[0, 0, *cell] == 9
    else:
        assert torch.all(result.features == 0)


def test_height_filter_uses_vehicle_z_not_camera_depth():
    # Vehicle = camera + [0, 0, -2]: camera z=1 -> vehicle z=-1.
    transform = torch.eye(4)[None]
    transform[:, 2, 3] = 2
    features = torch.ones(1, 1, 1, 1)
    support = _simple_support(torch.ones(1, 1, 1, 1))
    at_lower = _simple_lift(features, support, transform=transform)
    assert at_lower.pixel_mask.item()
    rejected = _simple_lift(features, support, transform=transform, grid=_grid(z_range_m=(-0.9, 3.)))
    assert not rejected.pixel_mask.item()


def test_collisions_use_mean_and_batches_do_not_mix():
    features = torch.tensor([[[[2., 8.]], [[4., 10.]]], [[[20., 80.]], [[40., 100.]]]])
    depth = torch.ones(2, 1, 1, 2)
    p = torch.eye(3, 4).repeat(2, 1, 1)
    p[:, 0, 0] = 10  # x=0 and x=0.1 -> same cell.
    result = _simple_lift(features, _simple_support(depth), p=p)
    torch.testing.assert_close(result.features[:, :, 0, 2], torch.tensor([[5., 7.], [50., 70.]]))
    assert result.pixel_count[:, :, 0, 2].tolist() == [[2], [2]]
    assert result.weight_sum[:, :, 0, 2].tolist() == [[2.], [2.]]
    assert result.pixel_count.sum() == 4


def test_confidence_weighting_is_optional_and_normalizes_small_positive_masses():
    features = torch.tensor([[[[2., 8.]]]])
    depth = torch.ones(1, 1, 1, 2)
    confidence = torch.tensor([[[[1e-12, 3e-12]]]])
    support = _simple_support(depth, source="splatted", confidence=confidence)
    p = torch.eye(3, 4)[None]
    p[:, 0, 0] = 10
    plain = _simple_lift(features, support, p=p)
    weighted = _simple_lift(features, support, p=p, weighted=True)
    assert plain.features[0, 0, 0, 2] == 5.
    torch.testing.assert_close(weighted.features[0, 0, 0, 2], torch.tensor(6.5))
    assert weighted.pixel_count[0, 0, 0, 2] == 2
    torch.testing.assert_close(weighted.weight_sum[0, 0, 0, 2], torch.tensor(4e-12), atol=0, rtol=1e-6)


def test_false_mask_zero_placeholders_and_nonfinite_pixels_never_contribute():
    features = torch.tensor([[[[float("nan"), 4., 10., float("inf"), 12.]]]], requires_grad=True)
    depth = torch.tensor([[[[float("nan"), 0., 1., 1., 1.]]]])
    mask = torch.tensor([[[[False, False, True, True, True]]]])
    p = torch.eye(3, 4)[None]
    p[:, 0, 0] = 10
    result = _simple_lift(features, _simple_support(depth, mask=mask), p=p)
    assert result.pixel_mask.flatten().tolist() == [False, False, True, False, True]
    assert result.pixel_count.sum() == 2
    assert torch.isfinite(result.features).all()
    result.features.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.flatten().tolist() == [0., 0., 0.5, 0., 0.5]


def test_invalid_weight_is_selected_out_before_multiplication():
    features = torch.tensor([[[[2., 8., 20.]]]])
    depth = torch.ones(1, 1, 1, 3)
    confidence = torch.tensor([[[[1., float("nan"), -1.]]]])
    p = torch.eye(3, 4)[None]
    p[:, 0, 0] = 10
    result = _simple_lift(features, _simple_support(depth, source="splatted", confidence=confidence), p=p, weighted=True)
    assert result.pixel_count.sum() == 1
    assert result.features[0, 0, 0, 2] == 2


def test_empty_support_has_zero_output_and_a_zero_feature_backward():
    features = torch.ones(2, 3, 1, 2, requires_grad=True)
    depth = torch.ones(2, 1, 1, 2)
    result = _simple_lift(features, _simple_support(depth, mask=torch.zeros_like(depth, dtype=torch.bool)))
    assert torch.all(result.features == 0)
    assert torch.all(result.pixel_count == 0)
    assert torch.all(result.weight_sum == 0)
    result.features.sum().backward()
    assert torch.all(features.grad == 0)


def test_zero_feature_vectors_are_distinguished_from_empty_cells():
    result = _simple_lift(torch.zeros(1, 2, 1, 1), _simple_support(torch.ones(1, 1, 1, 1)))
    assert torch.all(result.features == 0)
    assert result.pixel_count[0, 0, 0, 2] == 1


def test_lift_respects_full_p2_translation_and_vehicle_transform():
    # pixel (u=0,v=0), camera z=2, P translation [-1,-3,0.5]
    # -> camera XYZ [1,3,2]; vehicle XYZ [2,1,1] after inverse T.
    p = torch.eye(3, 4)[None]
    p[0, :, 3] = torch.tensor([-1., -3., 0.5])
    transform = torch.eye(4)[None]
    transform[0, :3, 3] = torch.tensor([-1., 2., 1.])
    result = _simple_lift(torch.tensor([[[[7.]]]]), _simple_support(torch.full((1, 1, 1, 1), 2.)), p=p, transform=transform)
    assert result.features[0, 0, 2, 3] == 7.


def test_rotated_kitti_axes_place_forward_left_and_up_correctly():
    p = torch.eye(3, 4)[None]
    p[0, :2, 3] = torch.tensor([-1., -2.])
    transform = torch.tensor([[
        [0., -1., 0., 0.], [0., 0., -1., 0.],
        [1., 0., 0., 0.], [0., 0., 0., 1.],
    ]])
    # Camera [1,2,5] -> vehicle [5,-1,-2], cell [5,2].
    result = _simple_lift(
        torch.ones(1, 1, 1, 1), _simple_support(torch.full((1, 1, 1, 1), 5.)),
        p=p, transform=transform,
        grid=_grid(x_range_m=(0., 6.), y_range_m=(-3., 3.), z_range_m=(-3., 0.)),
    )
    assert result.features[0, 0, 5, 2] == 1.


def test_rejected_and_accepted_pixels_receive_the_expected_weighted_gradient():
    features = torch.tensor([[[[2., 8., 4.]]]], requires_grad=True)
    depth = torch.ones(1, 1, 1, 3)
    support = _simple_support(
        depth, source="splatted", confidence=torch.tensor([[[[0.25, 0.75, 0.]]]]),
    )
    p = torch.eye(3, 4)[None]
    p[:, 0, 0] = 10.
    _simple_lift(features, support, p=p, weighted=True).features.sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([[[[0.25, 0.75, 0.]]]]))


def test_feature_backward_matches_numerical_derivative_but_not_depth_coordinates():
    features = torch.rand(1, 2, 1, 2, dtype=torch.float64, requires_grad=True)
    depth = torch.ones(1, 1, 1, 2, dtype=torch.float64, requires_grad=True)
    support = _simple_support(depth)
    p = torch.eye(3, 4, dtype=torch.float64)[None]
    p[:, 0, 0] = 10
    assert torch.autograd.gradcheck(lambda f: _simple_lift(f, support, p=p).features, (features,))
    _simple_lift(features, support, p=p).features.square().sum().backward()
    assert features.grad.abs().sum() > 0
    assert depth.grad is None


@pytest.mark.parametrize("source", ["sparse", "splatted", "predicted"])
def test_end_to_end_backbone_support_and_bev_preserve_depth_contract(source):
    torch.manual_seed(13)
    model = CameraDepthModel().eval()
    rgb = torch.rand(1, 3, 7, 9, requires_grad=True)
    lidar = torch.zeros(1, 8, 7, 9)
    lidar[:, 0] = 0.25
    lidar[:, 5] = 1.
    lidar[:, 6] = 0.25
    lidar[:, 7] = 0.5
    keys_before = tuple(model.state_dict())
    features = model.forward_features(rgb, lidar)
    depth_prediction = model.depth_head(features)
    options = {}
    if source == "predicted":
        # Supply known positive depth to isolate geometric plumbing, not model quality.
        options = dict(depth_prediction=torch.full_like(depth_prediction, 0.25),
                       prediction_support_mask=torch.ones_like(depth_prediction, dtype=torch.bool))
    support = prepare_camera_depth_support(
        lidar, source=source, max_depth_m=80., max_valid_depth_m=40., **options,
    )
    p = torch.tensor([[[10., 0, 4, 0], [0, 10., 3, 0], [0, 0, 1, 0]]])
    result = _simple_lift(features, support, p=p, grid=_grid(
        x_range_m=(-20., 20.), y_range_m=(-20., 20.), z_range_m=(0., 40.), resolution_m=5.,
    ))
    assert result.features.shape == (1, 32, 8, 8)
    assert result.pixel_count.sum() == 63
    result.features.square().mean().backward()
    assert torch.isfinite(rgb.grad).all() and rgb.grad.abs().sum() > 0
    for block in (model.stem, model.encoder_1, model.encoder_2, model.decoder_1, model.decoder_0):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        assert sum(p.grad.abs().sum() for p in block.parameters()) > 0
    assert all(p.grad is None for p in model.depth_head.parameters())
    assert tuple(model.state_dict()) == keys_before
    with torch.no_grad():
        torch.testing.assert_close(model(rgb, lidar), depth_prediction, rtol=0, atol=0)


@pytest.mark.parametrize("overrides", [
    {"resolution_m": 0.}, {"resolution_m": float("nan")},
    {"x_range_m": (2., 1.)}, {"y_range_m": (0., float("inf"))},
    {"z_range_m": (1., 1.)}, {"resolution_m": 0.3},
])
def test_grid_rejects_invalid_bounds_or_fractional_cells(overrides):
    with pytest.raises(ValueError):
        _grid(**overrides)


def test_weighting_cannot_assign_splat_uncertainty_to_predicted_depth():
    with pytest.raises(ValueError, match="splatted depth source"):
        _simple_lift(torch.ones(1, 1, 1, 1), _simple_support(torch.ones(1, 1, 1, 1), source="predicted"), weighted=True)


@pytest.mark.parametrize("features", [torch.ones(1, 1, 1), torch.empty(1, 0, 1, 1), torch.ones(1, 1, 1, 1).half()])
def test_lift_rejects_invalid_feature_contract(features):
    with pytest.raises(ValueError, match="image_features"):
        _simple_lift(features, _simple_support(torch.ones(1, 1, 1, 1)))


def test_lift_rejects_grid_or_dtype_mismatch_without_implicit_resize():
    with pytest.raises(ValueError, match="depth_m must share"):
        _simple_lift(torch.ones(1, 1, 2, 3), _simple_support(torch.ones(1, 1, 1, 1)))
    with pytest.raises(ValueError, match="depth_m must share"):
        _simple_lift(torch.ones(1, 1, 1, 1), _simple_support(torch.ones(1, 1, 1, 1).double()))


def test_lift_rejects_invalid_mask_and_confidence_shapes():
    features = torch.ones(1, 1, 1, 1)
    depth = torch.ones_like(features)
    with pytest.raises(ValueError, match="valid_mask must be bool"):
        _simple_lift(features, _simple_support(depth, mask=torch.ones_like(depth)))
    with pytest.raises(ValueError, match="splat_confidence must share"):
        _simple_lift(features, _simple_support(
            depth, source="splatted", confidence=torch.ones(1, 1, 2, 3),
        ), weighted=True)
