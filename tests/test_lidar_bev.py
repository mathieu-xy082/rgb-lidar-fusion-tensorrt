from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.lidar_bev import PillarFeatureNet, scatter_pillar_features
from rgb_lidar_fusion.lidar_pillars import build_lidar_pillars


def _sample_features():
    return torch.arange(3 * 4 * 9, dtype=torch.float32).reshape(3, 4, 9) / 20


def _mask():
    return torch.tensor([[True, True, False, False], [False, True, False, False], [False] * 4])


def test_pfn_matches_shared_linear_bn_relu_and_masked_per_pillar_max():
    torch.manual_seed(7)
    model = PillarFeatureNet()
    reference = deepcopy(model)
    features, mask = _sample_features(), _mask()
    expected_points = reference.norm(reference.linear(features[mask])).relu()
    expected = torch.stack([
        expected_points[:2].max(dim=0).values, expected_points[2], torch.zeros(32),
    ])

    actual = model(features, mask)

    assert model.linear.bias is None
    assert actual.shape == (3, 32)
    torch.testing.assert_close(actual, expected)
    for name, value in model.norm.state_dict().items():
        torch.testing.assert_close(value, reference.norm.state_dict()[name])
    assert model.norm.num_batches_tracked == 1


@pytest.mark.parametrize("training", [False, True])
def test_padding_values_and_extra_masked_slots_never_affect_bn_or_pooling(training):
    torch.manual_seed(13)
    model = PillarFeatureNet(out_channels=7).train(training)
    expanded_model = deepcopy(model)
    features, mask = _sample_features(), _mask()
    expanded = torch.cat([features, torch.full((3, 17, 9), float("nan"))], dim=1)
    expanded_mask = torch.cat([mask, torch.zeros(3, 17, dtype=torch.bool)], dim=1)
    expanded[~expanded_mask] = float("inf")
    features[~mask] = -1e30
    original = features.clone()

    expected = model(features, mask)
    actual = expanded_model(expanded, expanded_mask)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(features, original, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expanded_model.state_dict()[name], rtol=0, atol=0)
    assert torch.all(actual[2] == 0)


@pytest.mark.parametrize("training", [False, True])
def test_point_permutation_preserves_pillar_vectors(training):
    torch.manual_seed(17)
    model = PillarFeatureNet().train(training)
    permuted_model = deepcopy(model)
    features, mask = _sample_features(), _mask()
    permutation = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(model(features, mask), permuted_model(features[:, permutation], mask[:, permutation]))
    torch.testing.assert_close(model.norm.running_mean, permuted_model.norm.running_mean)
    torch.testing.assert_close(model.norm.running_var, permuted_model.norm.running_var)


def test_single_valid_point_in_training_uses_running_stats_without_updating_them():
    model = PillarFeatureNet(out_channels=4)
    with torch.no_grad():
        model.linear.weight.fill_(0.1)
        model.norm.running_mean.fill_(0.2)
        model.norm.running_var.fill_(2.)
        model.norm.weight.fill_(1.5)
        model.norm.bias.fill_(0.3)
    features = torch.ones(2, 3, 9, requires_grad=True)
    mask = torch.tensor([[False] * 3, [False, True, False]])
    before = {name: value.clone() for name, value in model.norm.state_dict().items()}
    expected = ((model.linear(features[mask]) - 0.2) / (2. + model.norm.eps) ** 0.5 * 1.5 + 0.3).relu()

    actual = model(features, mask)
    torch.testing.assert_close(actual[1], expected[0])
    assert torch.all(actual[0] == 0)
    actual.sum().backward()
    assert torch.isfinite(features.grad).all() and features.grad[mask].abs().sum() > 0
    assert torch.all(features.grad[~mask] == 0)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    for name, value in before.items():
        torch.testing.assert_close(model.norm.state_dict()[name], value, rtol=0, atol=0)


def test_one_point_per_pillar_still_updates_batchnorm_when_multiple_points_exist():
    model = PillarFeatureNet()
    features = torch.rand(3, 1, 9)
    result = model(features, torch.ones(3, 1, dtype=torch.bool))
    assert result.shape == (3, 32)
    assert torch.isfinite(result).all()
    assert model.norm.num_batches_tracked == 1


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("pillar_count", [0, 2])
def test_empty_or_fully_masked_pillars_have_finite_zero_backward(training, pillar_count):
    model = PillarFeatureNet().train(training)
    features = torch.full((pillar_count, 5, 9), float("nan"), requires_grad=True)
    mask = torch.zeros(pillar_count, 5, dtype=torch.bool)
    before = {name: value.clone() for name, value in model.norm.state_dict().items()}

    result = model(features, mask)
    assert result.shape == (pillar_count, 32)
    assert torch.all(result == 0)
    result.sum().backward()
    assert torch.all(features.grad == 0)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.all(parameter.grad == 0)
    for name, value in before.items():
        torch.testing.assert_close(model.norm.state_dict()[name], value, rtol=0, atol=0)


def test_pfn_backward_only_reaches_valid_features_and_has_trainable_parameters():
    torch.manual_seed(19)
    model = PillarFeatureNet()
    mask = _mask()
    features = _sample_features().masked_fill(~mask[..., None], float("nan")).requires_grad_()
    result = model(features, mask)
    result.square().sum().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad[mask].abs().sum() > 0
    assert torch.all(features.grad[~mask] == 0)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_pfn_supports_double_precision_and_numerical_feature_gradients():
    torch.manual_seed(23)
    model = PillarFeatureNet(out_channels=3).double().eval()
    features = torch.rand(2, 3, 9, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.autograd.gradcheck(lambda values: model(values, mask), (features,))


@pytest.mark.parametrize("channels", [0, -1, 1.5, True])
def test_invalid_channel_count_is_rejected(channels):
    with pytest.raises(ValueError, match="positive integer"):
        PillarFeatureNet(out_channels=channels)


@pytest.mark.parametrize("features", [torch.zeros(3, 9), torch.zeros(2, 0, 9), torch.zeros(2, 3, 8)])
def test_pfn_rejects_invalid_feature_shape(features):
    with pytest.raises(ValueError, match="shape"):
        PillarFeatureNet()(features, torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16])
def test_pfn_rejects_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="dtype"):
        PillarFeatureNet()(torch.zeros(2, 3, 9, dtype=dtype), torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("mask", [torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool)])
def test_pfn_rejects_invalid_mask(mask):
    with pytest.raises(ValueError, match="point_mask"):
        PillarFeatureNet()(torch.zeros(2, 3, 9), mask)


def test_pfn_rejects_dtype_mismatch_without_implicit_cast():
    with pytest.raises(ValueError, match="module's device"):
        PillarFeatureNet()(torch.zeros(2, 3, 9, dtype=torch.float64), torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("field", ["features", "mask"])
def test_pfn_rejects_device_mismatch_before_computation(field):
    features = torch.zeros(2, 3, 9, device="meta" if field == "features" else "cpu")
    mask = torch.ones(2, 3, dtype=torch.bool, device="meta" if field == "mask" else "cpu")
    with pytest.raises(ValueError, match="module's device"):
        PillarFeatureNet()(features, mask)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_pfn_rejects_nonfinite_selected_values_without_changing_bn(value):
    model = PillarFeatureNet()
    features = torch.zeros(1, 2, 9)
    features[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        model(features, torch.ones(1, 2, dtype=torch.bool))
    assert model.norm.num_batches_tracked == 0


def _scatter_inputs():
    features = torch.tensor([[2., 3.], [5., 7.], [11., 13.]], requires_grad=True)
    indices = torch.tensor([[0, 1, 4], [0, 6, 1], [2, 1, 4]])
    counts = torch.zeros(3, 1, 8, 6, dtype=torch.int64)
    counts[0, 0, 1, 4], counts[0, 0, 6, 1], counts[2, 0, 1, 4] = 90, 3, 1
    return features, indices, counts


def test_scatter_preserves_non_square_grid_batch_axes_and_measurement_counts():
    features, indices, counts = _scatter_inputs()
    original_counts = counts.clone()
    result = scatter_pillar_features(features, indices=indices, point_count=counts)
    assert result.features.shape == (3, 2, 8, 6)
    torch.testing.assert_close(result.features[0, :, 1, 4], features[0])
    torch.testing.assert_close(result.features[0, :, 6, 1], features[1])
    torch.testing.assert_close(result.features[2, :, 1, 4], features[2])
    assert torch.all(result.features[1] == 0)
    assert torch.all(result.features.permute(0, 2, 3, 1)[~result.observed_mask[:, 0]] == 0)
    torch.testing.assert_close(result.point_count, original_counts)
    assert result.point_count[0, 0, 1, 4] == 90
    assert result.observed_mask.sum() == 3
    result.features.sum().backward()
    torch.testing.assert_close(features.grad, torch.ones_like(features))
    counts.zero_()
    torch.testing.assert_close(result.point_count, original_counts)


def test_zero_pillar_features_are_not_mistaken_for_unobserved_cells():
    features, indices, counts = _scatter_inputs()
    result = scatter_pillar_features(torch.zeros_like(features), indices=indices, point_count=counts)
    assert torch.all(result.features == 0)
    assert result.observed_mask.sum() == 3


def test_scatter_does_not_depend_on_pillar_order():
    features, indices, counts = _scatter_inputs()
    permutation = torch.tensor([2, 0, 1])
    first = scatter_pillar_features(features, indices=indices, point_count=counts)
    second = scatter_pillar_features(features[permutation], indices=indices[permutation], point_count=counts)
    torch.testing.assert_close(first.features, second.features, rtol=0, atol=0)
    torch.testing.assert_close(first.point_count, second.point_count, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [0, 2])
def test_empty_scatter_has_zero_output_and_finite_backward(batch_size):
    features = torch.empty(0, 4, requires_grad=True)
    counts = torch.zeros(batch_size, 1, 5, 3, dtype=torch.int64)
    result = scatter_pillar_features(features, indices=torch.empty(0, 3, dtype=torch.int64), point_count=counts)
    assert result.features.shape == (batch_size, 4, 5, 3)
    assert torch.all(result.features == 0)
    assert not result.observed_mask.any()
    result.features.sum().backward()
    assert features.grad.shape == (0, 4)


def test_scatter_double_precision_gradcheck():
    features, indices, counts = _scatter_inputs()
    features = features.detach().double().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda values: scatter_pillar_features(values, indices=indices, point_count=counts).features,
        (features,),
    )


@pytest.mark.parametrize("bad_indices,match", [
    ([[0, 1, 4], [0, 1, 4], [2, 1, 4]], "unique"),
    ([[0, -1, 4], [0, 6, 1], [2, 1, 4]], "bounds"),
    ([[0, 8, 4], [0, 6, 1], [2, 1, 4]], "bounds"),
    ([[0, 1, 6], [0, 6, 1], [2, 1, 4]], "bounds"),
    ([[3, 1, 4], [0, 6, 1], [2, 1, 4]], "bounds"),
    ([[0, 0, 0], [0, 6, 1], [2, 1, 4]], "positive-count"),
])
def test_scatter_rejects_invalid_or_ambiguous_indices(bad_indices, match):
    features, _, counts = _scatter_inputs()
    with pytest.raises(ValueError, match=match):
        scatter_pillar_features(features, indices=torch.tensor(bad_indices), point_count=counts)


def test_scatter_rejects_missing_observed_pillars():
    features, indices, counts = _scatter_inputs()
    with pytest.raises(ValueError, match="cover every"):
        scatter_pillar_features(features[:2], indices=indices[:2], point_count=counts)


@pytest.mark.parametrize("field", ["features", "indices", "counts"])
def test_scatter_rejects_bad_shapes(field):
    features, indices, counts = _scatter_inputs()
    if field == "features":
        features = features[:, None]
    elif field == "indices":
        indices = indices[:, :2]
    else:
        counts = counts[:, 0]
    with pytest.raises(ValueError, match="shape"):
        scatter_pillar_features(features, indices=indices, point_count=counts)


@pytest.mark.parametrize("field", ["features", "indices", "counts"])
def test_scatter_rejects_bad_dtypes(field):
    features, indices, counts = _scatter_inputs()
    if field == "features":
        features = features.half()
    elif field == "indices":
        indices = indices.int()
    else:
        counts = counts.float()
    with pytest.raises(ValueError, match="dtype|int64"):
        scatter_pillar_features(features, indices=indices, point_count=counts)


@pytest.mark.parametrize("field", ["features", "indices", "counts"])
def test_scatter_rejects_device_mismatch_before_computation(field):
    features, indices, counts = _scatter_inputs()
    if field == "features":
        features = features.to("meta")
    elif field == "indices":
        indices = indices.to("meta")
    else:
        counts = counts.to("meta")
    with pytest.raises(ValueError, match="same device"):
        scatter_pillar_features(features, indices=indices, point_count=counts)


def test_scatter_rejects_negative_measurement_counts():
    features, indices, counts = _scatter_inputs()
    counts[1, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        scatter_pillar_features(features, indices=indices, point_count=counts)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_scatter_rejects_nonfinite_pillar_vectors(value):
    features, indices, counts = _scatter_inputs()
    features = features.detach()
    features[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        scatter_pillar_features(features, indices=indices, point_count=counts)


@pytest.mark.parametrize("empty", [False, True])
def test_raw_pillars_to_bev_preserves_evidence_and_finite_backward(empty):
    torch.manual_seed(29)
    grid = BEVGridConfig((0., 1.), (-0.4, 0.2), (-3., 3.), 0.2)
    cloud = np.array([[0.05, -0.35, -1., .2], [.08, -.32, 1., .8], [.45, .1, 0., .5]], dtype=np.float32)
    if empty:
        cloud = np.empty((0, 4), dtype=np.float32)
    pillars = build_lidar_pillars([cloud, np.empty((0, 4), dtype=np.float32)], grid=grid)
    features = torch.from_numpy(pillars.features).requires_grad_()
    model = PillarFeatureNet()
    pillar_vectors = model(features, torch.from_numpy(pillars.point_mask))
    result = scatter_pillar_features(
        pillar_vectors, indices=torch.from_numpy(pillars.indices),
        point_count=torch.from_numpy(pillars.point_count),
    )
    assert result.features.shape == (2, 32, 5, 3)
    assert torch.isfinite(result.features).all()
    np.testing.assert_array_equal(result.point_count.numpy(), pillars.point_count)
    np.testing.assert_array_equal(result.observed_mask.numpy(), pillars.observed_mask)
    result.features.square().sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert torch.all(features.grad[~torch.from_numpy(pillars.point_mask)] == 0)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if not empty:
            assert parameter.grad.abs().sum() > 0
