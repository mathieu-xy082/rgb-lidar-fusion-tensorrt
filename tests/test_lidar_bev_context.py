import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.lidar_bev import (
    LidarBEVContext, LidarBEVFeatures, PillarFeatureNet, scatter_pillar_features,
)
from rgb_lidar_fusion.lidar_pillars import build_lidar_pillars


def _bev(features, counts=None):
    if counts is None:
        counts = torch.ones(features.shape[0], 1, *features.shape[2:], dtype=torch.int64, device=features.device)
    return LidarBEVFeatures(features, counts)


def _positive_weights(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.)
    return model


def _comparison_class():
    spec = importlib.util.spec_from_file_location("context_comparison", "scripts/compare_lidar_bev_context.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._MultiResolutionContext


def test_context_structure_width_normalization_and_parameters():
    model = LidarBEVContext()
    convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    assert [module.dilation for module in convolutions] == [(1, 1), (2, 2), (2, 2), (4, 4)]
    assert model.receptive_field_cells == 19
    for module in convolutions:
        assert module.in_channels == module.out_channels == 32
        assert module.kernel_size == (3, 3) and module.stride == (1, 1)
        assert module.padding == module.dilation and module.bias is None
    assert not any(isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm)) for module in model.modules())
    assert sum(parameter.numel() for parameter in model.parameters()) == 36864


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (5, 8), (9, 7), (250, 250), (251, 249)])
def test_context_preserves_grid_shapes_and_evidence_without_input_mutation(shape):
    torch.manual_seed(13)
    model = LidarBEVContext(channels=2)
    features = torch.rand(2, 2, *shape)
    counts = torch.zeros(2, 1, *shape, dtype=torch.int64)
    counts[0, 0, 0, 0] = 90
    before = features.clone()
    result = model(_bev(features, counts))
    assert result.features.shape == features.shape
    assert torch.isfinite(result.features).all()
    torch.testing.assert_close(features, before, rtol=0, atol=0)
    torch.testing.assert_close(result.point_count, counts, rtol=0, atol=0)
    counts.zero_()
    assert result.point_count[0, 0, 0, 0] == 90


@pytest.mark.parametrize("training", [False, True])
def test_exact_19_by_19_receptive_field_on_controlled_gradient(training):
    model = _positive_weights(LidarBEVContext(channels=1)).train(training)
    features = torch.ones(1, 1, 41, 45, requires_grad=True)
    result = model(_bev(features))
    result.features[0, 0, 20, 22].backward()
    expected = torch.zeros_like(features, dtype=torch.bool)
    expected[:, :, 11:30, 13:32] = True
    assert torch.equal(features.grad > 0, expected)
    assert expected.sum() == 19 * 19
    assert torch.isfinite(features.grad).all()


def test_centered_impulse_spreads_features_without_creating_new_observations():
    model = _positive_weights(LidarBEVContext(channels=1))
    features = torch.zeros(1, 1, 41, 45)
    features[0, 0, 20, 22] = 1.
    counts = torch.zeros(1, 1, 41, 45, dtype=torch.int64)
    counts[0, 0, 20, 22] = 70
    result = model(_bev(features, counts))
    expected = torch.zeros_like(features, dtype=torch.bool)
    expected[:, :, 11:30, 13:32] = True
    assert torch.equal(result.features > 0, expected)
    assert result.observed_mask.sum() == 1
    assert torch.count_nonzero(result.features[~result.observed_mask]) == 360
    weights = result.features[0, 0]
    rows, cols = torch.meshgrid(torch.arange(41), torch.arange(45), indexing="ij")
    torch.testing.assert_close((weights * rows).sum() / weights.sum(), torch.tensor(20.))
    torch.testing.assert_close((weights * cols).sum() / weights.sum(), torch.tensor(22.))
    assert result.point_count.sum() == 70


def test_roi_edge_padding_does_not_wrap_features_to_opposite_borders():
    model = _positive_weights(LidarBEVContext(channels=1))
    features = torch.zeros(1, 1, 25, 27)
    features[0, 0, 0, 0] = 1.
    expected = torch.zeros_like(features, dtype=torch.bool)
    expected[:, :, :10, :10] = True
    assert torch.equal(model(_bev(features)).features > 0, expected)


def test_residual_path_preserves_positive_cell_locations_when_convolutions_are_zero():
    model = LidarBEVContext(channels=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    features = torch.rand(2, 2, 7, 11)
    torch.testing.assert_close(model(_bev(features)).features, features, rtol=0, atol=0)


def test_context_does_not_mix_batch_samples():
    model = _positive_weights(LidarBEVContext(channels=1))
    features = torch.zeros(2, 1, 23, 25)
    features[0, 0, 10, 12] = 1.
    result = model(_bev(features))
    assert result.features[0].sum() > 0
    assert torch.all(result.features[1] == 0)


@pytest.mark.parametrize("batch", [0, 2])
def test_zero_inputs_and_empty_batches_have_finite_zero_backward(batch):
    model = LidarBEVContext(channels=2)
    features = torch.zeros(batch, 2, 5, 7, requires_grad=True)
    counts = torch.zeros(batch, 1, 5, 7, dtype=torch.int64)
    result = model(_bev(features, counts))
    assert torch.all(result.features == 0) and not result.observed_mask.any()
    result.features.sum().backward()
    assert torch.all(features.grad == 0)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.all(parameter.grad == 0)


def test_context_has_finite_nonzero_feature_and_parameter_backward():
    torch.manual_seed(17)
    model = LidarBEVContext(channels=3)
    features = torch.rand(2, 3, 9, 11, requires_grad=True)
    result = model(_bev(features))
    result.features.square().mean().backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_double_precision_gradcheck():
    torch.manual_seed(19)
    model = LidarBEVContext(channels=1).double().eval()
    # Positive paths keep the numerical probe away from ReLU's zero boundary.
    _positive_weights(model)
    features = torch.rand(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: model(_bev(values)).features, (features,))


@pytest.mark.parametrize("channels", [0, -1, 1.5, True])
def test_invalid_width_is_rejected(channels):
    with pytest.raises(ValueError, match="positive integer"):
        LidarBEVContext(channels=channels)


@pytest.mark.parametrize("features", [torch.zeros(1, 2, 3), torch.zeros(1, 3, 4, 5), torch.zeros(1, 2, 0, 5)])
def test_invalid_feature_shapes_are_rejected(features):
    with pytest.raises(ValueError, match="shape"):
        LidarBEVContext(channels=2)(_bev(features))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int64])
def test_invalid_feature_dtype_is_rejected(dtype):
    with pytest.raises(ValueError, match="dtype"):
        LidarBEVContext(channels=2)(_bev(torch.zeros(1, 2, 3, 4, dtype=dtype)))


def test_module_dtype_mismatch_is_rejected():
    with pytest.raises(ValueError, match="module's device and dtype"):
        LidarBEVContext(channels=2)(_bev(torch.zeros(1, 2, 3, 4, dtype=torch.float64)))


@pytest.mark.parametrize("field", ["features", "counts"])
def test_device_mismatch_is_rejected_before_computation(field):
    features = torch.ones(1, 2, 3, 4, device="meta" if field == "features" else "cpu")
    counts = torch.ones(1, 1, 3, 4, dtype=torch.int64, device="meta" if field == "counts" else "cpu")
    with pytest.raises(ValueError, match="device"):
        LidarBEVContext(channels=2)(_bev(features, counts))


@pytest.mark.parametrize("counts", [torch.ones(1, 3, 4), torch.ones(1, 1, 3, 4), torch.ones(1, 1, 3, 5, dtype=torch.int64)])
def test_invalid_count_contract_is_rejected(counts):
    with pytest.raises(ValueError, match="point_count"):
        LidarBEVContext(channels=2)(_bev(torch.ones(1, 2, 3, 4), counts))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_features_are_rejected(value):
    features = torch.ones(1, 2, 3, 4)
    features[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        LidarBEVContext(channels=2)(_bev(features))


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        LidarBEVContext(channels=2)(_bev(torch.ones(1, 2, 3, 4), -torch.ones(1, 1, 3, 4, dtype=torch.int64)))


@pytest.mark.parametrize("empty", [False, True])
def test_raw_pillars_pfn_scatter_and_context_end_to_end_backward(empty):
    torch.manual_seed(23)
    grid = BEVGridConfig((0., 1.), (-0.4, 0.2), (-3., 3.), .2)
    cloud = np.array([[.05, -.35, -1., .2], [.08, -.32, 1., .8], [.45, .1, 0., .5]], dtype=np.float32)
    if empty:
        cloud = np.empty((0, 4), dtype=np.float32)
    pillars = build_lidar_pillars([cloud], grid=grid)
    features = torch.from_numpy(pillars.features).requires_grad_()
    pfn, context = PillarFeatureNet(), LidarBEVContext()
    scattered = scatter_pillar_features(
        pfn(features, torch.from_numpy(pillars.point_mask)),
        indices=torch.from_numpy(pillars.indices), point_count=torch.from_numpy(pillars.point_count),
    )
    result = context(scattered)
    assert result.features.shape == (1, 32, 5, 3) and torch.isfinite(result.features).all()
    np.testing.assert_array_equal(result.point_count.numpy(), pillars.point_count)
    np.testing.assert_array_equal(result.observed_mask.numpy(), pillars.observed_mask)
    result.features.square().sum().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.all(features.grad[~torch.from_numpy(pillars.point_mask)] == 0)
    for module in (pfn, context):
        for parameter in module.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            if not empty:
                assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (7, 10), (11, 9)])
def test_multiresolution_reference_restores_explicit_output_size_and_backward(shape):
    model = _comparison_class()(2)
    features = torch.rand(1, 2, *shape, requires_grad=True)
    bev = _bev(features)
    result = model(bev)
    assert result.features.shape == features.shape and torch.isfinite(result.features).all()
    torch.testing.assert_close(result.point_count, bev.point_count)
    result.features.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("row,col,field", [(20, 22, (17, 17)), (21, 22, (19, 17)), (21, 23, (19, 19))])
def test_multiresolution_reference_receptive_field_depends_on_output_parity(row, col, field):
    model = _positive_weights(_comparison_class()(1))
    features = torch.ones(1, 1, 43, 47, requires_grad=True)
    model(_bev(features)).features[0, 0, row, col].backward()
    expected = torch.zeros_like(features, dtype=torch.bool)
    rh, rw = field[0] // 2, field[1] // 2
    expected[:, :, row - rh:row + rh + 1, col - rw:col + rw + 1] = True
    assert torch.equal(features.grad > 0, expected)


def test_comparison_cli_reports_cpu_measurements_without_claiming_detector_quality(tmp_path):
    output = tmp_path / "context.json"
    completed = subprocess.run([
        sys.executable, "scripts/compare_lidar_bev_context.py", "--height", "7", "--width", "10",
        "--channels", "2", "--iterations", "1", "--warmup", "1", "--output", str(output),
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report == json.loads(output.read_text())
    assert report["device"] == "cpu" and report["shape"] == [1, 2, 7, 10]
    assert "Random weights" in report["scope"] and "peak-memory" in report["scope"]
    assert [item["parameter_count"] for item in report["results"]] == [144, 216]
    assert all(item["forward_ms"] > 0 and item["forward_backward_ms"] > 0 for item in report["results"])
