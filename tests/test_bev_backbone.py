from dataclasses import replace
import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.bev_backbone import SharedBEVBackbone
from rgb_lidar_fusion.bev_fusion import BEVFeatureFusion, FusedBEVFeatures
from rgb_lidar_fusion.camera_bev import BEVGridConfig, CameraBEVFeatures
from rgb_lidar_fusion.lidar_bev import LidarBEVFeatures


def _bev(features, *, camera_count=None, lidar_count=None):
    shape = (features.shape[0], 1, *features.shape[2:])
    if camera_count is None:
        camera_count = torch.ones(shape, dtype=torch.int64, device=features.device)
    if lidar_count is None:
        lidar_count = torch.ones(shape, dtype=torch.int64, device=features.device) * 90
    grid = BEVGridConfig((1., 1. + features.shape[2] * .2),
                        (-2., -2. + features.shape[3] * .2), (-3., 3.), .2)
    return FusedBEVFeatures(features, grid, camera_count,
                            camera_count.to(features.dtype) * .25, lidar_count)


def _positive_model():
    model = SharedBEVBackbone(channels=(1, 1, 1))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.1)
    return model


def test_default_architecture_widths_and_parameter_budget():
    model = SharedBEVBackbone()
    assert model.channels == (32, 64, 96)
    convolutions = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    transposed = [m for m in model.modules() if isinstance(m, torch.nn.ConvTranspose2d)]
    assert len(convolutions) == 10 and len(transposed) == 2
    assert (model.down1.in_channels, model.down1.out_channels) == (32, 64)
    assert (model.down2.in_channels, model.down2.out_channels) == (64, 96)
    assert (model.up1.in_channels, model.up1.out_channels) == (96, 64)
    assert (model.up0.in_channels, model.up0.out_channels) == (64, 32)
    for layer in convolutions + transposed:
        assert layer.kernel_size == (3, 3) and layer.padding == (1, 1)
        assert layer.bias is None
    assert model.down1.stride == model.down2.stride == (2, 2)
    assert model.up1.stride == model.up0.stride == (2, 2)
    assert not any(isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.LayerNorm))
                   for m in model.modules())
    assert sum(p.numel() for p in model.parameters()) == 423936


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1), (2, 3), (5, 8), (9, 7), (250, 250), (251, 249)])
def test_preserves_odd_even_grid_shapes_and_independent_evidence(shape):
    bev = _bev(torch.rand(2, 2, *shape))
    model = SharedBEVBackbone(channels=(2, 3, 4))
    inputs = [bev.features, bev.camera_pixel_count, bev.camera_weight_sum, bev.lidar_point_count]
    saved = [value.clone() for value in inputs]
    result = model(bev)
    assert result.grid == bev.grid and result.features.shape == bev.features.shape
    assert torch.isfinite(result.features).all()
    for actual, expected in zip(inputs, saved, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, source in (
        (result.camera_pixel_count, bev.camera_pixel_count),
        (result.camera_weight_sum, bev.camera_weight_sum),
        (result.lidar_point_count, bev.lidar_point_count),
    ):
        torch.testing.assert_close(actual, source, rtol=0, atol=0)
        reference = source.clone()
        actual.zero_()
        torch.testing.assert_close(source, reference, rtol=0, atol=0)
        actual.copy_(reference)
        source.zero_()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (7, 10), (11, 9), (250, 250), (251, 249)])
def test_center_taps_keep_the_same_lattice_across_both_down_up_pairs(shape):
    model = SharedBEVBackbone(channels=(1, 1, 1))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for layer in (model.down1, model.down2, model.up1, model.up0):
            layer.weight[0, 0, 1, 1] = 1
    features = torch.arange(1, 1 + shape[0] * shape[1], dtype=torch.float32).reshape(1, 1, *shape)
    expected = features.clone()
    expected[:, :, ::2, ::2] += features[:, :, ::2, ::2]
    expected[:, :, ::4, ::4] += features[:, :, ::4, ::4]
    torch.testing.assert_close(model(_bev(features)).features, expected, rtol=0, atol=0)


def test_encoder_and_decoder_intermediate_sizes_match_for_default_grid():
    model = SharedBEVBackbone()
    shapes = {}
    def record(name):
        def hook(module, args, output):
            shapes[name] = tuple(output.shape)
        return hook
    handles = [getattr(model, name).register_forward_hook(record(name))
               for name in ("fine", "middle", "deep", "up1", "up0")]
    try:
        with torch.no_grad():
            result = model(_bev(torch.zeros(1, 32, 250, 250)))
    finally:
        for handle in handles:
            handle.remove()
    assert shapes == {"fine": (1, 32, 250, 250), "middle": (1, 64, 125, 125),
                      "deep": (1, 96, 63, 63), "up1": (1, 64, 125, 125),
                      "up0": (1, 32, 250, 250)}
    assert result.features.shape == (1, 32, 250, 250)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("row,col", [(40, 44), (41, 45), (42, 46), (43, 47), (40, 46)])
def test_controlled_gradients_verify_phase_dependent_receptive_field(training, row, col):
    model = _positive_model().train(training)
    features = torch.ones(1, 1, 89, 93, requires_grad=True)
    model(_bev(features)).features[0, 0, row, col].backward()
    # Independently specified ranges relative to each output's stride-4 phase.
    ranges = [(-21, 21), (-22, 20), (-19, 19), (-20, 22)]
    row_min, row_max = ranges[row % 4]
    col_min, col_max = ranges[col % 4]
    expected = torch.zeros_like(features, dtype=torch.bool)
    expected[:, :, row + row_min:row + row_max + 1, col + col_min:col + col_max + 1] = True
    assert torch.equal(features.grad > 0, expected)
    assert torch.isfinite(features.grad).all()
    assert row_max - row_min + 1 in (39, 43) and col_max - col_min + 1 in (39, 43)


def test_roi_corner_clips_context_without_wrapping_or_creating_observations():
    features = torch.ones(1, 1, 57, 61, requires_grad=True)
    counts = torch.zeros(1, 1, 57, 61, dtype=torch.int64)
    bev = _bev(features, camera_count=counts, lidar_count=counts.clone())
    result = _positive_model()(bev)
    result.features[0, 0, 0, 0].backward()
    expected = torch.zeros_like(features, dtype=torch.bool)
    expected[:, :, :22, :22] = True
    assert torch.equal(features.grad > 0, expected)
    assert not result.camera_support.any() and not result.lidar_observed_mask.any()
    assert result.features.min() > 0  # No evidence mask is reapplied.


def test_impulse_spreads_into_unobserved_cells_but_evidence_is_unchanged():
    features = torch.zeros(1, 1, 57, 61)
    features[0, 0, 28, 30] = 1
    counts = torch.zeros(1, 1, 57, 61, dtype=torch.int64)
    counts[0, 0, 28, 30] = 70
    bev = _bev(features, camera_count=torch.zeros_like(counts), lidar_count=counts)
    result = _positive_model()(bev)
    assert (result.features > 0).sum() > 19 * 19
    assert not result.camera_support.any() and result.lidar_observed_mask.sum() == 1
    assert result.lidar_point_count.sum() == 70
    assert result.features[~result.lidar_observed_mask].sum() > 0


def test_batch_samples_are_not_mixed():
    features = torch.zeros(2, 1, 25, 29)
    features[0, 0, 12, 14] = 1
    result = _positive_model()(_bev(features))
    assert result.features[0].sum() > 0 and torch.all(result.features[1] == 0)


def test_zero_convolutions_preserve_the_positive_fine_skip_path():
    model = SharedBEVBackbone(channels=(2, 3, 4))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    features = torch.rand(2, 2, 7, 11)
    torch.testing.assert_close(model(_bev(features)).features, features, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1), (7, 10), (11, 9)])
def test_zero_features_preserve_zero_response_and_finite_backward(shape):
    model = SharedBEVBackbone(channels=(2, 3, 4))
    features = torch.zeros(2, 2, *shape, requires_grad=True)
    result = model(_bev(features))
    assert torch.all(result.features == 0)
    result.features.sum().backward()
    assert torch.all(features.grad == 0)
    assert all(parameter.grad is not None and torch.all(parameter.grad == 0)
               for parameter in model.parameters())


def test_double_precision_feature_gradcheck():
    model = _positive_model().double()
    features = torch.ones(1, 1, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: model(_bev(values)).features, (features,))


def test_fusion_and_backbone_propagate_gradients_to_both_feature_branches():
    camera_features = torch.ones(1, 2, 9, 11, requires_grad=True)
    lidar_features = torch.ones(1, 3, 9, 11, requires_grad=True)
    counts = torch.ones(1, 1, 9, 11, dtype=torch.int64)
    camera = CameraBEVFeatures(camera_features, counts, counts.float(),
                               torch.ones(1, 1, 1, 1, dtype=torch.bool))
    lidar = LidarBEVFeatures(lidar_features, counts.clone())
    grid = _bev(camera_features).grid
    fusion = BEVFeatureFusion(camera_channels=2, lidar_channels=3, out_channels=1)
    backbone = _positive_model()
    with torch.no_grad():
        fusion.projection.weight.fill_(.1)
    fused = fusion(camera, lidar, camera_grid=grid, lidar_grid=grid)
    result = backbone(fused)
    result.features.square().mean().backward()
    for tensor in (camera_features, lidar_features):
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    for module in (fusion, backbone):
        for parameter in module.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("channels", [None, 32, True, (), (1, 2), (1, 2, 3, 4), (0, 2, 3),
                                      (1, -2, 3), (1, 2, True), (1., 2, 3)])
def test_invalid_widths_are_rejected(channels):
    with pytest.raises(ValueError, match="three positive integer widths"):
        SharedBEVBackbone(channels=channels)


@pytest.mark.parametrize("case", ["ndim", "batch", "channels", "shape"])
def test_malformed_feature_shapes_are_rejected(case):
    bev = _bev(torch.ones(2, 1, 5, 7))
    changed = {"ndim": bev.features[0], "batch": bev.features[:0],
               "channels": bev.features.repeat(1, 2, 1, 1), "shape": bev.features[:, :, :1]}[case]
    with pytest.raises(ValueError, match="non-empty shape"):
        _positive_model()(replace(bev, features=changed))


def test_missing_grid_is_rejected():
    with pytest.raises(ValueError, match="explicit BEVGridConfig"):
        _positive_model()(replace(_bev(torch.ones(1, 1, 3, 5)), grid=None))


@pytest.mark.parametrize("dtype", [torch.float16, torch.int64, torch.float64])
def test_invalid_or_mismatched_feature_dtypes_are_rejected(dtype):
    with pytest.raises(ValueError, match="dtype"):
        _positive_model()(_bev(torch.ones(1, 1, 3, 5, dtype=dtype)))


def test_feature_device_mismatch_is_rejected_before_data_operations():
    with pytest.raises(ValueError, match="device and dtype"):
        _positive_model()(_bev(torch.ones(1, 1, 3, 5, device="meta")))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_features_are_rejected(value):
    with pytest.raises(ValueError, match="features must be finite"):
        _positive_model()(_bev(torch.full((1, 1, 3, 5), value)))


@pytest.mark.parametrize("field", ["camera_pixel_count", "lidar_point_count"])
@pytest.mark.parametrize("case", ["shape", "dtype", "device", "negative"])
def test_invalid_count_evidence_is_rejected(field, case):
    bev = _bev(torch.ones(1, 1, 3, 5))
    original = getattr(bev, field)
    changed = {"shape": original[:, :, :1], "dtype": original.float(),
               "device": original.to("meta"), "negative": -original}[case]
    with pytest.raises(ValueError, match=field):
        _positive_model()(replace(bev, **{field: changed}))


@pytest.mark.parametrize("case", ["shape", "dtype", "device", "negative", "nan", "inf", "zero"])
def test_invalid_camera_mass_is_rejected(case):
    bev = _bev(torch.ones(1, 1, 3, 5))
    original = bev.camera_weight_sum
    changed = {"shape": original[:, :, :1], "dtype": original.double(),
               "device": original.to("meta"), "negative": -original,
               "nan": original * float("nan"), "inf": original * float("inf"),
               "zero": torch.zeros_like(original)}[case]
    with pytest.raises(ValueError, match="positive support" if case == "zero" else "camera_weight_sum"):
        _positive_model()(replace(bev, camera_weight_sum=changed))


def test_cpu_probe_cli_reports_bounded_synthetic_measurements(tmp_path):
    output = tmp_path / "backbone.json"
    completed = subprocess.run([
        sys.executable, "scripts/probe_bev_backbone.py", "--height", "7", "--width", "10",
        "--channels", "1", "2", "3", "--iterations", "1", "--warmup", "1", "--output", str(output),
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report == json.loads(output.read_text())
    assert report["device"] == "cpu" and report["shape"] == [1, 1, 7, 10]
    assert report["channels"] == [1, 2, 3] and report["spatial_shapes"] == [[7, 10], [4, 5], [2, 3]]
    assert report["receptive_field_cells_by_axis_phase"] == [43, 43, 39, 43]
    assert report["parameter_count"] == sum(p.numel() for p in SharedBEVBackbone(channels=(1, 2, 3)).parameters())
    assert report["parameter_storage_mib"] == report["parameter_count"] * 4 / 1024**2
    assert report["forward_ms"] > 0 and report["forward_backward_ms"] > 0
    assert "Synthetic random weights" in report["scope"] and "peak-memory" in report["scope"]
    assert "No encoders" in report["scope"] and "GPU" in report["scope"]
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("args", [["--height", "0"], ["--width", "-1"],
                                   ["--channels", "1", "0", "3"], ["--iterations", "0"]])
def test_cpu_probe_cli_rejects_invalid_sizes_before_running(args, tmp_path):
    output = tmp_path / "invalid.json"
    completed = subprocess.run([sys.executable, "scripts/probe_bev_backbone.py", *args,
                                "--output", str(output)], capture_output=True, text=True, check=False)
    assert completed.returncode == 2 and "positive integer" in completed.stderr
    assert not output.exists()
