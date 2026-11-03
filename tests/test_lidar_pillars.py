import subprocess
import sys

import numpy as np
import pytest

from rgb_lidar_fusion.lidar_pillars import PILLAR_FEATURE_CHANNELS, build_lidar_pillars
from rgb_lidar_fusion.project_lidar import load_velodyne_bin


@pytest.fixture
def grid():
    pytest.importorskip("torch")
    from rgb_lidar_fusion.camera_bev import BEVGridConfig

    return BEVGridConfig(
        x_range_m=(0., 1.), y_range_m=(-0.4, 0.2), z_range_m=(-3., 3.),
        resolution_m=0.2,
    )


def _cloud(rows):
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def test_nine_features_use_retained_centroid_and_metric_cell_center(grid):
    points = _cloud([[0.02, -0.38, -1., 0.25], [0.08, -0.32, 1., 0.75]])
    result = build_lidar_pillars([points], grid=grid, max_points_per_pillar=4)

    assert len(PILLAR_FEATURE_CHANNELS) == 9
    assert result.features.dtype == np.float32
    assert result.grid is grid
    assert result.features.shape == (1, 4, 9)
    np.testing.assert_array_equal(result.indices, [[0, 0, 0]])
    np.testing.assert_allclose(result.features[0, :2, :4], points)
    np.testing.assert_allclose(result.features[0, :2, 4:7], [
        [-0.03, -0.03, -1.], [0.03, 0.03, 1.],
    ], atol=1e-7)
    np.testing.assert_allclose(result.features[0, :2, 7:9], [
        [-0.08, -0.08], [-0.02, -0.02],
    ], atol=1e-7)
    np.testing.assert_array_equal(result.features[0, 2:], 0)
    np.testing.assert_array_equal(result.point_mask, [[True, True, False, False]])
    np.testing.assert_array_equal(result.point_indices, [[0, 1, -1, -1]])
    np.testing.assert_array_equal(result.retained_count, [2])
    np.testing.assert_array_equal(result.dropped_count, [0])
    assert result.point_count.dtype == np.int64
    assert result.point_count.shape == (1, 1, 5, 3)
    assert result.point_count[0, 0, 0, 0] == 2
    assert result.observed_mask.dtype == np.bool_
    assert result.observed_mask.sum() == 1


def test_non_square_grid_and_batches_do_not_mix_or_filter_camera_fov(grid):
    left_of_camera = [0.45, -0.35, 0., 0.1]
    points = _cloud([[0.85, 0.1, 1., 0.8], left_of_camera, [0.45, -0.35, 2., 0.2]])
    result = build_lidar_pillars([points, _cloud([]), _cloud([left_of_camera])], grid=grid)

    np.testing.assert_array_equal(result.indices, [[0, 2, 0], [0, 4, 2], [2, 2, 0]])
    np.testing.assert_array_equal(result.retained_count, [2, 1, 1])
    np.testing.assert_array_equal(result.point_indices[0, :2], [1, 2])
    assert result.point_count[0, 0, 2, 0] == 2
    assert result.point_count[2, 0, 2, 0] == 1
    assert result.point_count[1].sum() == 0
    assert result.point_count.sum() == 4


def test_nonzero_grid_origin_and_configurable_resolution():
    pytest.importorskip("torch")
    from rgb_lidar_fusion.camera_bev import BEVGridConfig

    grid = BEVGridConfig((-2., 2.), (10., 13.), (-1., 3.), 0.5)
    result = build_lidar_pillars([_cloud([[-1.25, 12.25, 1., 0.5]])], grid=grid)
    assert result.point_count.shape == (1, 1, 8, 6)
    np.testing.assert_array_equal(result.indices, [[0, 1, 4]])
    np.testing.assert_array_equal(result.features[0, 0, 7:9], [0., 0.])


@pytest.mark.parametrize("point,cell", [
    ([0., -0.4, -3., 0.], (0, 0)),
    ([0.2, -0.2, 0., 0.], (1, 1)),
    ([0.99, 0.19, 2.99, 0.], (4, 2)),
    ([-0.001, 0., 0., 0.], None),
    ([1., 0., 0., 0.], None),
    ([0.1, -0.401, 0., 0.], None),
    ([0.1, 0.2, 0., 0.], None),
    ([0.1, 0., -3.001, 0.], None),
    ([0.1, 0., 3., 0.], None),
])
def test_half_open_xyz_bounds_and_floor_cell_assignment(grid, point, cell):
    result = build_lidar_pillars([_cloud([point])], grid=grid)
    if cell is None:
        assert result.features.shape == (0, 64, 9)
        assert result.point_count.sum() == 0
    else:
        np.testing.assert_array_equal(result.indices, [[0, *cell]])
        assert result.point_count[0, 0, *cell] == 1


@pytest.mark.parametrize("empty_batch", [False, True])
def test_empty_clouds_and_empty_batch_have_explicit_shapes(grid, empty_batch):
    clouds = [] if empty_batch else [_cloud([]), _cloud([])]
    result = build_lidar_pillars(clouds, grid=grid)
    assert result.features.shape == (0, 64, 9)
    assert result.indices.shape == (0, 3)
    assert result.indices.dtype == np.int64
    assert result.point_mask.shape == result.point_indices.shape == (0, 64)
    assert result.point_indices.dtype == np.int64
    assert result.point_count.shape == (len(clouds), 1, 5, 3)
    assert result.observed_mask.shape == result.point_count.shape
    assert result.point_count.sum() == result.observed_mask.sum() == 0
    assert result.retained_count.shape == result.dropped_count.shape == (0,)


def test_overflow_is_reproducible_and_centroid_excludes_dropped_points_and_padding(grid):
    points = np.zeros((90, 4), dtype=np.float32)
    points[:, 0] = np.linspace(0.01, 0.19, 90)
    points[:, 1] = -0.3
    points[:, 2] = np.linspace(-2., 2., 90)
    points[:, 3] = np.arange(90)  # Source identity, not clipped to [0, 1].
    first = build_lidar_pillars([points], grid=grid)
    repeated = build_lidar_pillars([points], grid=grid)
    changed_seed = build_lidar_pillars([points], grid=grid, seed=14)

    np.testing.assert_array_equal(first.features, repeated.features)
    np.testing.assert_array_equal(first.point_indices, repeated.point_indices)
    assert not np.array_equal(first.point_indices, changed_seed.point_indices)
    selected = first.point_indices[0]
    assert len(np.unique(selected)) == 64
    assert np.all(np.diff(selected) > 0)
    np.testing.assert_array_equal(first.features[0, :, :4], points[selected])
    np.testing.assert_array_equal(first.retained_count, [64])
    np.testing.assert_array_equal(first.dropped_count, [26])
    assert first.point_count[0, 0, 0, 0] == 90
    np.testing.assert_allclose(first.features[0, :, 4:7].mean(axis=0), 0., atol=1e-6)
    np.testing.assert_allclose(
        first.features[0, :, 4:7], points[selected, :3] - points[selected, :3].mean(axis=0),
        atol=1e-6,
    )


def test_selection_does_not_consume_global_rng_or_depend_on_other_pillars(grid):
    points = np.tile(_cloud([[0.45, -0.1, 1., 0.5]]), (80, 1))
    points[:, 3] = np.arange(80)
    other = points.copy()
    other[:, 0] = 0.05
    before = np.random.get_state()
    baseline = build_lidar_pillars([points], grid=grid)
    together = build_lidar_pillars([np.concatenate([points, other])], grid=grid)
    after = np.random.get_state()

    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(baseline.point_indices[0], together.point_indices[1])
    np.testing.assert_array_equal(baseline.features[0], together.features[1])


def test_point_accounting_is_conserved_for_several_capped_pillars(grid):
    first = np.tile(_cloud([[0.05, -0.35, 1., 0.5]]), (70, 1))
    second = np.tile(_cloud([[0.45, 0.1, 0., 0.2]]), (100, 1))
    outside = _cloud([[50., 0., 0., 1.]])
    result = build_lidar_pillars([np.concatenate([first, second, outside])], grid=grid)
    np.testing.assert_array_equal(result.retained_count, [64, 64])
    np.testing.assert_array_equal(result.dropped_count, [6, 36])
    assert result.point_count.sum() == 170
    assert result.retained_count.sum() + result.dropped_count.sum() == result.point_count.sum()


def test_no_overflow_preserves_all_points_under_reordering(grid):
    points = _cloud([[0.02, -0.38, -1., 0.25], [0.08, -0.32, 1., 0.75]])
    first = build_lidar_pillars([points], grid=grid, max_points_per_pillar=2)
    reversed_points = build_lidar_pillars([points[::-1]], grid=grid, max_points_per_pillar=2)
    np.testing.assert_array_equal(first.features[0], reversed_points.features[0, ::-1])
    assert first.dropped_count.sum() == 0


def test_inputs_are_not_mutated_or_aliased_and_float64_is_converted(grid):
    points = np.array([[0.05, -0.35, 1., -2.]], dtype=np.float64)
    points.setflags(write=False)
    expected = points.copy()
    result = build_lidar_pillars((points,), grid=grid, max_points_per_pillar=1)
    np.testing.assert_array_equal(points, expected)
    np.testing.assert_array_equal(result.features[0, 0, 4:7], [0., 0., 0.])
    assert result.features.dtype == np.float32
    assert result.features[0, 0, 3] == -2.
    result.features[:] = 0.
    np.testing.assert_array_equal(points, expected)


@pytest.mark.parametrize("channel", range(4))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e300])
def test_nonfinite_or_unrepresentable_xyzi_is_rejected_before_roi_filtering(grid, channel, value):
    points = np.array([[100., 100., 100., 0.5]], dtype=np.float64)
    points[0, channel] = value
    with pytest.raises(ValueError, match="finite XYZI"):
        build_lidar_pillars([points], grid=grid)


@pytest.mark.parametrize("points", [np.zeros(4), np.zeros((2, 3)), np.zeros((1, 2, 4))])
def test_wrong_point_shape_is_rejected(grid, points):
    with pytest.raises(ValueError, match="shape"):
        build_lidar_pillars([points], grid=grid)


@pytest.mark.parametrize("dtype", [np.int32, np.float16, np.complex64, object])
def test_wrong_dtype_is_rejected(grid, dtype):
    with pytest.raises(ValueError, match="dtype"):
        build_lidar_pillars([np.zeros((1, 4), dtype=dtype)], grid=grid)


@pytest.mark.parametrize("clouds", [None, np.zeros((1, 4)), np.zeros((2, 3, 4)), "cloud.bin", [None]])
def test_implicit_dense_batches_and_nonarrays_are_rejected(grid, clouds):
    with pytest.raises(TypeError, match="NumPy arrays|NumPy array"):
        build_lidar_pillars(clouds, grid=grid)


@pytest.mark.parametrize("cap", [0, -1, 1.5, True])
def test_invalid_point_cap_is_rejected(grid, cap):
    with pytest.raises(ValueError, match="positive integer"):
        build_lidar_pillars([], grid=grid, max_points_per_pillar=cap)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_invalid_seed_is_rejected(grid, seed):
    with pytest.raises(ValueError, match="non-negative integer"):
        build_lidar_pillars([], grid=grid, seed=seed)


def test_numpy_preprocessing_module_does_not_import_torch():
    completed = subprocess.run([
        sys.executable, "-c",
        "import sys; from rgb_lidar_fusion.lidar_pillars import build_lidar_pillars; "
        "assert 'torch' not in sys.modules",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_velodyne_loader_output_can_be_grouped_directly(grid, tmp_path):
    points = _cloud([[0.45, -0.35, 0., 0.1], [0.85, 0.1, 1., 0.8]])
    path = tmp_path / "000000.bin"
    points.tofile(path)
    result = build_lidar_pillars([load_velodyne_bin(path)], grid=grid)
    np.testing.assert_array_equal(result.indices, [[0, 2, 0], [0, 4, 2]])
    np.testing.assert_array_equal(result.features[:, 0, :4], points)


def test_initial_roi_has_250_by_250_cells():
    pytest.importorskip("torch")
    from rgb_lidar_fusion.camera_bev import BEVGridConfig

    grid = BEVGridConfig((0., 50.), (-25., 25.), (-3., 3.), 0.2)
    result = build_lidar_pillars([_cloud([[0.1, -24.9, 0., 1.], [49.9, 24.9, 0., 1.]])], grid=grid)
    assert grid.shape == (250, 250)
    assert result.point_count.shape == (1, 1, 250, 250)
    np.testing.assert_array_equal(result.indices, [[0, 0, 0], [0, 249, 249]])


def test_shared_grid_placement_matches_camera_lift_at_float32_boundaries():
    torch = pytest.importorskip("torch")
    from rgb_lidar_fusion.camera_bev import BEVGridConfig, lift_camera_features_to_bev
    from rgb_lidar_fusion.camera_depth_support import CameraDepthSupport

    grid = BEVGridConfig((0., 2.), (-0.4, 0.2), (-3., 3.), 0.2)
    points = _cloud([[0., -0.4, 1., 1.], [0.6, -0.2, 1., 1.], [1.4, 0., 1., 1.]])
    points = np.concatenate([points, _cloud([[
        np.nextafter(np.float32(2.), np.float32(0.)),
        np.nextafter(np.float32(0.2), np.float32(0.)), 1., 1.,
    ]])])
    batch_size = len(points)
    pillars = build_lidar_pillars([points[i:i + 1] for i in range(batch_size)], grid=grid)
    projection = torch.eye(3, 4).repeat(batch_size, 1, 1)
    projection[:, 0, 3] = -torch.from_numpy(points[:, 0].copy())
    projection[:, 1, 3] = -torch.from_numpy(points[:, 1].copy())
    depth = torch.ones(batch_size, 1, 1, 1)
    camera = lift_camera_features_to_bev(
        torch.ones_like(depth), depth_support=CameraDepthSupport(
            "sparse", depth, torch.ones_like(depth, dtype=torch.bool), None,
        ),
        camera_projection=projection,
        t_vehicle_to_camera=torch.eye(4).repeat(batch_size, 1, 1), grid=grid,
    )
    np.testing.assert_array_equal(pillars.point_count, camera.pixel_count.numpy())
