import numpy as np
import pytest

from rgb_lidar_fusion.lidar_splatting import (
    SplattingConfig,
    splat_projected_depth,
    splat_sparse_depth,
)


def test_single_sparse_point_expands_depth_and_confidence_locally():
    sparse_depth = np.zeros((5, 5), dtype=np.float32)
    sparse_mask = np.zeros((5, 5), dtype=bool)
    sparse_depth[2, 2] = 10.0
    sparse_mask[2, 2] = True

    splat = splat_sparse_depth(
        sparse_depth,
        sparse_mask,
        config=SplattingConfig(radius_px=1, sigma_px=1.0),
    )

    assert splat.depth_expanded.shape == (5, 5)
    assert splat.confidence.shape == (5, 5)
    assert splat.depth_expanded[2, 2] == pytest.approx(10.0)
    assert splat.confidence[2, 2] == pytest.approx(1.0)
    assert splat.depth_expanded[2, 3] == pytest.approx(10.0)
    assert splat.confidence[2, 3] == pytest.approx(np.exp(-0.5))
    assert splat.depth_expanded[0, 0] == pytest.approx(0.0)
    assert splat.confidence[0, 0] == pytest.approx(0.0)


def test_overlapping_splats_deterministically_keep_nearest_depth():
    sparse_depth = np.zeros((3, 5), dtype=np.float32)
    sparse_mask = np.zeros((3, 5), dtype=bool)
    sparse_depth[1, 1] = 20.0
    sparse_mask[1, 1] = True
    sparse_depth[1, 3] = 10.0
    sparse_mask[1, 3] = True

    splat = splat_sparse_depth(
        sparse_depth,
        sparse_mask,
        config=SplattingConfig(radius_px=2, sigma_px=1.0),
    )

    assert splat.depth_expanded[1, 2] == pytest.approx(10.0)
    assert splat.confidence[1, 2] == pytest.approx(np.exp(-0.5))


def test_splatting_clips_kernel_at_image_edges():
    sparse_depth = np.zeros((3, 3), dtype=np.float32)
    sparse_mask = np.zeros((3, 3), dtype=bool)
    sparse_depth[0, 0] = 7.0
    sparse_mask[0, 0] = True

    splat = splat_sparse_depth(
        sparse_depth,
        sparse_mask,
        config=SplattingConfig(radius_px=1, sigma_px=1.0),
    )

    expected_occupied = np.array(
        [
            [True, True, False],
            [True, True, False],
            [False, False, False],
        ]
    )
    np.testing.assert_array_equal(splat.confidence > 0.0, expected_occupied)
    assert splat.depth_expanded[1, 1] == pytest.approx(7.0)


def test_confidence_decreases_with_distance_from_source_point():
    sparse_depth = np.zeros((7, 7), dtype=np.float32)
    sparse_mask = np.zeros((7, 7), dtype=bool)
    sparse_depth[3, 3] = 12.0
    sparse_mask[3, 3] = True

    splat = splat_sparse_depth(
        sparse_depth,
        sparse_mask,
        config=SplattingConfig(radius_px=3, sigma_px=2.0),
    )

    assert splat.confidence[3, 3] > splat.confidence[3, 4]
    assert splat.confidence[3, 4] > splat.confidence[3, 6]


def test_splatting_preserves_sparse_inputs_as_separate_outputs():
    sparse_depth = np.zeros((4, 4), dtype=np.float32)
    sparse_mask = np.zeros((4, 4), dtype=bool)
    sparse_depth[2, 2] = 9.0
    sparse_mask[2, 2] = True

    splat = splat_sparse_depth(
        sparse_depth,
        sparse_mask,
        config=SplattingConfig(radius_px=1, sigma_px=1.0),
    )

    np.testing.assert_array_equal(splat.sparse_depth, sparse_depth)
    np.testing.assert_array_equal(splat.sparse_mask, sparse_mask)
    assert splat.depth_expanded[2, 3] == pytest.approx(9.0)
    assert sparse_depth[2, 3] == pytest.approx(0.0)
    assert not sparse_mask[2, 3]


def test_projected_points_input_builds_sparse_map_before_splatting():
    pixels = np.array(
        [
            [1.2, 2.0],
            [1.4, 2.2],
            [4.0, 4.0],
        ],
        dtype=np.float32,
    )
    depths = np.array([8.0, 5.0, 9.0], dtype=np.float32)

    splat = splat_projected_depth(
        pixels=pixels,
        depths=depths,
        image_shape=(5, 5),
        config=SplattingConfig(radius_px=0, sigma_px=1.0),
    )

    assert splat.sparse_depth[2, 1] == pytest.approx(5.0)
    assert splat.sparse_mask[2, 1]
    assert splat.depth_expanded[2, 1] == pytest.approx(5.0)
    assert splat.confidence[2, 1] == pytest.approx(1.0)
    assert splat.sparse_depth[4, 4] == pytest.approx(9.0)


def test_square_splat_kernel_includes_diagonal_neighbors_with_gaussian_confidence():
    sparse_depth = np.zeros((5, 5), dtype=np.float32)
    sparse_mask = np.zeros((5, 5), dtype=bool)
    sparse_depth[2, 2] = 6.0
    sparse_mask[2, 2] = True

    splat = splat_sparse_depth(
        sparse_depth,
        sparse_mask,
        config=SplattingConfig(radius_px=1, sigma_px=2.0),
    )

    assert splat.depth_expanded[1, 1] == pytest.approx(6.0)
    assert splat.confidence[1, 1] == pytest.approx(np.exp(-2.0 / 8.0))
    assert splat.depth_expanded[1, 3] == pytest.approx(6.0)
    assert splat.depth_expanded[3, 1] == pytest.approx(6.0)
    assert splat.depth_expanded[3, 3] == pytest.approx(6.0)
    assert splat.confidence[0, 2] == pytest.approx(0.0)


def test_equal_depth_and_confidence_overlap_keeps_earlier_row_major_source():
    sparse_depth = np.zeros((3, 5), dtype=np.float32)
    sparse_mask = np.zeros((3, 5), dtype=bool)
    sparse_depth[1, 1] = 10.0
    sparse_mask[1, 1] = True
    sparse_depth[1, 3] = 10.0
    sparse_mask[1, 3] = True

    splat = splat_sparse_depth(
        sparse_depth,
        sparse_mask,
        config=SplattingConfig(radius_px=2, sigma_px=1.0),
    )

    assert splat.depth_expanded[1, 2] == pytest.approx(10.0)
    assert splat.confidence[1, 2] == pytest.approx(np.exp(-0.5))
    assert splat.sparse_mask[1, 1]
    assert splat.sparse_mask[1, 3]


def test_projected_depth_discards_non_positive_depths_before_splatting():
    pixels = np.array([[2.0, 2.0], [3.0, 2.0], [1.0, 1.0]], dtype=np.float32)
    depths = np.array([0.0, -4.0, 7.0], dtype=np.float32)

    splat = splat_projected_depth(
        pixels=pixels,
        depths=depths,
        image_shape=(5, 5),
        config=SplattingConfig(radius_px=0, sigma_px=1.0),
    )

    assert not splat.sparse_mask[2, 2]
    assert not splat.sparse_mask[2, 3]
    assert splat.sparse_mask[1, 1]
    assert splat.sparse_depth[1, 1] == pytest.approx(7.0)
    assert int(splat.sparse_mask.sum()) == 1


def test_projected_depth_matches_sparse_depth_splatting_for_same_rasterization():
    pixels = np.array([[2.0, 2.0], [4.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    depths = np.array([8.0, 12.0, 5.0], dtype=np.float32)
    image_shape = (6, 6)
    config = SplattingConfig(radius_px=1, sigma_px=1.5)

    from_projected = splat_projected_depth(
        pixels=pixels,
        depths=depths,
        image_shape=image_shape,
        config=config,
    )
    sparse_depth = np.zeros(image_shape, dtype=np.float32)
    sparse_mask = np.zeros(image_shape, dtype=bool)
    sparse_depth[2, 2] = 5.0
    sparse_mask[2, 2] = True
    sparse_depth[1, 4] = 12.0
    sparse_mask[1, 4] = True
    from_sparse = splat_sparse_depth(sparse_depth, sparse_mask, config=config)

    np.testing.assert_allclose(from_projected.sparse_depth, from_sparse.sparse_depth)
    np.testing.assert_array_equal(from_projected.sparse_mask, from_sparse.sparse_mask)
    np.testing.assert_allclose(from_projected.depth_expanded, from_sparse.depth_expanded)
    np.testing.assert_allclose(from_projected.confidence, from_sparse.confidence)
