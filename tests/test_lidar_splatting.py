import numpy as np
import pytest

from rgb_lidar_fusion.lidar_splatting import SplattingConfig, splat_sparse_depth


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
