import numpy as np
import pytest

from rgb_lidar_fusion.model_batch import (
    build_depth_holdout_mask,
    dataset_items_to_model_batch,
    model_batch_to_camera_depth_training_batch,
)


def test_random_holdout_preserves_original_multisample_selection():
    valid = np.zeros((2, 1, 3, 4), dtype=bool)
    valid[0, 0].flat[[0, 2, 4, 6, 8, 10]] = True
    valid[1, 0].flat[[1, 3, 5, 7, 9, 11]] = True

    mask = build_depth_holdout_mask(valid, holdout_fraction=0.4, seed=19)

    # Recorded from the original inline implementation before extraction.
    assert [np.flatnonzero(sample).tolist() for sample in mask] == [[4, 10], [5, 9]]
    assert mask.shape == valid.shape
    assert mask.dtype == np.bool_
    assert not np.any(mask & ~valid)


def test_random_holdout_is_reproducible_without_mutating_valid_pixels():
    valid = np.ones((2, 1, 10, 10), dtype=bool)
    before = valid.copy()

    first = build_depth_holdout_mask(valid, seed=13)
    repeated = build_depth_holdout_mask(valid, seed=13)
    different = build_depth_holdout_mask(valid, seed=14)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(valid, before)
    assert not np.array_equal(first, different)
    assert first.sum(axis=(1, 2, 3)).tolist() == [20, 20]


@pytest.mark.parametrize("fraction", [0.01, 0.99])
def test_two_point_sample_keeps_one_point_and_holds_out_one(fraction):
    valid = np.array([[[[True, False, True]]]])

    mask = build_depth_holdout_mask(valid, holdout_fraction=fraction)

    assert mask.sum() == 1
    assert (valid & ~mask).sum() == 1


def test_single_point_sample_preserves_existing_target_only_behavior():
    valid = np.array([[[[False, True, False]]]])

    np.testing.assert_array_equal(build_depth_holdout_mask(valid), valid)


def test_empty_sample_fails_with_its_batch_index():
    valid = np.ones((2, 1, 2, 3), dtype=bool)
    valid[1] = False

    with pytest.raises(ValueError, match="batch sample 1 has no valid sparse depth"):
        build_depth_holdout_mask(valid)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, np.nan, np.inf])
def test_holdout_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="holdout_fraction must be between 0 and 1"):
        build_depth_holdout_mask(np.ones((1, 1, 2, 3)), holdout_fraction=fraction)


@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3), (1, 2, 3, 4), (0, 1, 2, 3)])
def test_holdout_rejects_invalid_mask_shape(shape):
    with pytest.raises(ValueError, match="valid_mask must have non-empty shape"):
        build_depth_holdout_mask(np.ones(shape))


def test_training_batch_uses_extracted_mask_and_preserves_rgb_and_targets():
    sparse = np.zeros((6, 3, 4), dtype=np.float32)
    for channel in range(6):
        sparse[channel].flat[[0, 2, 4, 6, 8, 10]] = 0.1 * (channel + 1)
    rgb = np.full((3, 3, 4), 0.25, dtype=np.float32)
    batch = dataset_items_to_model_batch(
        [{"image": rgb, "lidar_maps": sparse}], include_splatted_depth=False
    )
    original_sparse = batch["sparse_lidar_maps"].copy()
    expected_mask = build_depth_holdout_mask(
        original_sparse[:, 5:6] > 0, holdout_fraction=0.4, seed=19
    )

    prepared = model_batch_to_camera_depth_training_batch(
        batch, holdout_fraction=0.4, seed=19, include_splatted_depth=False
    )

    np.testing.assert_array_equal(prepared["loss_mask"], expected_mask)
    np.testing.assert_array_equal(prepared["rgb"][0], rgb)
    np.testing.assert_array_equal(prepared["depth_target"], original_sparse[:, 0:1])
    np.testing.assert_array_equal(batch["sparse_lidar_maps"], original_sparse)
    held_out = expected_mask[0, 0]
    assert np.all(prepared["lidar_maps"][0, :, held_out] == 0)
    np.testing.assert_array_equal(
        prepared["lidar_maps"][0, :, ~held_out], original_sparse[0, :, ~held_out]
    )
