import numpy as np
import pytest

from rgb_lidar_fusion.dataset import KittiSparseLidarDataset


FIXTURE_ROOT = "tests/fixtures/synthetic_kitti"


def test_dataset_returns_rgb_lidar_maps_target_and_meta():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT)

    sample = dataset[0]

    assert len(dataset) == 1
    assert set(sample) == {"image", "lidar_maps", "target", "meta"}
    assert sample["image"].shape == (3, 4, 5)
    assert sample["image"].dtype == np.float32
    np.testing.assert_allclose(sample["image"][:, 0, 0], [1.0, 0.0, 0.0])
    assert sample["lidar_maps"].shape == (6, 4, 5)
    assert sample["lidar_maps"].dtype == np.float32
    assert sample["lidar_maps"][5].sum() == pytest.approx(2.0)
    assert sample["lidar_maps"][0, 2, 2] == pytest.approx(10.0 / 80.0)
    assert sample["lidar_maps"][4, 2, 2] == pytest.approx(0.7)
    assert sample["target"] == {"class_name": "synthetic_car", "bbox_2d": [1, 1, 3, 3]}
    assert sample["meta"] == {"sample_id": "000001", "image_shape": [4, 5]}


def test_dataset_rejects_missing_sample_files():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT, manifest_name="missing-file-manifest.json")

    with pytest.raises(FileNotFoundError, match="missing-image.npy"):
        _ = dataset[0]


def test_dataset_reports_missing_required_manifest_fields():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT, manifest_name="missing-field-manifest.json")

    with pytest.raises(ValueError, match="sample 000002.*lidar"):
        _ = dataset[0]
