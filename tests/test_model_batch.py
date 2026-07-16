import numpy as np
import pytest

from rgb_lidar_fusion.model_batch import (
    dataset_item_to_model_batch,
    dataset_items_to_model_batch,
)
from rgb_lidar_fusion.lidar_splatting import SplattingConfig
from rgb_lidar_fusion.project_lidar import LIDAR_MAP_CHANNELS


def synthetic_dataset_item() -> dict:
    image = np.zeros((3, 2, 3), dtype=np.float32)
    image[0, 0, 0] = 0.25
    image[1, 0, 1] = 0.5
    image[2, 0, 2] = 0.75
    lidar_maps = np.zeros((6, 2, 3), dtype=np.float32)
    lidar_maps[0, 1, 1] = 0.125
    lidar_maps[1, 1, 1] = 0.01
    lidar_maps[2, 1, 1] = 0.02
    lidar_maps[3, 1, 1] = 0.03
    lidar_maps[4, 1, 1] = 0.9
    lidar_maps[5, 1, 1] = 1.0
    return {
        "image": image,
        "lidar_maps": lidar_maps,
        "target": {"class_name": "synthetic"},
        "meta": {"sample_id": "synthetic-1"},
    }


def test_dataset_item_batch_adds_batch_dimension_and_documents_channel_order():
    item = synthetic_dataset_item()

    batch = dataset_item_to_model_batch(item)

    assert batch["inputs"].shape == (1, 9, 2, 3)
    assert batch["inputs"].dtype == np.float32
    assert batch["input_channels"] == [
        "rgb_red",
        "rgb_green",
        "rgb_blue",
        *LIDAR_MAP_CHANNELS,
    ]
    np.testing.assert_allclose(batch["inputs"][0, 0:3], item["image"])
    np.testing.assert_allclose(batch["inputs"][0, 3:9], item["lidar_maps"])


def test_dataset_item_batch_rejects_mismatched_image_and_lidar_shapes():
    item = synthetic_dataset_item()
    item["lidar_maps"] = np.zeros((6, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="image and lidar_maps must share height and width"):
        dataset_item_to_model_batch(item)


def test_dataset_item_batch_rejects_wrong_channel_counts():
    item = synthetic_dataset_item()
    item["image"] = np.zeros((2, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"image must have shape \[3, H, W\]"):
        dataset_item_to_model_batch(item)


def test_dataset_item_batch_rejects_dataset_channel_metadata_drift():
    item = synthetic_dataset_item()
    item["meta"] = {"lidar_map_channels": ["unexpected"]}

    with pytest.raises(ValueError, match="lidar_map_channels metadata must match"):
        dataset_item_to_model_batch(item)


def test_dataset_item_batch_can_append_splatted_depth_and_confidence_channels():
    item = synthetic_dataset_item()

    batch = dataset_item_to_model_batch(
        item,
        include_splatted_depth=True,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
    )

    assert batch["inputs"].shape == (1, 11, 2, 3)
    assert batch["input_channels"][-2:] == ["depth_expanded", "confidence"]
    assert batch["inputs"][0, 9, 1, 1] == pytest.approx(0.125)
    assert batch["inputs"][0, 10, 1, 1] == pytest.approx(1.0)
    assert batch["inputs"][0, 9, 1, 0] == pytest.approx(0.125)
    assert batch["inputs"][0, 10, 1, 0] == pytest.approx(np.exp(-0.5))


def test_dataset_item_batch_preserves_sparse_lidar_maps_separately_from_splats():
    item = synthetic_dataset_item()

    batch = dataset_item_to_model_batch(
        item,
        include_splatted_depth=True,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
    )

    np.testing.assert_allclose(batch["sparse_lidar_maps"], item["lidar_maps"][np.newaxis])
    assert batch["sparse_lidar_maps"][0, 0, 1, 0] == pytest.approx(0.0)
    assert batch["inputs"][0, 9, 1, 0] == pytest.approx(0.125)


def test_dataset_items_batch_stacks_synthetic_items_without_losing_per_item_contract():
    first = synthetic_dataset_item()
    second = synthetic_dataset_item()
    second["image"] = second["image"] + 0.1
    second["lidar_maps"] = second["lidar_maps"].copy()
    second["lidar_maps"][0, 0, 2] = 0.5
    second["lidar_maps"][5, 0, 2] = 1.0

    batch = dataset_items_to_model_batch([first, second])

    assert batch["inputs"].shape == (2, 9, 2, 3)
    assert batch["sparse_lidar_maps"].shape == (2, 6, 2, 3)
    assert batch["input_channels"] == [
        "rgb_red",
        "rgb_green",
        "rgb_blue",
        *LIDAR_MAP_CHANNELS,
    ]
    assert batch["targets"] == [first["target"], second["target"]]
    assert batch["metas"] == [first["meta"], second["meta"]]
    np.testing.assert_allclose(batch["inputs"][0, 0:3], first["image"])
    np.testing.assert_allclose(batch["inputs"][1, 0:3], second["image"])
    np.testing.assert_allclose(batch["sparse_lidar_maps"][0], first["lidar_maps"])
    np.testing.assert_allclose(batch["sparse_lidar_maps"][1], second["lidar_maps"])
