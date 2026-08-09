from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_depth_model import CameraDepthModel, masked_depth_loss
from rgb_lidar_fusion.lidar_splatting import SplattingConfig
from rgb_lidar_fusion.model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_camera_depth_training_batch,
)
from rgb_lidar_fusion.training import train_camera_depth_step


def _item(offset: int = 0) -> dict:
    image = np.zeros((3, 17, 23), dtype=np.float32)
    image[0, 2:15, 3:20] = 0.25
    image[1, 4:12, 5:18] = 0.5
    sparse = np.zeros((6, 17, 23), dtype=np.float32)
    for index, (y, x) in enumerate(((2, 2), (4, 8), (7, 14), (10, 5), (14, 20))):
        depth = 0.1 * (index + 1) + 0.01 * offset
        sparse[0, y, x] = depth
        sparse[1, y, x] = 0.01 * (index + 1)
        sparse[2, y, x] = -0.01 * index
        sparse[3, y, x] = 0.02 * index
        sparse[4, y, x] = 0.8
        sparse[5, y, x] = 1.0
    return {
        "image": image,
        "lidar_maps": sparse,
        "target": {},
        "meta": {"sample_id": f"depth-{offset}"},
    }


def test_camera_depth_model_preserves_odd_input_resolution() -> None:
    model = CameraDepthModel()
    rgb = torch.zeros((2, 3, 31, 47), dtype=torch.float32)
    lidar_maps = torch.zeros((2, 8, 31, 47), dtype=torch.float32)

    prediction = model(rgb, lidar_maps)

    assert prediction.shape == (2, 1, 31, 47)
    assert torch.isfinite(prediction).all()


def test_masked_depth_loss_uses_only_selected_pixels_and_rejects_empty_mask() -> None:
    prediction = torch.tensor([[[[0.0, 100.0], [0.0, 0.0]]]])
    target = torch.tensor([[[[1.0, -100.0], [0.0, 0.0]]]])
    mask = torch.tensor([[[[True, False], [False, False]]]])

    loss = masked_depth_loss(prediction, target, mask)

    assert loss.item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="at least one depth target"):
        masked_depth_loss(prediction, target, torch.zeros_like(mask))


def test_camera_depth_holdout_is_deterministic_and_removes_target_contributions() -> None:
    batch = dataset_items_to_model_batch([_item()], include_splatted_depth=False)
    config = SplattingConfig(radius_px=2, sigma_px=1.0)

    first = model_batch_to_camera_depth_training_batch(
        batch,
        holdout_fraction=0.4,
        seed=19,
        splatting_config=config,
    )
    second = model_batch_to_camera_depth_training_batch(
        batch,
        holdout_fraction=0.4,
        seed=19,
        splatting_config=config,
    )

    np.testing.assert_array_equal(first["loss_mask"], second["loss_mask"])
    np.testing.assert_array_equal(first["lidar_maps"], second["lidar_maps"])
    held_out = first["loss_mask"][0, 0]
    assert int(held_out.sum()) == 2
    assert np.all(first["kept_sparse_lidar_maps"][0, :, held_out] == 0.0)
    assert np.all(first["depth_target"][0, 0, held_out] > 0.0)

    kept = first["kept_sparse_lidar_maps"][0]
    expected_batch = dataset_items_to_model_batch(
        [{"image": _item()["image"], "lidar_maps": kept, "target": {}, "meta": {}}],
        include_splatted_depth=True,
        splatting_config=config,
    )
    np.testing.assert_allclose(first["lidar_maps"][0], expected_batch["inputs"][0, 3:])


def test_camera_depth_training_step_has_finite_non_zero_gradients() -> None:
    torch.manual_seed(7)
    batch = dataset_items_to_model_batch(
        [_item(0), _item(1)],
        include_splatted_depth=False,
    )
    depth_batch = model_batch_to_camera_depth_training_batch(
        batch,
        holdout_fraction=0.4,
        seed=23,
        splatting_config=SplattingConfig(radius_px=1, sigma_px=1.0),
    )
    model = CameraDepthModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = train_camera_depth_step(
        model=model,
        optimizer=optimizer,
        rgb=torch.from_numpy(depth_batch["rgb"]),
        lidar_maps=torch.from_numpy(depth_batch["lidar_maps"]),
        depth_target=torch.from_numpy(depth_batch["depth_target"]),
        loss_mask=torch.from_numpy(depth_batch["loss_mask"]),
        device=torch.device("cpu"),
    )

    assert np.isfinite(metrics.loss)
    assert np.isfinite(metrics.grad_norm)
    assert metrics.grad_norm > 0.0
