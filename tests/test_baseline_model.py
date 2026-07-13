import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.baseline_model import BaselineFusionModel


def test_baseline_fusion_model_forward_returns_batch_predictions_for_synthetic_rgb_and_lidar_maps():
    model = BaselineFusionModel(lidar_channels=6, output_dim=4)
    rgb = torch.zeros((2, 3, 32, 48), dtype=torch.float32)
    lidar_maps = torch.zeros((2, 6, 32, 48), dtype=torch.float32)

    output = model(rgb, lidar_maps)

    assert output.shape == (2, 4)
    assert output.dtype == torch.float32


def test_baseline_fusion_model_rejects_unaligned_rgb_and_lidar_maps_with_clear_shape_error():
    model = BaselineFusionModel(lidar_channels=6, output_dim=4)
    rgb = torch.zeros((2, 3, 32, 48), dtype=torch.float32)
    lidar_maps = torch.zeros((2, 6, 16, 48), dtype=torch.float32)

    with pytest.raises(ValueError, match="same batch and spatial dimensions"):
        model(rgb, lidar_maps)
