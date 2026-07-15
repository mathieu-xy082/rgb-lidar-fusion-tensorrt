import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.baseline_model import (
    BaselineFusionModel,
    ENRICHED_LIDAR_CHANNELS,
    SPARSE_LIDAR_CHANNELS,
)


def test_baseline_fusion_model_defaults_to_enriched_sparse_plus_splatted_lidar_contract():
    model = BaselineFusionModel(output_dim=4)
    rgb = torch.zeros((2, 3, 32, 48), dtype=torch.float32)
    lidar_maps = torch.zeros((2, ENRICHED_LIDAR_CHANNELS, 32, 48), dtype=torch.float32)

    output = model(rgb, lidar_maps)

    assert model.lidar_channels == 8
    assert output.shape == (2, 4)
    assert output.dtype == torch.float32


def test_baseline_fusion_model_can_be_configured_for_sparse_only_lidar_contract():
    model = BaselineFusionModel(lidar_mode="sparse", output_dim=4)
    rgb = torch.zeros((2, 3, 32, 48), dtype=torch.float32)
    lidar_maps = torch.zeros((2, SPARSE_LIDAR_CHANNELS, 32, 48), dtype=torch.float32)

    output = model(rgb, lidar_maps)

    assert model.lidar_channels == 6
    assert output.shape == (2, 4)


def test_baseline_fusion_model_rejects_unaligned_rgb_and_lidar_maps_with_clear_shape_error():
    model = BaselineFusionModel(lidar_mode="sparse", output_dim=4)
    rgb = torch.zeros((2, 3, 32, 48), dtype=torch.float32)
    lidar_maps = torch.zeros((2, 6, 16, 48), dtype=torch.float32)

    with pytest.raises(ValueError, match="same batch and spatial dimensions"):
        model(rgb, lidar_maps)


def test_baseline_fusion_model_rejects_lidar_channel_count_that_differs_from_declared_shape():
    model = BaselineFusionModel(lidar_mode="sparse", output_dim=4)
    rgb = torch.zeros((2, 3, 32, 48), dtype=torch.float32)
    lidar_maps = torch.zeros((2, 5, 32, 48), dtype=torch.float32)

    with pytest.raises(ValueError, match="lidar_maps must have 6 channels"):
        model(rgb, lidar_maps)


def test_baseline_fusion_model_rejects_missing_batch_dimension_with_clear_error():
    model = BaselineFusionModel(output_dim=4)
    rgb = torch.zeros((3, 32, 48), dtype=torch.float32)
    lidar_maps = torch.zeros((ENRICHED_LIDAR_CHANNELS, 32, 48), dtype=torch.float32)

    with pytest.raises(ValueError, match="rgb and lidar_maps must both have shape"):
        model(rgb, lidar_maps)


def test_baseline_fusion_model_rejects_unknown_lidar_mode():
    with pytest.raises(ValueError, match="lidar_mode must be one of"):
        BaselineFusionModel(lidar_mode="dense", output_dim=4)
