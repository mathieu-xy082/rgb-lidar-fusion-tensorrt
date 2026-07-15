"""Minimal PyTorch baseline for RGB + sparse LiDAR map fusion.

Shape contract:
    rgb: Tensor[B, 3, H, W]
    lidar_maps: Tensor[B, L, H, W], where L defaults to 6 channels from
        ``build_sparse_lidar_maps`` (normalized depth, xyz, intensity, mask).
    output: Tensor[B, output_dim]
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineFusionModel(nn.Module):
    """Small early-fusion CNN consuming RGB and sparse LiDAR maps."""

    def __init__(self, lidar_channels: int = 6, output_dim: int = 1) -> None:
        super().__init__()
        self.lidar_channels = lidar_channels
        input_channels = 3 + lidar_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, output_dim)

    def forward(self, rgb: torch.Tensor, lidar_maps: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            rgb: RGB image batch with shape ``[B, 3, H, W]``.
            lidar_maps: Sparse LiDAR maps with shape ``[B, L, H, W]``.

        Returns:
            Prediction tensor with shape ``[B, output_dim]``.

        Raises:
            ValueError: If RGB and LiDAR maps are not aligned as ``[B, C, H, W]``.
        """

        if rgb.shape[0] != lidar_maps.shape[0] or rgb.shape[2:] != lidar_maps.shape[2:]:
            raise ValueError(
                "rgb and lidar_maps must share the same batch and spatial dimensions "
                "as [B, C, H, W]."
            )
        if rgb.shape[1] != 3:
            raise ValueError("rgb must have 3 channels.")
        if lidar_maps.shape[1] != self.lidar_channels:
            raise ValueError(f"lidar_maps must have {self.lidar_channels} channels.")

        fused = torch.cat([rgb, lidar_maps], dim=1)
        features = self.features(fused).flatten(1)
        return self.head(features)
