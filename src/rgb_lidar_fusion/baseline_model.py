"""Minimal PyTorch baseline for RGB + sparse/enriched LiDAR map fusion.

Shape contract:
    rgb: Tensor[B, 3, H, W]
    lidar_maps: Tensor[B, L, H, W]
        sparse mode: L = 6 channels from ``build_sparse_lidar_maps``
            (normalized depth, xyz, intensity, mask).
        enriched mode: L = 8 channels preserving the 6 sparse maps and
            appending ``depth_expanded`` and ``confidence`` splatting maps.
    output: Tensor[B, output_dim]
"""

from __future__ import annotations

import torch
from torch import nn

SPARSE_LIDAR_CHANNELS = 6
ENRICHED_LIDAR_CHANNELS = 8
LIDAR_CHANNELS_BY_MODE = {
    "sparse": SPARSE_LIDAR_CHANNELS,
    "enriched": ENRICHED_LIDAR_CHANNELS,
}


class BaselineFusionModel(nn.Module):
    """Small early-fusion CNN consuming RGB and sparse/enriched LiDAR maps."""

    def __init__(
        self,
        lidar_channels: int | None = None,
        output_dim: int = 1,
        lidar_mode: str = "enriched",
    ) -> None:
        super().__init__()
        if lidar_mode not in LIDAR_CHANNELS_BY_MODE:
            valid_modes = ", ".join(sorted(LIDAR_CHANNELS_BY_MODE))
            raise ValueError(f"lidar_mode must be one of: {valid_modes}.")
        expected_lidar_channels = LIDAR_CHANNELS_BY_MODE[lidar_mode]
        if lidar_channels is not None and lidar_channels != expected_lidar_channels:
            raise ValueError(
                "lidar_channels must match "
                f"{lidar_mode} mode ({expected_lidar_channels} channels)."
            )
        self.lidar_mode = lidar_mode
        self.lidar_channels = expected_lidar_channels
        input_channels = 3 + self.lidar_channels
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
            lidar_maps: LiDAR maps with shape ``[B, L, H, W]``. In the default
                enriched contract, channels are the six sparse maps followed by
                ``depth_expanded`` and ``confidence``.

        Returns:
            Prediction tensor with shape ``[B, output_dim]``.

        Raises:
            ValueError: If RGB and LiDAR maps are not aligned as ``[B, C, H, W]``.
        """

        if rgb.ndim != 4 or lidar_maps.ndim != 4:
            raise ValueError(
                "rgb and lidar_maps must both have shape [B, C, H, W]."
            )
        if rgb.shape[1] != 3:
            raise ValueError("rgb must have 3 channels.")
        if rgb.shape[0] != lidar_maps.shape[0] or rgb.shape[2:] != lidar_maps.shape[2:]:
            raise ValueError(
                "rgb and lidar_maps must share the same batch and spatial dimensions "
                "as [B, C, H, W]."
            )
        if lidar_maps.shape[1] != self.lidar_channels:
            raise ValueError(f"lidar_maps must have {self.lidar_channels} channels.")

        fused = torch.cat([rgb, lidar_maps], dim=1)
        features = self.features(fused).flatten(1)
        return self.head(features)
