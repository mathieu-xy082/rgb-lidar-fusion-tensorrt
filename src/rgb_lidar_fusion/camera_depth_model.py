"""Dense camera-depth model for RGB and projected LiDAR fusion."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .baseline_model import LIDAR_CHANNEL_NAMES_BY_MODE, LIDAR_CHANNELS_BY_MODE


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class CameraDepthModel(nn.Module):
    """Small encoder/decoder predicting normalized camera depth per pixel."""

    def __init__(self, *, lidar_mode: str = "enriched") -> None:
        super().__init__()
        if lidar_mode not in LIDAR_CHANNELS_BY_MODE:
            valid_modes = ", ".join(sorted(LIDAR_CHANNELS_BY_MODE))
            raise ValueError(f"lidar_mode must be one of: {valid_modes}.")

        self.lidar_mode = lidar_mode
        self.lidar_channels = LIDAR_CHANNELS_BY_MODE[lidar_mode]
        self.lidar_channel_names = LIDAR_CHANNEL_NAMES_BY_MODE[lidar_mode]

        self.stem = _ConvBlock(3 + self.lidar_channels, 32)
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            _ConvBlock(64, 64),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            _ConvBlock(96, 96),
        )
        self.decoder_1 = _ConvBlock(96 + 64, 64)
        self.decoder_0 = _ConvBlock(64 + 32, 32)
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, rgb: torch.Tensor, lidar_maps: torch.Tensor) -> torch.Tensor:
        """Return normalized depth with shape ``[B, 1, H, W]``."""

        self._validate_inputs(rgb, lidar_maps)
        stem = self.stem(torch.cat([rgb, lidar_maps], dim=1))
        encoded_1 = self.encoder_1(stem)
        encoded_2 = self.encoder_2(encoded_1)

        decoded_1 = F.interpolate(
            encoded_2,
            size=encoded_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_1 = self.decoder_1(torch.cat([decoded_1, encoded_1], dim=1))
        decoded_0 = F.interpolate(
            decoded_1,
            size=stem.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_0 = self.decoder_0(torch.cat([decoded_0, stem], dim=1))
        return self.depth_head(decoded_0)

    def _validate_inputs(
        self,
        rgb: torch.Tensor,
        lidar_maps: torch.Tensor,
    ) -> None:
        if rgb.ndim != 4 or lidar_maps.ndim != 4:
            raise ValueError(
                "rgb and lidar_maps must both have shape [B, C, H, W]."
            )
        if rgb.shape[1] != 3:
            raise ValueError("rgb must have 3 channels.")
        if rgb.shape[0] != lidar_maps.shape[0] or rgb.shape[2:] != lidar_maps.shape[2:]:
            raise ValueError(
                "rgb and lidar_maps must share the same batch and spatial dimensions."
            )
        if lidar_maps.shape[1] != self.lidar_channels:
            raise ValueError(f"lidar_maps must have {self.lidar_channels} channels.")


def masked_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute Smooth L1 only at held-out LiDAR measurement pixels."""

    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("prediction must have shape [B, 1, H, W].")
    if target.shape != prediction.shape:
        raise ValueError("target must have the same shape as prediction.")
    if mask.shape != prediction.shape:
        raise ValueError("mask must have the same shape as prediction.")
    selected = mask.to(dtype=torch.bool)
    if not torch.any(selected):
        raise ValueError("mask must select at least one depth target pixel.")
    if not torch.isfinite(target[selected]).all():
        raise ValueError("selected depth targets must be finite.")
    return F.smooth_l1_loss(
        prediction[selected],
        target[selected],
        beta=beta,
        reduction="mean",
    )
