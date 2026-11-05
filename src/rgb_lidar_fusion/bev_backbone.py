"""Three-scale residual context on the shared metric BEV feature grid."""

from __future__ import annotations

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .bev_fusion import FusedBEVFeatures
from .camera_bev import BEVGridConfig


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features + self.convs(features))


class SharedBEVBackbone(nn.Module):
    """Add three-scale spatial context while preserving stride-1 BEV evidence.

    Widths default to (32, 64, 96). Residual blocks at fine, middle and deep
    scales each use two 3x3 convolutions. Two 3x3 stride-2/padding-1 downsamplers
    preserve the index-zero center, with sizes ceil(H/2), then ceil(H/4).
    Matching 3x3 transposed convolutions add their outputs to encoder skips;
    a final fine-scale residual block returns the initial width/resolution.

    Explicit output_size restores odd/even/non-square grids without fractional
    interpolation, cropping or a half-cell offset. Internal index i maps to
    fine-grid index 2*i or 4*i; these are not new BEVGridConfig cell origins.
    Extra output padding, if needed, is on the high-index side only.

    Relative to its input feature grid, the interior theoretical field spans
    39 cells at output indices i % 4 == 2, otherwise 43, per axis. Ranges by phase are
    [-21,21], [-22,20], [-19,19], [-20,22]; stride phases are not translation
    invariant. At 20 cm this is 7.8/8.6 m per axis, not an effective learned
    field or object-recognition claim. ROI padding introduces no measurements.

    All convolutions are bias-free, without normalization, preserving a zero
    response to zero features and a local dependency field in train/eval modes.
    Evidence is cloned, never downsampled, reweighted or reapplied as a mask:
    contextual features may exist outside both direct support maps.
    """

    def __init__(self, *, channels: tuple[int, int, int] = (32, 64, 96)) -> None:
        super().__init__()
        if (
            not isinstance(channels, (tuple, list)) or len(channels) != 3
            or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
                   for value in channels)
        ):
            raise ValueError("channels must contain three positive integer widths.")
        self.channels = tuple(int(value) for value in channels)
        fine, middle, deep = self.channels
        self.fine = _ResidualBlock(fine)
        self.down1 = nn.Conv2d(fine, middle, 3, stride=2, padding=1, bias=False)
        self.middle = _ResidualBlock(middle)
        self.down2 = nn.Conv2d(middle, deep, 3, stride=2, padding=1, bias=False)
        self.deep = _ResidualBlock(deep)
        self.up1 = nn.ConvTranspose2d(deep, middle, 3, stride=2, padding=1, bias=False)
        self.up0 = nn.ConvTranspose2d(middle, fine, 3, stride=2, padding=1, bias=False)
        self.output = _ResidualBlock(fine)

    def forward(self, bev: FusedBEVFeatures) -> FusedBEVFeatures:
        """Return aligned features plus independent, unchanged evidence snapshots."""

        features = bev.features
        if not isinstance(bev.grid, BEVGridConfig):
            raise ValueError("grid must be an explicit BEVGridConfig.")
        if (
            features.ndim != 4 or features.shape[0] < 1
            or features.shape[1] != self.channels[0] or features.shape[2:] != bev.grid.shape
        ):
            raise ValueError("features must have non-empty shape [B, channels[0], *grid.shape].")
        if features.dtype not in (torch.float32, torch.float64):
            raise ValueError("features must have float32 or float64 dtype.")
        parameter = next(self.parameters())
        if features.dtype != parameter.dtype or features.device != parameter.device:
            raise ValueError("features must share the module's device and dtype.")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite.")
        evidence_shape = (features.shape[0], 1, *bev.grid.shape)
        for name, counts in (
            ("camera_pixel_count", bev.camera_pixel_count),
            ("lidar_point_count", bev.lidar_point_count),
        ):
            if counts.shape != evidence_shape or counts.dtype != torch.int64:
                raise ValueError(f"{name} must have int64 shape [B, 1, *grid.shape].")
            if counts.device != features.device:
                raise ValueError(f"{name} must share the feature device.")
            if torch.any(counts < 0):
                raise ValueError(f"{name} must be non-negative.")
        mass = bev.camera_weight_sum
        if mass.shape != evidence_shape or mass.dtype != features.dtype or mass.device != features.device:
            raise ValueError("camera_weight_sum must share the evidence shape and feature dtype/device.")
        if not torch.isfinite(mass).all() or torch.any(mass < 0):
            raise ValueError("camera_weight_sum must be finite and non-negative.")
        if not torch.equal(bev.camera_pixel_count > 0, mass > 0):
            raise ValueError("camera count and weight sum must have the same positive support.")

        fine = self.fine(features)
        middle = self.middle(F.relu(self.down1(fine)))
        deep = self.deep(F.relu(self.down2(middle)))
        # Paired output sizes retain the encoder lattice, not merely its shape.
        decoded = F.relu(middle + self.up1(deep, output_size=middle.shape))
        output = self.output(F.relu(fine + self.up0(decoded, output_size=fine.shape)))
        return FusedBEVFeatures(
            features=output, grid=bev.grid,
            camera_pixel_count=bev.camera_pixel_count.clone(),
            camera_weight_sum=mass.clone(),
            lidar_point_count=bev.lidar_point_count.clone(),
        )
