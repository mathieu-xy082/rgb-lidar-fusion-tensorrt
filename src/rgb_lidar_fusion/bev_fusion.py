"""Cell-local camera/LiDAR BEV feature fusion with separate evidence maps."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn

from .camera_bev import BEVGridConfig, CameraBEVFeatures
from .lidar_bev import LidarBEVFeatures


@dataclass(frozen=True)
class FusedBEVFeatures:
    """Learned [B, C, Hbev, Wbev] features and unchanged measurement evidence.

    Evidence tensors are independent snapshots, all [B, 1, Hbev, Wbev].
    Counts are int64; camera_weight_sum shares the feature dtype/device.
    Derived masks describe contributions/observations, not free space,
    object scores, calibrated confidence, or the extent of contextual features.
    The grid retains rows along vehicle x and columns along vehicle y.
    """

    features: torch.Tensor
    grid: BEVGridConfig
    camera_pixel_count: torch.Tensor
    camera_weight_sum: torch.Tensor
    lidar_point_count: torch.Tensor

    @property
    def camera_support(self) -> torch.Tensor:
        return self.camera_pixel_count > 0

    @property
    def lidar_observed_mask(self) -> torch.Tensor:
        return self.lidar_point_count > 0


class BEVFeatureFusion(nn.Module):
    """Concatenate camera, contextual LiDAR and two masks, then apply a 1x1.

    Defaults are 32 camera + 32 LiDAR + 2 support channels -> 32 outputs.
    The projection is bias-free and linear, with no activation, normalization,
    learned gate or confidence multiplication. It mixes channels within each
    cell only; the shared spatial backbone is a separate, later component.

    Supply both preprocessing grids explicitly: equal shapes do not imply
    equal metric alignment. Callers remain responsible for sample identity,
    batch ordering, calibration and the actual provenance of the declared grids.
    No resize, crop, transpose, device move or floating dtype cast is performed.

    Unsupported camera cells must contain zero lifted features. LiDAR context
    may be non-zero outside raw observations and is never erased by its mask.
    B >= 1 is required, including all-zero/unsupported samples; this does not
    change the standalone LiDAR encoder's empty-batch behavior.
    """

    def __init__(
        self,
        *,
        camera_channels: int = 32,
        lidar_channels: int = 32,
        out_channels: int = 32,
    ) -> None:
        super().__init__()
        for name, value in (
            ("camera_channels", camera_channels),
            ("lidar_channels", lidar_channels),
            ("out_channels", out_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
            setattr(self, name, int(value))
        self.projection = nn.Conv2d(
            self.camera_channels + self.lidar_channels + 2,
            self.out_channels, kernel_size=1, bias=False,
        )

    def forward(
        self,
        camera_bev: CameraBEVFeatures,
        lidar_bev: LidarBEVFeatures,
        *,
        camera_grid: BEVGridConfig,
        lidar_grid: BEVGridConfig,
    ) -> FusedBEVFeatures:
        """Validate aligned features/evidence, preserving both feature graphs.

        Camera pixel_mask belongs to the image grid and is not consumed here.
        Neither count map nor camera aggregation mass is a learned input:
        only their two boolean support indicators are appended to features.
        Projection gradients reach both feature maps, not discrete placement.
        """

        if not isinstance(camera_grid, BEVGridConfig) or not isinstance(lidar_grid, BEVGridConfig):
            raise ValueError("camera_grid and lidar_grid must be explicit BEVGridConfig values.")
        if camera_grid.resolution_m != lidar_grid.resolution_m or any(
            tuple(getattr(camera_grid, axis)) != tuple(getattr(lidar_grid, axis))
            for axis in ("x_range_m", "y_range_m", "z_range_m")
        ):
            raise ValueError("camera_grid and lidar_grid must have identical metric bounds and resolution.")

        camera, lidar = camera_bev.features, lidar_bev.features
        weight = self.projection.weight
        for name, features, channels in (
            ("camera", camera, self.camera_channels),
            ("LiDAR", lidar, self.lidar_channels),
        ):
            if (
                features.ndim != 4 or features.shape[0] < 1
                or features.shape[1] != channels or features.shape[2:] != camera_grid.shape
            ):
                raise ValueError(f"{name} features must have non-empty shape [B, channels, *grid.shape].")
            if features.dtype not in (torch.float32, torch.float64):
                raise ValueError(f"{name} features must have float32 or float64 dtype.")
            if features.device != weight.device or features.dtype != weight.dtype:
                raise ValueError(f"{name} features must share the module's device and dtype.")
            if not torch.isfinite(features).all():
                raise ValueError(f"{name} features must be finite.")
        if camera.shape[0] != lidar.shape[0]:
            raise ValueError("camera and LiDAR features must share the same batch size.")

        evidence_shape = (camera.shape[0], 1, *camera_grid.shape)
        for name, counts in (
            ("camera pixel_count", camera_bev.pixel_count),
            ("LiDAR point_count", lidar_bev.point_count),
        ):
            if counts.shape != evidence_shape or counts.dtype != torch.int64:
                raise ValueError(f"{name} must have int64 shape [B, 1, *grid.shape].")
            if counts.device != camera.device:
                raise ValueError(f"{name} must share the feature device.")
            if torch.any(counts < 0):
                raise ValueError(f"{name} must be non-negative.")

        mass = camera_bev.weight_sum
        if mass.shape != evidence_shape or mass.dtype != camera.dtype or mass.device != camera.device:
            raise ValueError("camera weight_sum must share the evidence shape and feature dtype/device.")
        if not torch.isfinite(mass).all() or torch.any(mass < 0):
            raise ValueError("camera weight_sum must be finite and non-negative.")
        camera_support = camera_bev.pixel_count > 0
        lidar_observed = lidar_bev.observed_mask
        if not torch.equal(camera_support, mass > 0):
            raise ValueError("camera pixel_count and weight_sum must have the same positive support.")
        if torch.any((camera != 0).any(dim=1, keepdim=True) & ~camera_support):
            raise ValueError("camera features must be zero outside camera support.")

        combined = torch.cat([
            camera, lidar,
            camera_support.to(dtype=camera.dtype), lidar_observed.to(dtype=camera.dtype),
        ], dim=1)
        return FusedBEVFeatures(
            features=self.projection(combined), grid=camera_grid,
            camera_pixel_count=camera_bev.pixel_count.clone(),
            camera_weight_sum=mass.clone(),
            lidar_point_count=lidar_bev.point_count.clone(),
        )
