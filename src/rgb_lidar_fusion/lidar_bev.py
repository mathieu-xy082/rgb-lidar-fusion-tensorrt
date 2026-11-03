"""Masked pillar encoding, stride-1 scatter, and local spatial BEV context."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .lidar_pillars import PILLAR_FEATURE_CHANNELS


class PillarFeatureNet(nn.Module):
    """Encode [P, M, 9] point features into [P, C] pillar vectors (C=32).

    A shared bias-free linear layer, BatchNorm1d, and ReLU are applied only
    to points selected by point_mask. BatchNorm statistics aggregate valid
    points across pillars, never padding. Channel-wise max then pools each
    pillar. Fully masked pillars return zeros, not learned observations.

    In training with fewer than two valid points in total, BatchNorm uses its
    running statistics without updating them, while retaining affine/input
    gradients. This includes empty input and avoids undefined singleton batch
    variance. Initial running statistics are mean=0 and variance=1, not a
    claim of calibrated features. Normal training batches update statistics.

    Inputs must already share the module's dtype/device. This module has no
    spatial context, raw-point preprocessing, detection head, or training loss.
    """

    def __init__(self, *, out_channels: int = 32) -> None:
        super().__init__()
        if isinstance(out_channels, bool) or not isinstance(out_channels, Integral) or out_channels < 1:
            raise ValueError("out_channels must be a positive integer.")
        self.out_channels = int(out_channels)
        self.linear = nn.Linear(len(PILLAR_FEATURE_CHANNELS), self.out_channels, bias=False)
        self.norm = nn.BatchNorm1d(self.out_channels, eps=1e-3, momentum=0.01)

    def forward(self, features: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """Exclude padding before both normalization and channel-wise pooling."""

        if features.ndim != 3 or features.shape[1] < 1 or features.shape[2] != len(PILLAR_FEATURE_CHANNELS):
            raise ValueError("features must have shape [P, M, 9] with M >= 1.")
        if features.dtype not in (torch.float32, torch.float64):
            raise ValueError("features must have float32 or float64 dtype.")
        if point_mask.shape != features.shape[:2] or point_mask.dtype != torch.bool:
            raise ValueError("point_mask must be boolean with shape [P, M].")
        if (
            features.device != self.linear.weight.device
            or features.dtype != self.linear.weight.dtype
            or point_mask.device != features.device
        ):
            raise ValueError("features must share the module's device and dtype; point_mask must share the feature device.")
        points = features[point_mask]
        if not torch.isfinite(points).all():
            raise ValueError("selected point features must be finite.")

        encoded = self.linear(points)
        if self.training and encoded.shape[0] < 2:
            encoded = F.batch_norm(
                encoded, self.norm.running_mean, self.norm.running_var,
                self.norm.weight, self.norm.bias, training=False, eps=self.norm.eps,
            )
        else:
            encoded = self.norm(encoded)
        encoded = F.relu(encoded)

        # Select before the linear/BN path, then restore only for per-pillar max.
        # Empty inputs retain the graph and yield finite zero gradients.
        padded = features.new_full((*features.shape[:2], self.out_channels), -torch.inf)
        padded[point_mask] = encoded
        pooled = padded.max(dim=1).values
        return torch.where(point_mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))


@dataclass(frozen=True)
class LidarBEVFeatures:
    """Stride-1 features [B, C, Hbev, Wbev] and pre-cap measurement evidence.

    point_count is a cloned int64 [B, 1, Hbev, Wbev] snapshot. Its derived
    observed_mask expresses direct raw-point support, not feature magnitude,
    free space, occupancy probability, or camera/splat confidence.
    """

    features: torch.Tensor
    point_count: torch.Tensor

    @property
    def observed_mask(self) -> torch.Tensor:
        return self.point_count > 0


class _BEVResidualBlock(nn.Module):
    def __init__(self, channels: int, dilations: tuple[int, int]) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features + self.convs(features))


class LidarBEVContext(nn.Module):
    """Add pre-fusion context without changing the metric grid or raw evidence.

    Two residual blocks use 3x3 convolutions with dilations (1, 2) and (2, 4).
    All four have stride 1 and padding equal to dilation, preserving cell
    centers even for odd/non-square grids. Width stays constant (default 32).
    No bias or normalization is added: zero input stays zero, and spatial
    normalization statistics cannot introduce dependencies outside the local
    convolutional field. The PFN's own BatchNorm is unchanged.

    The maximum theoretical field is 1 + 2*(1+2+2+4) = 19 cells per axis,
    or 3.8 m at 20 cm resolution, not a guaranteed learned/effective field or
    proof of object recognition. At ROI edges, padding adds no observations.
    Non-zero output can propagate into unobserved neighboring cells without
    changing point_count/observed_mask. No support mask is reapplied to erase it.

    Inputs and model must already share dtype/device (float32 or float64).
    Counts are cloned into the result; inputs are never mutated. Empty batches
    and empty samples are valid. No fusion, detector, or training runner is added.
    """

    dilations = (1, 2, 2, 4)
    receptive_field_cells = 19

    def __init__(self, *, channels: int = 32) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, Integral) or channels < 1:
            raise ValueError("channels must be a positive integer.")
        self.channels = int(channels)
        self.blocks = nn.Sequential(
            _BEVResidualBlock(self.channels, self.dilations[:2]),
            _BEVResidualBlock(self.channels, self.dilations[2:]),
        )

    def forward(self, bev: LidarBEVFeatures) -> LidarBEVFeatures:
        features, counts = bev.features, bev.point_count
        if (
            features.ndim != 4 or features.shape[1] != self.channels
            or features.shape[2] < 1 or features.shape[3] < 1
        ):
            raise ValueError("features must have shape [B, channels, Hbev, Wbev] with positive spatial sizes.")
        if features.dtype not in (torch.float32, torch.float64):
            raise ValueError("features must have float32 or float64 dtype.")
        parameter = next(self.parameters())
        if features.device != parameter.device or features.dtype != parameter.dtype:
            raise ValueError("features must share the module's device and dtype.")
        if counts.shape != (features.shape[0], 1, *features.shape[2:]) or counts.dtype != torch.int64:
            raise ValueError("point_count must have int64 shape [B, 1, Hbev, Wbev] aligned with features.")
        if counts.device != features.device:
            raise ValueError("point_count must share the feature device.")
        if not torch.isfinite(features).all() or torch.any(counts < 0):
            raise ValueError("features must be finite and point_count non-negative.")
        return LidarBEVFeatures(features=self.blocks(features), point_count=counts.clone())


def scatter_pillar_features(
    pillar_features: torch.Tensor,
    *,
    indices: torch.Tensor,
    point_count: torch.Tensor,
) -> LidarBEVFeatures:
    """Place one [P, C] feature vector at each occupied (batch, row-x, col-y).

    Use indices and pre-cap point_count from the same PillarBatch, converted to
    int64 tensors on the feature device. Counts define B, Hbev, Wbev; no axis
    transposition, interpolation, stride change, or evidence reweighting occurs.
    Every positive-count cell must have exactly one pillar; missing, duplicate,
    empty-cell, and out-of-grid indices are rejected rather than overwritten.

    Empty samples and empty batches are valid. Features stay differentiable
    to pillar vectors, including the zero-size case. Coordinates and counts
    are discrete data, not differentiable placement parameters. Runtime input
    checks are intentional; GPU latency and export are not validated here.
    """

    if pillar_features.ndim != 2 or pillar_features.shape[1] < 1:
        raise ValueError("pillar_features must have shape [P, C] with C >= 1.")
    if pillar_features.dtype not in (torch.float32, torch.float64):
        raise ValueError("pillar_features must have float32 or float64 dtype.")
    if indices.shape != (pillar_features.shape[0], 3) or indices.dtype != torch.int64:
        raise ValueError("indices must have int64 shape [P, 3].")
    if (
        point_count.ndim != 4 or point_count.shape[1] != 1
        or point_count.shape[2] < 1 or point_count.shape[3] < 1
        or point_count.dtype != torch.int64
    ):
        raise ValueError("point_count must have int64 shape [B, 1, Hbev, Wbev] with positive spatial sizes.")
    if indices.device != pillar_features.device or point_count.device != pillar_features.device:
        raise ValueError("features, indices, and point_count must share the same device.")
    if not torch.isfinite(pillar_features).all():
        raise ValueError("pillar_features must be finite.")
    if torch.any(point_count < 0):
        raise ValueError("point_count must be non-negative.")

    batch, _, height, width = point_count.shape
    b, row, col = indices.unbind(dim=1)
    if torch.any((b < 0) | (b >= batch) | (row < 0) | (row >= height) | (col < 0) | (col >= width)):
        raise ValueError("indices must lie within point_count's batch and spatial bounds.")
    cell_ids = b * (height * width) + row * width + col
    if torch.unique(cell_ids).numel() != indices.shape[0]:
        raise ValueError("indices must identify unique occupied cells.")
    if torch.any(point_count[b, 0, row, col] <= 0):
        raise ValueError("indices must refer to positive-count cells.")
    if torch.count_nonzero(point_count).item() != indices.shape[0]:
        raise ValueError("indices must cover every positive-count cell exactly once.")

    channels = pillar_features.shape[1]
    flat = pillar_features.new_zeros((batch * height * width, channels)).index_copy(
        0, cell_ids, pillar_features,
    )
    return LidarBEVFeatures(
        features=flat.reshape(batch, height, width, channels).permute(0, 3, 1, 2),
        point_count=point_count.clone(),
    )
