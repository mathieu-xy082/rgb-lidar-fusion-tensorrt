"""Optional LiDAR-only Car-center head, masked objective, and point decoding.

This diagnostic is not required by the eventual fused detector. It predicts
centers only, not 3D sizes, yaw, velocity, or KITTI detection AP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .camera_bev import BEVGridConfig
from .lidar_bev import LidarBEVContext, PillarFeatureNet, scatter_pillar_features


@dataclass(frozen=True)
class CenterPrediction:
    heatmap_logits: torch.Tensor  # [B,1,Hbev,Wbev]
    offset_xy: torch.Tensor  # [B,2,Hbev,Wbev], sigmoid row-x / column-y fractions


class LidarCenterHead(nn.Module):
    """Two 1x1 heads preserve the encoder's spatial field and grid."""

    def __init__(self, *, channels: int = 32) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, Integral) or channels < 1:
            raise ValueError("channels must be a positive integer.")
        self.channels = int(channels)
        self.heatmap = nn.Conv2d(self.channels, 1, 1)
        self.offset = nn.Conv2d(self.channels, 2, 1)
        nn.init.constant_(self.heatmap.bias, -2.19)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, features: torch.Tensor) -> CenterPrediction:
        if features.ndim != 4 or features.shape[1] != self.channels or min(features.shape[2:]) < 1:
            raise ValueError("features must have shape [B, channels, Hbev, Wbev].")
        return CenterPrediction(self.heatmap(features), self.offset(features).sigmoid())


class LidarCenterDiagnostic(nn.Module):
    """Compose the existing encoder with an optional Car-center objective.

    Inputs are the explicit Torch tensors from a PillarBatch; callers control
    preprocessing and device placement. No camera input or target is consumed.
    Existing encoder classes/state schemas and the camera checkpoint stay intact.
    """

    def __init__(self, *, channels: int = 32) -> None:
        super().__init__()
        self.pfn = PillarFeatureNet(out_channels=channels)
        self.context = LidarBEVContext(channels=channels)
        self.head = LidarCenterHead(channels=channels)

    def encode(self, features, point_mask, indices, point_count):
        scattered = scatter_pillar_features(
            self.pfn(features, point_mask), indices=indices, point_count=point_count,
        )
        return scattered, self.context(scattered)

    def forward(self, features, point_mask, indices, point_count) -> CenterPrediction:
        _, bev = self.encode(features, point_mask, indices, point_count)
        return self.head(bev.features)


@dataclass(frozen=True)
class CenterLoss:
    total: torch.Tensor
    heatmap: torch.Tensor
    offset: torch.Tensor


def lidar_center_loss(
    prediction: CenterPrediction,
    *,
    heatmap: torch.Tensor,
    offset_xy: torch.Tensor,
    center_mask: torch.Tensor,
    supervision_mask: torch.Tensor,
    offset_weight: float = 1.,
) -> CenterLoss:
    """Stable Gaussian focal loss plus L1 at annotated center cells only.

    Focal exponents are 2 for predicted confidence and 4 for Gaussian tails.
    Heatmap sum is normalized by max(number of centers,1); offset L1 is averaged
    over the two coordinates of centers. Unknown/ignored cells have zero loss
    and gradient. An entirely unsupervised batch yields differentiable zero,
    not evidence of learning. The future runner must track/skip such batches.
    """

    logits, offsets = prediction.heatmap_logits, prediction.offset_xy
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("heatmap logits must have shape [B,1,Hbev,Wbev].")
    expected_offsets = (logits.shape[0], 2, *logits.shape[2:])
    if heatmap.shape != logits.shape or offsets.shape != expected_offsets or offset_xy.shape != offsets.shape:
        raise ValueError("heatmap and offset targets must match prediction shapes.")
    if center_mask.shape != logits.shape or supervision_mask.shape != logits.shape:
        raise ValueError("masks must match the heatmap shape.")
    if center_mask.dtype != torch.bool or supervision_mask.dtype != torch.bool:
        raise ValueError("masks must be boolean.")
    tensors = (heatmap, offsets, offset_xy, center_mask, supervision_mask)
    if any(value.device != logits.device for value in tensors):
        raise ValueError("all predictions, targets and masks must share a device.")
    if logits.dtype not in (torch.float32, torch.float64) or any(
        value.dtype != logits.dtype for value in (heatmap, offsets, offset_xy)
    ):
        raise ValueError("predictions and targets must share float32/float64 dtype.")
    if not math.isfinite(offset_weight) or offset_weight < 0:
        raise ValueError("offset_weight must be finite and non-negative.")
    if torch.any(center_mask & ~supervision_mask) or not torch.equal(center_mask, heatmap == 1):
        raise ValueError("center masks must identify all unit heatmap peaks and be supervised.")
    if not torch.isfinite(heatmap).all() or torch.any((heatmap < 0) | (heatmap > 1)):
        raise ValueError("heatmap targets must be finite in [0,1].")
    selected_logits, selected_target = logits[supervision_mask], heatmap[supervision_mask]
    selected_offsets = center_mask.expand_as(offsets)
    predicted_offsets, target_offsets = offsets[selected_offsets], offset_xy[selected_offsets]
    if not torch.isfinite(selected_logits).all() or not torch.isfinite(predicted_offsets).all():
        raise ValueError("selected predictions must be finite.")
    if not torch.isfinite(target_offsets).all() or torch.any((target_offsets < 0) | (target_offsets >= 1)):
        raise ValueError("selected offset targets must lie in [0,1).")
    probability = selected_logits.sigmoid()
    terms = torch.where(
        selected_target == 1,
        F.softplus(-selected_logits) * (1 - probability).square(),
        F.softplus(selected_logits) * probability.square() * (1 - selected_target).pow(4),
    )
    count = center_mask.sum().clamp_min(1)
    heatmap_loss = terms.sum() / count
    offset_loss = (predicted_offsets - target_offsets).abs().sum() / (2 * count)
    return CenterLoss(heatmap_loss + offset_weight * offset_loss, heatmap_loss, offset_loss)


def decode_lidar_centers(
    prediction: CenterPrediction, *, grid: BEVGridConfig, score_threshold: float = .3, top_k: int = 50,
    valid_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Return per-sample [N,3] x/y/score after 3x3 local-maximum filtering.

    Decode x=x_min+(row+offset_x)*resolution, likewise column/y. There is no
    additional half-cell shift: offset .5 already denotes the cell center.
    An optional evaluation-domain mask excludes unknown cells, not model input.
    This is center decoding, not oriented-box NMS or an AP evaluation.
    """

    logits, offsets = prediction.heatmap_logits, prediction.offset_xy
    if logits.ndim != 4 or logits.shape[1:] != (1, *grid.shape):
        raise ValueError("prediction heatmap must match the BEV grid.")
    if offsets.shape != (logits.shape[0], 2, *grid.shape) or offsets.device != logits.device:
        raise ValueError("offsets must match heatmap batch/grid/device.")
    if logits.dtype not in (torch.float32, torch.float64) or offsets.dtype != logits.dtype:
        raise ValueError("predictions must share float32/float64 dtype.")
    if not 0 <= score_threshold <= 1 or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("invalid score threshold or top_k.")
    if not torch.isfinite(logits).all() or not torch.isfinite(offsets).all():
        raise ValueError("predictions must be finite.")
    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    if valid_mask.shape != logits.shape or valid_mask.dtype != torch.bool or valid_mask.device != logits.device:
        raise ValueError("valid_mask must be boolean and match the heatmap.")
    scores = logits.sigmoid().masked_fill(~valid_mask, -1.)
    peaks = scores == F.max_pool2d(scores, 3, stride=1, padding=1)
    scores = scores.masked_fill(~peaks, -1.)
    decoded = []
    for index in range(len(scores)):
        values, cells = scores[index, 0].flatten().topk(min(top_k, scores[index, 0].numel()))
        cells, values = cells[values >= score_threshold], values[values >= score_threshold]
        rows, cols = cells // grid.shape[1], cells % grid.shape[1]
        delta = offsets[index, :, rows, cols].clamp(0, 1 - torch.finfo(offsets.dtype).eps)
        xy = torch.stack([
            grid.x_range_m[0] + (rows + delta[0]) * grid.resolution_m,
            grid.y_range_m[0] + (cols + delta[1]) * grid.resolution_m,
        ], dim=1)
        decoded.append(torch.cat([xy, values[:, None]], dim=1))
    return decoded
