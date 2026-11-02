"""Explicit depth-source and support policies for the future camera BEV lift."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .camera_geometry import normalized_depth_to_meters


@dataclass(frozen=True)
class CameraDepthSupport:
    """Selected source, safe metric depth, valid pixels, and optional splat score.

    depth_m and valid_mask are [B, 1, H, W]. Invalid depth entries are zero
    numerical placeholders, not measurements; downstream geometry must exclude
    their XYZ using valid_mask. No inputs are modified or detached.

    splat_confidence is available only for the splatted source. It is the
    unmodified heuristic map, not a probability or learned-depth uncertainty.
    Only valid pixels may contribute to future BEV aggregation/weighting.
    """

    source: str
    depth_m: torch.Tensor
    valid_mask: torch.Tensor
    splat_confidence: torch.Tensor | None


def build_depth_valid_mask(
    depth_m: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    max_valid_depth_m: float,
    min_valid_depth_m: float = 0.0,
) -> torch.Tensor:
    """Intersect explicit support with finite min < camera-z <= max, in meters.

    Bounds must satisfy 0 <= min < max. The maximum is a geometric selection
    limit, independent of the scale used to normalize network inputs/outputs.
    A finite, positive prediction alone does not establish reliable support.
    """

    if (
        depth_m.ndim != 4 or depth_m.shape[1] != 1
        or any(size == 0 for size in depth_m.shape)
        or not depth_m.is_floating_point()
    ):
        raise ValueError("depth_m must have non-empty floating shape [B, 1, H, W].")
    if (
        support_mask.shape != depth_m.shape or support_mask.dtype != torch.bool
        or support_mask.device != depth_m.device
    ):
        raise ValueError("support_mask must be bool with depth_m's shape and device.")
    if (
        not math.isfinite(min_valid_depth_m) or not math.isfinite(max_valid_depth_m)
        or not 0.0 <= min_valid_depth_m < max_valid_depth_m
    ):
        raise ValueError("depth bounds must be finite with 0 <= min < max.")
    return (
        support_mask & torch.isfinite(depth_m)
        & (depth_m > min_valid_depth_m) & (depth_m <= max_valid_depth_m)
    )


def prepare_camera_depth_support(
    lidar_maps: torch.Tensor,
    *,
    source: str,
    max_depth_m: float | torch.Tensor,
    max_valid_depth_m: float,
    min_valid_depth_m: float = 0.0,
    min_splat_confidence: float = 0.0,
    depth_prediction: torch.Tensor | None = None,
    prediction_support_mask: torch.Tensor | None = None,
) -> CameraDepthSupport:
    """Select sparse, splatted, or predicted depth without reading targets.

    lidar_maps is the actual model input [B, 6/8, H, W], after holdout and
    resplatting when applicable. Never pass the full pre-holdout maps here.
    Channels 0/5 are sparse depth/presence; 6/7 are splat depth/confidence.
    max_depth_m must match the input/checkpoint normalization scale.

    sparse uses only retained measurements. splatted additionally requires
    finite 0 < confidence <= 1 and confidence >= min_splat_confidence.
    predicted requires normalized depth_prediction [B, 1, H, W] and an
    explicit bool prediction_support_mask. It does not inherit splat scores
    as uncertainty, or silently use an all-true mask. A caller can deliberately
    select a wider prediction domain later, but that does not prove reliability.

    Zeroing invalid outputs prevents NaN/Inf contamination of a subsequent
    dense linear solve. This is not a clamp that promotes invalid depth to
    valid depth: valid_mask remains false and must govern BEV inclusion.
    """

    if (
        lidar_maps.ndim != 4 or lidar_maps.shape[1] not in (6, 8)
        or any(size == 0 for size in lidar_maps.shape)
        or not lidar_maps.is_floating_point()
    ):
        raise ValueError("lidar_maps must have non-empty floating shape [B, 6/8, H, W].")
    if source not in ("sparse", "splatted", "predicted"):
        raise ValueError("source must be 'sparse', 'splatted', or 'predicted'.")
    if not math.isfinite(min_splat_confidence) or not 0.0 <= min_splat_confidence <= 1.0:
        raise ValueError("min_splat_confidence must be finite and between 0 and 1.")
    if source != "predicted" and (
        depth_prediction is not None or prediction_support_mask is not None
    ):
        raise ValueError("prediction arguments require source='predicted'.")

    confidence = None
    if source == "sparse":
        depth = lidar_maps[:, 0:1]
        presence = lidar_maps[:, 5:6]
        support = torch.isfinite(presence) & (presence > 0)
    elif source == "splatted":
        if lidar_maps.shape[1] != 8:
            raise ValueError("splatted depth requires 8 LiDAR channels.")
        depth = lidar_maps[:, 6:7]
        confidence = lidar_maps[:, 7:8]
        support = (
            torch.isfinite(confidence) & (confidence > 0) & (confidence <= 1)
            & (confidence >= min_splat_confidence)
        )
    else:
        if depth_prediction is None or prediction_support_mask is None:
            raise ValueError("predicted depth requires depth_prediction and prediction_support_mask.")
        if (
            depth_prediction.shape != lidar_maps[:, 0:1].shape
            or depth_prediction.dtype != lidar_maps.dtype
            or depth_prediction.device != lidar_maps.device
        ):
            raise ValueError("depth_prediction must share sparse depth's shape, dtype, and device.")
        depth = depth_prediction
        support = prediction_support_mask

    depth_m = normalized_depth_to_meters(depth, max_depth_m=max_depth_m)
    valid = build_depth_valid_mask(
        depth_m, support_mask=support,
        min_valid_depth_m=min_valid_depth_m, max_valid_depth_m=max_valid_depth_m,
    )
    return CameraDepthSupport(
        source=source,
        depth_m=torch.where(valid, depth_m, torch.zeros_like(depth_m)),
        valid_mask=valid,
        splat_confidence=confidence,
    )
