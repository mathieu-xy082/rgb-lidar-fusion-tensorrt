"""Local LiDAR depth splatting utilities.

This module implements a deliberately simple level-1 spatial splat: each sparse
LiDAR depth sample is copied into a small image-space neighbourhood with a
Gaussian confidence decay. It does not use RGB edges, estimate planes, or infer
concave surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplattingConfig:
    """Configuration for local projected-LiDAR depth splatting.

    Args:
        radius_px: Inclusive pixel radius around each sparse source point.
        sigma_px: Gaussian decay sigma in pixels. Confidence is
            ``exp(-distance_px**2 / (2 * sigma_px**2))``.
        conflict_strategy: Deterministic overlap rule. The current supported
            value, ``"nearest_depth_then_confidence"``, keeps the nearest
            (smallest positive) depth when several source points influence a
            pixel; equal depths keep the higher confidence; remaining exact ties
            keep the earlier source in row-major image order.
    """

    radius_px: int = 2
    sigma_px: float = 1.0
    conflict_strategy: str = "nearest_depth_then_confidence"

    def __post_init__(self) -> None:
        if self.radius_px < 0:
            raise ValueError("radius_px must be non-negative.")
        if self.sigma_px <= 0.0:
            raise ValueError("sigma_px must be positive.")
        if self.conflict_strategy != "nearest_depth_then_confidence":
            raise ValueError(
                "conflict_strategy must be 'nearest_depth_then_confidence'."
            )


@dataclass(frozen=True)
class SplattedDepth:
    """Dense-ish local splat output plus untouched sparse inputs."""

    depth_expanded: np.ndarray
    confidence: np.ndarray
    sparse_depth: np.ndarray
    sparse_mask: np.ndarray


def splat_sparse_depth(
    sparse_depth: np.ndarray,
    sparse_mask: np.ndarray,
    config: SplattingConfig | None = None,
) -> SplattedDepth:
    """Expand sparse depth samples into local image-space splats.

    Args:
        sparse_depth: ``[H, W]`` depth map. Values are copied as-is, so callers
            may pass metric or normalized depth as long as they are consistent.
        sparse_mask: ``[H, W]`` boolean/0-1 mask identifying valid sparse depth
            samples. The raw sparse map is preserved separately in the result.
        config: Splatting radius, confidence decay, and conflict rule.

    Returns:
        ``SplattedDepth`` with ``depth_expanded`` and ``confidence`` aligned to
        the input image grid, plus copies of ``sparse_depth`` and ``sparse_mask``.

    Notes:
        This is intentionally not image-guided and does not replace the sparse
        representation. Pixels outside all kernels remain zero depth and zero
        confidence. Overlaps are deterministic: nearest depth wins, then higher
        confidence, then earlier row-major source order.
    """

    cfg = config or SplattingConfig()
    depth = np.asarray(sparse_depth, dtype=np.float32)
    mask = np.asarray(sparse_mask, dtype=bool)
    if depth.ndim != 2:
        raise ValueError("sparse_depth must have shape [H, W].")
    if mask.shape != depth.shape:
        raise ValueError("sparse_mask must have the same shape as sparse_depth.")

    height, width = depth.shape
    depth_expanded = np.zeros((height, width), dtype=np.float32)
    confidence = np.zeros((height, width), dtype=np.float32)
    source_order = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)

    source_pixels = np.argwhere(mask)
    for order, (src_v, src_u) in enumerate(source_pixels):
        src_depth = depth[src_v, src_u]
        if src_depth <= 0.0:
            continue

        v_min = max(0, int(src_v) - cfg.radius_px)
        v_max = min(height - 1, int(src_v) + cfg.radius_px)
        u_min = max(0, int(src_u) - cfg.radius_px)
        u_max = min(width - 1, int(src_u) + cfg.radius_px)

        for v in range(v_min, v_max + 1):
            dv = float(v - src_v)
            for u in range(u_min, u_max + 1):
                du = float(u - src_u)
                distance_sq = du * du + dv * dv
                candidate_confidence = float(
                    np.exp(-distance_sq / (2.0 * cfg.sigma_px * cfg.sigma_px))
                )
                if _candidate_wins(
                    candidate_depth=float(src_depth),
                    candidate_confidence=candidate_confidence,
                    candidate_order=order,
                    current_depth=float(depth_expanded[v, u]),
                    current_confidence=float(confidence[v, u]),
                    current_order=int(source_order[v, u]),
                ):
                    depth_expanded[v, u] = src_depth
                    confidence[v, u] = candidate_confidence
                    source_order[v, u] = order

    return SplattedDepth(
        depth_expanded=depth_expanded,
        confidence=confidence,
        sparse_depth=depth.copy(),
        sparse_mask=mask.copy(),
    )


def _candidate_wins(
    *,
    candidate_depth: float,
    candidate_confidence: float,
    candidate_order: int,
    current_depth: float,
    current_confidence: float,
    current_order: int,
) -> bool:
    if current_confidence == 0.0:
        return True
    if candidate_depth < current_depth:
        return True
    if candidate_depth > current_depth:
        return False
    if candidate_confidence > current_confidence:
        return True
    if candidate_confidence < current_confidence:
        return False
    return candidate_order < current_order
