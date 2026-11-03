"""NumPy-first raw XYZI grouping and point features, without a learned encoder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .camera_bev import BEVGridConfig


PILLAR_FEATURE_CHANNELS = (
    "x", "y", "z", "intensity",
    "cluster_offset_x", "cluster_offset_y", "cluster_offset_z",
    "pillar_offset_x", "pillar_offset_y",
)


@dataclass(frozen=True)
class PillarBatch:
    """Occupied pillars across a batch, ordered by (batch, row-x, column-y).

    features: float32 [P, M, 9], with zero padding after retained points.
    indices: int64 [P, 3], containing (batch_index, row, column).
    point_mask: bool [P, M]; only true entries may enter a future PFN.
    point_indices: int64 [P, M], original per-cloud indices, or -1 for padding.
    point_count: int64 [B, 1, Hbev, Wbev], before the per-pillar cap.
    grid: the supplied shared camera/LiDAR metric grid.

    P is the number of occupied pillars across all samples; M is the cap.
    No empty pillars are materialized in features. Measurement evidence is
    independent of future convolutional activations, not an occupancy score.
    """

    features: np.ndarray
    indices: np.ndarray
    point_mask: np.ndarray
    point_indices: np.ndarray
    point_count: np.ndarray
    grid: BEVGridConfig

    @property
    def retained_count(self) -> np.ndarray:
        """Return [P] counts of retained (non-padding) points."""

        return self.point_mask.sum(axis=1, dtype=np.int64)

    @property
    def dropped_count(self) -> np.ndarray:
        """Return [P] overflow counts, excluding points filtered out of the ROI."""

        b, row, col = self.indices.T
        return self.point_count[b, 0, row, col] - self.retained_count

    @property
    def observed_mask(self) -> np.ndarray:
        """Return bool [B, 1, Hbev, Wbev] presence of raw measurements."""

        return self.point_count > 0


def build_lidar_pillars(
    point_clouds: Sequence[np.ndarray],
    *,
    grid: BEVGridConfig,
    max_points_per_pillar: int = 64,
    seed: int = 13,
) -> PillarBatch:
    """Group a sequence of variable-length float32/float64 [Ni, 4] XYZI clouds.

    XYZ is in meters in the shared grid's vehicle/LiDAR frame. Intensity is
    copied unchanged (no normalization or clipping). Empty [0, 4] samples and
    an empty batch are valid. A single cloud must be wrapped in a sequence;
    dense/padded arrays and implicit presence masks are not accepted.

    Inputs are converted to float32, like the camera lift's nominal geometry.
    Non-finite XYZI, including intensity and values overflowing float32, raise
    ValueError before ROI filtering. Finite points outside half-open XYZ bounds
    are ignored, with no camera-FOV filtering and no clamping to edge cells.
    Cell division uses float32; rounded out-of-grid indices are excluded just
    as in the camera lift. Other grid resolutions remain usable explicitly.

    Overflow is sampled without replacement using a local generator keyed by
    (seed, batch index, row, column). Repeating the same inputs/order and seed
    reproduces the selection without touching global NumPy RNG state. Selection
    may change if point order or batch index changes. Retained indices are sorted
    back into source order; centroid offsets use only these points, not padding
    or discarded points. No total-pillar cap is applied.

    This is CPU preprocessing, not an autograd operation or a dataset/training
    integration. It does not mutate inputs or introduce learned parameters.
    """

    if not isinstance(point_clouds, Sequence) or isinstance(point_clouds, (str, bytes)):
        raise TypeError("point_clouds must be a sequence of [Ni, 4] NumPy arrays.")
    if (
        isinstance(max_points_per_pillar, bool)
        or not isinstance(max_points_per_pillar, Integral)
        or max_points_per_pillar < 1
    ):
        raise ValueError("max_points_per_pillar must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    cap = int(max_points_per_pillar)
    height, width = grid.shape
    point_count = np.zeros((len(point_clouds), 1, height, width), dtype=np.int64)
    feature_batches = []
    index_batches = []
    mask_batches = []
    source_index_batches = []

    for batch_index, cloud in enumerate(point_clouds):
        if not isinstance(cloud, np.ndarray):
            raise TypeError("each point cloud must be a NumPy array.")
        if cloud.ndim != 2 or cloud.shape[1] != 4:
            raise ValueError("each point cloud must have shape [Ni, 4] (XYZI).")
        if cloud.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("point clouds must have float32 or float64 dtype.")
        with np.errstate(over="ignore", invalid="ignore"):
            points = cloud.astype(np.float32, copy=False)
        if not np.isfinite(points).all():
            raise ValueError("point clouds must contain finite XYZI representable as float32.")

        inside = np.ones(len(points), dtype=bool)
        for axis, bounds in enumerate((grid.x_range_m, grid.y_range_m, grid.z_range_m)):
            inside &= (points[:, axis] >= bounds[0]) & (points[:, axis] < bounds[1])
        source_indices = np.flatnonzero(inside)
        rows = np.floor(
            (points[source_indices, 0] - grid.x_range_m[0]) / grid.resolution_m,
        ).astype(np.int64)
        cols = np.floor(
            (points[source_indices, 1] - grid.y_range_m[0]) / grid.resolution_m,
        ).astype(np.int64)
        in_grid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows, cols, source_indices = rows[in_grid], cols[in_grid], source_indices[in_grid]

        # Stable grouping preserves source order inside each occupied cell.
        cell_ids = rows * width + cols
        order = np.argsort(cell_ids, kind="stable")
        cells, starts, counts = np.unique(cell_ids[order], return_index=True, return_counts=True)
        pillar_rows, pillar_cols = cells // width, cells % width
        point_count[batch_index, 0].reshape(-1)[cells] = counts
        features = np.zeros((len(cells), cap, len(PILLAR_FEATURE_CHANNELS)), dtype=np.float32)
        point_mask = np.zeros((len(cells), cap), dtype=bool)
        point_indices = np.full((len(cells), cap), -1, dtype=np.int64)
        indices = np.column_stack((
            np.full(len(cells), batch_index, dtype=np.int64), pillar_rows, pillar_cols,
        ))

        for pillar_index, (start, count) in enumerate(zip(starts, counts, strict=True)):
            selected = source_indices[order[start:start + count]]
            if count > cap:
                rng = np.random.default_rng(np.random.SeedSequence([
                    int(seed), batch_index,
                    int(pillar_rows[pillar_index]), int(pillar_cols[pillar_index]),
                ]))
                selected = np.sort(rng.choice(selected, size=cap, replace=False))
            retained = points[selected]
            size = len(retained)
            point_mask[pillar_index, :size] = True
            point_indices[pillar_index, :size] = selected
            features[pillar_index, :size, :4] = retained
            center = np.array([
                grid.x_range_m[0] + (pillar_rows[pillar_index] + 0.5) * grid.resolution_m,
                grid.y_range_m[0] + (pillar_cols[pillar_index] + 0.5) * grid.resolution_m,
            ], dtype=np.float32)
            centroid = retained[:, :3].mean(axis=0, dtype=np.float64).astype(np.float32)
            features[pillar_index, :size, 4:7] = retained[:, :3] - centroid
            features[pillar_index, :size, 7:9] = retained[:, :2] - center

        feature_batches.append(features)
        index_batches.append(indices)
        mask_batches.append(point_mask)
        source_index_batches.append(point_indices)

    return PillarBatch(
        features=np.concatenate(feature_batches) if feature_batches else np.zeros(
            (0, cap, len(PILLAR_FEATURE_CHANNELS)), dtype=np.float32,
        ),
        indices=np.concatenate(index_batches) if index_batches else np.empty((0, 3), dtype=np.int64),
        point_mask=np.concatenate(mask_batches) if mask_batches else np.zeros((0, cap), dtype=bool),
        point_indices=np.concatenate(source_index_batches) if source_index_batches else np.empty(
            (0, cap), dtype=np.int64,
        ),
        point_count=point_count,
        grid=grid,
    )
