"""Conservative Car-center BEV targets for an optional LiDAR-only diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .calibration import CameraCalibration
from .kitti_labels import KittiLabel, label_box_vehicle

if TYPE_CHECKING:
    from .camera_bev import BEVGridConfig


@dataclass(frozen=True)
class LidarCenterTargets:
    heatmap: np.ndarray  # float32 [1,Hbev,Wbev], Gaussian peaks exactly 1
    offset_xy: np.ndarray  # float32 [2,Hbev,Wbev], fractional row-x / column-y
    center_mask: np.ndarray  # bool [1,Hbev,Wbev]
    supervision_mask: np.ndarray  # bool [1,Hbev,Wbev], not an observation mask
    centers_vehicle: np.ndarray  # float32 [N,3], accepted Car geometric centers
    corners_vehicle: np.ndarray  # float32 [N,8,3]
    ignored_label_count: int


def build_lidar_center_targets(
    labels: list[KittiLabel],
    *,
    calibration: CameraCalibration,
    image_shape: tuple[int, int],
    grid: BEVGridConfig,
    gaussian_radius: int = 2,
) -> LidarCenterTargets:
    """Build Car-only centers/offsets, not full CenterPoint detection targets.

    Negatives require the ENTIRE metric height column to project inside the
    original image. Endpoint projection suffices for this affine/projective
    segment while its depth stays positive. This intentionally excludes extra
    near-field negatives instead of assuming unlabelled space is background.

    DontCare, non-Car classes (including Van), and out-of-ROI Cars exclude the
    image-rectangle frustum at all depths. A projected segment's bounding box
    is used for conservative intersection, potentially ignoring extra cells.
    Explicit annotated Car peaks override exclusions at their own cells only.
    Gaussian tails do not promote unknown regions to supervised negatives.
    No splat, raw observation mask, target-dependent input filtering, or depth
    head is used. Two centers sharing a cell raise rather than overwrite offsets.
    """

    if isinstance(gaussian_radius, bool) or not isinstance(gaussian_radius, int) or gaussian_radius < 0:
        raise ValueError("gaussian_radius must be a non-negative integer.")
    image_h, image_w = image_shape
    if min(image_shape) < 1:
        raise ValueError("image_shape must be positive.")
    height, width = grid.shape
    x = grid.x_range_m[0] + (np.arange(height) + .5) * grid.resolution_m
    y = grid.y_range_m[0] + (np.arange(width) + .5) * grid.resolution_m
    xx, yy = np.meshgrid(x, y, indexing="ij")
    projection = calibration.projection_matrix
    if projection is None:
        projection = np.column_stack([calibration.k, np.zeros(3)])
    if not np.isfinite(projection).all() or np.linalg.matrix_rank(projection[:, :3]) != 3:
        raise ValueError("camera projection must be finite with an invertible intrinsic block.")
    matrix = np.asarray(projection, dtype=np.float64) @ calibration.t_vehicle_to_camera
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(calibration.t_vehicle_to_camera) != 4:
        raise ValueError("calibration must be finite and invertible.")
    endpoints = []
    positive = np.ones((height, width), dtype=bool)
    for z in grid.z_range_m:
        points = np.stack([xx, yy, np.full_like(xx, z), np.ones_like(xx)])
        pixels = (matrix @ points.reshape(4, -1)).reshape(3, height, width)
        camera_depth = (calibration.t_vehicle_to_camera[2] @ points.reshape(4, -1)).reshape(height, width)
        positive &= (camera_depth > 1e-6) & (pixels[2] > 1e-6)
        safe_depth = np.where(pixels[2] > 1e-6, pixels[2], 1.)
        endpoints.append(pixels[:2] / safe_depth)
    lo, hi = np.minimum(*endpoints), np.maximum(*endpoints)
    supervision = positive & (lo[0] >= 0) & (hi[0] < image_w) & (lo[1] >= 0) & (hi[1] < image_h)
    heatmap = np.zeros((height, width), dtype=np.float32)
    offsets = np.zeros((2, height, width), dtype=np.float32)
    center_mask = np.zeros((height, width), dtype=bool)
    accepted, ignore = [], []
    for label in labels:
        if label.category == "Car":
            center, corners = label_box_vehicle(label, calibration)
            inside = all(bounds[0] <= center[axis] < bounds[1] for axis, bounds in enumerate(
                (grid.x_range_m, grid.y_range_m, grid.z_range_m),
            ))
            if inside:
                coordinate = (center[:2] - np.array([grid.x_range_m[0], grid.y_range_m[0]], dtype=np.float32)) / np.float32(grid.resolution_m)
                row, col = np.floor(coordinate).astype(np.int64)
                if 0 <= row < height and 0 <= col < width:
                    accepted.append((center, corners, coordinate, row, col))
                    continue
        ignore.append(label)
    for label in ignore:
        left, top, right, bottom = label.bbox_xyxy
        intersects = (hi[0] >= left) & (lo[0] <= right) & (hi[1] >= top) & (lo[1] <= bottom)
        supervision &= ~intersects
    for center, corners, coordinate, row, col in accepted:
        if center_mask[row, col]:
            raise ValueError("two Car centers occupy the same BEV cell; offsets cannot represent both.")
        center_mask[row, col] = True
        offsets[:, row, col] = coordinate - [row, col]
        radius = gaussian_radius
        r0, r1 = max(0, row - radius), min(height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(width, col + radius + 1)
        dr, dc = np.meshgrid(np.arange(r0, r1) - row, np.arange(c0, c1) - col, indexing="ij")
        sigma = (2 * radius + 1) / 6
        gaussian = np.exp(-(dr**2 + dc**2) / (2 * sigma**2)).astype(np.float32)
        heatmap[r0:r1, c0:c1] = np.maximum(heatmap[r0:r1, c0:c1], gaussian)
    supervision |= center_mask
    return LidarCenterTargets(
        heatmap[None], offsets, center_mask[None], supervision[None],
        np.asarray([item[0] for item in accepted], dtype=np.float32).reshape(-1, 3),
        np.asarray([item[1] for item in accepted], dtype=np.float32).reshape(-1, 8, 3), len(ignore),
    )
