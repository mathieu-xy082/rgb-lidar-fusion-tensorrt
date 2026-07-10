"""Project LiDAR points into image space and build sparse geometry maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import CameraCalibration


@dataclass(frozen=True)
class ProjectedLidar:
    """Projected LiDAR points that lie inside the image and in front of camera."""

    pixels: np.ndarray  # [N, 2], columns u, v
    camera_points: np.ndarray  # [N, 3]
    vehicle_points: np.ndarray  # [N, 3]
    intensity: np.ndarray  # [N]


def _as_lidar_array(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError("LiDAR points must have shape [N, 3] or [N, 4].")
    if arr.shape[1] == 3:
        intensity = np.ones((arr.shape[0], 1), dtype=np.float32)
        arr = np.concatenate([arr, intensity], axis=1)
    return arr


def load_velodyne_bin(path: str | Path) -> np.ndarray:
    """Load a KITTI Velodyne ``.bin`` file as ``[N, 4]`` float32 XYZI points."""

    values = np.fromfile(Path(path), dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(
            f"KITTI Velodyne file must contain a multiple of 4 float32 values, got {values.size}."
        )
    return values.reshape(-1, 4)


def project_lidar_to_image(
    points_vehicle: np.ndarray,
    calibration: CameraCalibration,
    image_shape: tuple[int, int],
) -> ProjectedLidar:
    """Project vehicle-frame LiDAR points into image pixels.

    Args:
        points_vehicle: LiDAR points as [N, 3] xyz or [N, 4] xyzi.
        calibration: camera intrinsics and vehicle-to-camera transform.
        image_shape: `(height, width)`.

    Returns:
        Only points with positive camera depth and pixel coordinates inside the
        image bounds.
    """

    height, width = image_shape
    points = _as_lidar_array(points_vehicle)
    xyz_vehicle = points[:, :3]
    intensity = points[:, 3]

    homogeneous = np.concatenate(
        [xyz_vehicle, np.ones((xyz_vehicle.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    camera_h = (calibration.t_camera_from_vehicle @ homogeneous.T).T
    camera_xyz = camera_h[:, :3]

    positive_depth = camera_xyz[:, 2] > 1e-6
    camera_xyz = camera_xyz[positive_depth]
    xyz_vehicle = xyz_vehicle[positive_depth]
    intensity = intensity[positive_depth]

    if camera_xyz.size == 0:
        return ProjectedLidar(
            pixels=np.empty((0, 2), dtype=np.float32),
            camera_points=np.empty((0, 3), dtype=np.float32),
            vehicle_points=np.empty((0, 3), dtype=np.float32),
            intensity=np.empty((0,), dtype=np.float32),
        )

    if calibration.projection_matrix is None:
        projected = (calibration.k @ camera_xyz.T).T
    else:
        projected = (calibration.projection_matrix @ camera_h[positive_depth].T).T
    pixels = projected[:, :2] / projected[:, 2:3]

    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    return ProjectedLidar(
        pixels=pixels[inside].astype(np.float32),
        camera_points=camera_xyz[inside].astype(np.float32),
        vehicle_points=xyz_vehicle[inside].astype(np.float32),
        intensity=intensity[inside].astype(np.float32),
    )


def build_sparse_lidar_maps(
    projected: ProjectedLidar,
    image_shape: tuple[int, int],
    max_depth_m: float = 80.0,
) -> np.ndarray:
    """Build sparse LiDAR feature maps aligned to the image grid.

    Output channels are:
        0. normalized camera depth z / max_depth_m
        1. vehicle x / max_depth_m
        2. vehicle y / max_depth_m
        3. vehicle z / max_depth_m
        4. intensity
        5. point-presence mask

    When multiple points land on the same pixel, keep the nearest depth.
    """

    height, width = image_shape
    maps = np.zeros((6, height, width), dtype=np.float32)

    if projected.pixels.shape[0] == 0:
        return maps

    uv = np.rint(projected.pixels).astype(np.int32)
    uv[:, 0] = np.clip(uv[:, 0], 0, width - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height - 1)

    depth = projected.camera_points[:, 2]
    order = np.argsort(depth)[::-1]  # far first, nearest overwrites

    for idx in order:
        u, v = uv[idx]
        maps[0, v, u] = depth[idx] / max_depth_m
        maps[1, v, u] = projected.vehicle_points[idx, 0] / max_depth_m
        maps[2, v, u] = projected.vehicle_points[idx, 1] / max_depth_m
        maps[3, v, u] = projected.vehicle_points[idx, 2] / max_depth_m
        maps[4, v, u] = projected.intensity[idx]
        maps[5, v, u] = 1.0

    return maps
