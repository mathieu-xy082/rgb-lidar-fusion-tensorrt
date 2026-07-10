"""Calibration helpers for camera/LiDAR projection.

The first prototype keeps the camera/LiDAR geometry explicit: synthetic tests can
use simple intrinsics, while KITTI calibration files are parsed into the same
calibration object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Camera intrinsics and vehicle-to-camera extrinsics.

    Attributes:
        k: 3x3 camera intrinsic matrix.
        t_camera_from_vehicle: 4x4 homogeneous transform from vehicle/LiDAR
            coordinates to camera coordinates.
        projection_matrix: optional 3x4 camera projection matrix. KITTI ``P2``
            includes a fourth translation column that must be preserved for
            exact rectified-camera projection.
    """

    k: np.ndarray
    t_camera_from_vehicle: np.ndarray
    projection_matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        k = np.asarray(self.k, dtype=np.float32)
        transform = np.asarray(self.t_camera_from_vehicle, dtype=np.float32)
        projection = (
            None
            if self.projection_matrix is None
            else np.asarray(self.projection_matrix, dtype=np.float32)
        )
        if k.shape != (3, 3):
            raise ValueError(f"Camera intrinsic matrix must have shape (3, 3), got {k.shape}.")
        if transform.shape != (4, 4):
            raise ValueError(
                f"Vehicle-to-camera transform must have shape (4, 4), got {transform.shape}."
            )
        if projection is not None and projection.shape != (3, 4):
            raise ValueError(
                f"Camera projection matrix must have shape (3, 4), got {projection.shape}."
            )
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "t_camera_from_vehicle", transform)
        object.__setattr__(self, "projection_matrix", projection)


def make_identity_calibration(
    fx: float = 100.0,
    fy: float = 100.0,
    cx: float = 0.0,
    cy: float = 0.0,
) -> CameraCalibration:
    """Return a simple calibration useful for tests and smoke examples."""

    return CameraCalibration(
        k=np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        t_camera_from_vehicle=np.eye(4, dtype=np.float32),
    )


def _parse_kitti_calibration_rows(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, values = line.split(":", maxsplit=1)
        rows[key] = np.fromstring(values, sep=" ", dtype=np.float32)
    return rows


def _require_row(rows: Mapping[str, np.ndarray], key: str, expected_size: int) -> np.ndarray:
    if key not in rows:
        raise ValueError(f"KITTI calibration file is missing required row {key!r}.")
    values = rows[key]
    if values.size != expected_size:
        raise ValueError(
            f"KITTI calibration row {key!r} must have {expected_size} floats, got {values.size}."
        )
    return values


def parse_kitti_calibration_file(path: str | Path) -> CameraCalibration:
    """Parse a KITTI object-detection calibration file for Velodyne projection.

    The returned calibration uses camera matrix ``P2[:3, :3]`` as intrinsics and
    composes ``R0_rect @ Tr_velo_to_cam`` into the homogeneous
    vehicle/LiDAR-to-rectified-camera transform expected by
    :func:`project_lidar_to_image`.
    """

    rows = _parse_kitti_calibration_rows(Path(path))
    p2 = _require_row(rows, "P2", 12).reshape(3, 4)
    r0_rect = _require_row(rows, "R0_rect", 9).reshape(3, 3)
    tr_velo_to_cam = _require_row(rows, "Tr_velo_to_cam", 12).reshape(3, 4)

    rect_4x4 = np.eye(4, dtype=np.float32)
    rect_4x4[:3, :3] = r0_rect

    velo_to_cam_4x4 = np.eye(4, dtype=np.float32)
    velo_to_cam_4x4[:3, :] = tr_velo_to_cam

    return CameraCalibration(
        k=p2[:3, :3],
        t_camera_from_vehicle=rect_4x4 @ velo_to_cam_4x4,
        projection_matrix=p2,
    )
