"""Calibration helpers for camera/LiDAR projection.

The first prototype focuses on explicit matrices instead of hiding geometry in a
large framework. KITTI parsing will be added once sample data is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Camera intrinsics and vehicle-to-camera extrinsics.

    Attributes:
        k: 3x3 camera intrinsic matrix.
        t_camera_from_vehicle: 4x4 homogeneous transform from vehicle/LiDAR
            coordinates to camera coordinates.
    """

    k: np.ndarray
    t_camera_from_vehicle: np.ndarray

    def __post_init__(self) -> None:
        k = np.asarray(self.k, dtype=np.float32)
        transform = np.asarray(self.t_camera_from_vehicle, dtype=np.float32)
        if k.shape != (3, 3):
            raise ValueError(f"Camera intrinsic matrix must have shape (3, 3), got {k.shape}.")
        if transform.shape != (4, 4):
            raise ValueError(
                f"Vehicle-to-camera transform must have shape (4, 4), got {transform.shape}."
            )
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "t_camera_from_vehicle", transform)


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
