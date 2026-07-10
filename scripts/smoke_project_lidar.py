"""Smoke CLI for projecting synthetic LiDAR points.

This intentionally avoids dataset dependencies. Real KITTI loading will be added
in a later milestone.
"""

from __future__ import annotations

import numpy as np

from rgb_lidar_fusion.calibration import make_identity_calibration
from rgb_lidar_fusion.project_lidar import build_sparse_lidar_maps, project_lidar_to_image


def main() -> None:
    calibration = make_identity_calibration(fx=100, fy=100, cx=320, cy=180)
    points = np.array(
        [
            [0.0, 0.0, 10.0, 0.9],
            [1.0, 0.5, 20.0, 0.6],
            [-1.0, 0.2, 15.0, 0.7],
        ],
        dtype=np.float32,
    )
    projected = project_lidar_to_image(points, calibration, image_shape=(360, 640))
    maps = build_sparse_lidar_maps(projected, image_shape=(360, 640))
    print(f"projected_points={projected.pixels.shape[0]}")
    print(f"lidar_maps_shape={maps.shape}")
    print(f"occupied_pixels={int(maps[5].sum())}")


if __name__ == "__main__":
    main()
