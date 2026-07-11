"""Smoke CLI for projecting synthetic LiDAR points.

This intentionally avoids dataset dependencies. It can also write a small ASCII
PPM overlay for dependency-free visual inspection.
"""

from __future__ import annotations

import argparse

import numpy as np

from rgb_lidar_fusion.calibration import make_identity_calibration
from rgb_lidar_fusion.project_lidar import (
    build_sparse_lidar_maps,
    project_lidar_to_image,
    render_lidar_overlay,
    write_ppm_image,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-output",
        help="Optional path for an ASCII PPM LiDAR overlay generated from synthetic points.",
    )
    args = parser.parse_args(argv)

    image_shape = (360, 640)
    calibration = make_identity_calibration(fx=100, fy=100, cx=320, cy=180)
    points = np.array(
        [
            [0.0, 0.0, 10.0, 0.9],
            [1.0, 0.5, 20.0, 0.6],
            [-1.0, 0.2, 15.0, 0.7],
        ],
        dtype=np.float32,
    )
    projected = project_lidar_to_image(points, calibration, image_shape=image_shape)
    maps = build_sparse_lidar_maps(projected, image_shape=image_shape)
    print(f"projected_points={projected.pixels.shape[0]}")
    print(f"lidar_maps_shape={maps.shape}")
    print(f"occupied_pixels={int(maps[5].sum())}")
    if args.overlay_output:
        image = np.zeros((*image_shape, 3), dtype=np.uint8)
        overlay = render_lidar_overlay(image, projected, max_depth_m=80.0, point_radius=2)
        write_ppm_image(args.overlay_output, overlay)
        print(f"overlay_output={args.overlay_output}")


if __name__ == "__main__":
    main()
