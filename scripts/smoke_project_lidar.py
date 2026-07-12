"""Smoke CLI for projecting synthetic LiDAR points.

This intentionally avoids dataset dependencies. It can also write a small ASCII
PPM overlay for dependency-free visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rgb_lidar_fusion.calibration import make_identity_calibration, parse_kitti_calibration_file
from rgb_lidar_fusion.lidar_splatting import (
    SplattingConfig,
    splat_projected_depth,
    splat_sparse_depth,
)
from rgb_lidar_fusion.project_lidar import (
    build_sparse_lidar_maps,
    load_velodyne_bin,
    project_lidar_to_image,
    read_ppm_image,
    render_lidar_overlay,
    write_ppm_image,
)


SYNTHETIC_POINTS = np.array(
    [
        [0.0, 0.0, 10.0, 0.9],
        [1.0, 0.5, 20.0, 0.6],
        [-1.0, 0.2, 15.0, 0.7],
    ],
    dtype=np.float32,
)

SYNTHETIC_KITTI_CALIBRATION = "\n".join(
    [
        "P2: 100.0 0.0 320.0 0.0 0.0 100.0 180.0 0.0 0.0 0.0 1.0 0.0",
        "R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0",
        "Tr_velo_to_cam: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0",
    ]
)


def _write_synthetic_kitti_sample(root: str) -> tuple[Path, Path]:
    sample_root = Path(root)
    calib_file = sample_root / "training" / "calib" / "000000.txt"
    velodyne_file = sample_root / "training" / "velodyne" / "000000.bin"
    calib_file.parent.mkdir(parents=True, exist_ok=True)
    velodyne_file.parent.mkdir(parents=True, exist_ok=True)
    calib_file.write_text(SYNTHETIC_KITTI_CALIBRATION + "\n", encoding="utf-8")
    SYNTHETIC_POINTS.tofile(velodyne_file)
    return calib_file, velodyne_file


def _parse_image_size(value: str) -> tuple[int, int]:
    """Parse CLI image size as HEIGHTxWIDTH."""

    try:
        height_text, width_text = value.lower().split("x", maxsplit=1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("image size must use HEIGHTxWIDTH, e.g. 375x1242") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("image size dimensions must be positive")
    return height, width


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-output",
        help="Optional path for an ASCII PPM LiDAR overlay generated from synthetic points.",
    )
    parser.add_argument(
        "--sparse-output",
        help="Optional path for a compressed .npz file containing the sparse lidar_maps array.",
    )
    parser.add_argument(
        "--image-file",
        help="Optional ASCII PPM (P3) RGB canvas to draw the overlay on instead of a black image.",
    )
    parser.add_argument("--calib-file", help="Optional KITTI calibration .txt file to project.")
    parser.add_argument("--velodyne-file", help="Optional KITTI Velodyne .bin file to project.")
    parser.add_argument(
        "--write-synthetic-sample",
        help="Optional output root for a tiny KITTI-format synthetic calib/velodyne sample.",
    )
    parser.add_argument(
        "--image-size",
        type=_parse_image_size,
        default=None,
        help="Image canvas size as HEIGHTxWIDTH (default: 360x640, or inferred from --image-file).",
    )
    args = parser.parse_args(argv)

    image = None
    if args.image_file:
        image = read_ppm_image(args.image_file)
        image_shape = image.shape[:2] if args.image_size is None else args.image_size
        if image.shape[:2] != image_shape:
            parser.error(
                "--image-file dimensions must match --image-size "
                f"({image.shape[0]}x{image.shape[1]} != {image_shape[0]}x{image_shape[1]})"
            )
    else:
        image_shape = (360, 640) if args.image_size is None else args.image_size
    print(f"image_shape={image_shape[0]}x{image_shape[1]}")

    if bool(args.calib_file) != bool(args.velodyne_file):
        parser.error("--calib-file and --velodyne-file must be provided together")
    if args.calib_file:
        calibration = parse_kitti_calibration_file(args.calib_file)
        points = load_velodyne_bin(args.velodyne_file)
        print("mode=kitti")
        print(f"loaded_points={points.shape[0]}")
    else:
        calibration = make_identity_calibration(fx=100, fy=100, cx=320, cy=180)
        points = SYNTHETIC_POINTS
        print("mode=synthetic")

    projected = project_lidar_to_image(points, calibration, image_shape=image_shape)
    maps = build_sparse_lidar_maps(projected, image_shape=image_shape)
    splat = splat_sparse_depth(
        sparse_depth=maps[0],
        sparse_mask=maps[5] > 0.0,
        config=SplattingConfig(radius_px=2, sigma_px=1.0),
    )
    projected_splat = splat_projected_depth(
        pixels=projected.pixels,
        depths=projected.camera_points[:, 2],
        image_shape=image_shape,
        config=SplattingConfig(radius_px=2, sigma_px=1.0),
    )

    print(f"projected_points={projected.pixels.shape[0]}")
    print(f"lidar_maps_shape={maps.shape}")
    print(f"occupied_pixels={int(maps[5].sum())}")
    print(f"splatted_pixels={int((splat.confidence > 0.0).sum())}")
    print(f"projected_splatted_pixels={int((projected_splat.confidence > 0.0).sum())}")
    print(f"splat_max_confidence={float(splat.confidence.max()):.3f}")

    if args.sparse_output:
        np.savez_compressed(args.sparse_output, lidar_maps=maps)
        print(f"sparse_output={args.sparse_output}")
    if args.overlay_output:
        if image is not None:
            print("image_source=ppm")
        else:
            image = np.zeros((*image_shape, 3), dtype=np.uint8)
            print("image_source=blank")
        overlay = render_lidar_overlay(image, projected, max_depth_m=80.0, point_radius=2)
        write_ppm_image(args.overlay_output, overlay)
        print(f"overlay_output={args.overlay_output}")
    if args.write_synthetic_sample:
        calib_file, velodyne_file = _write_synthetic_kitti_sample(args.write_synthetic_sample)
        print(f"synthetic_sample={args.write_synthetic_sample}")
        print(f"synthetic_calib={calib_file}")
        print(f"synthetic_velodyne={velodyne_file}")


if __name__ == "__main__":
    main()
