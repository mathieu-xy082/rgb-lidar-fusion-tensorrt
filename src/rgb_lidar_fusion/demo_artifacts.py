"""Deterministic demo artifact generation for RGB/LiDAR fusion examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from rgb_lidar_fusion.calibration import make_identity_calibration
from rgb_lidar_fusion.lidar_splatting import SplattingConfig, splat_sparse_depth
from rgb_lidar_fusion.project_lidar import (
    build_sparse_lidar_maps,
    project_lidar_to_image,
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


def _artifact_integrity(path: Path) -> dict[str, int | str]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def generate_synthetic_demo_artifacts(
    output_dir: str | Path = "results/demo_artifacts",
    *,
    image_shape: tuple[int, int] = (360, 640),
    splat_config: SplattingConfig | None = None,
) -> dict[str, Any]:
    """Generate small deterministic demo artifacts from synthetic LiDAR points.

    The output intentionally uses lightweight formats that can be regenerated
    locally and should remain ignored by Git: a PPM overlay, compressed sparse
    LiDAR maps, compressed splatted maps, and a JSON manifest.
    """

    output_root = Path(output_dir)
    overlay_dir = output_root / "projected_lidar_examples"
    sparse_dir = output_root / "sparse_maps"
    splat_dir = output_root / "splatted_maps"
    for directory in (overlay_dir, sparse_dir, splat_dir):
        directory.mkdir(parents=True, exist_ok=True)

    calibration = make_identity_calibration(fx=100, fy=100, cx=320, cy=180)
    projected = project_lidar_to_image(SYNTHETIC_POINTS, calibration, image_shape=image_shape)
    lidar_maps = build_sparse_lidar_maps(projected, image_shape=image_shape)
    splat = splat_sparse_depth(
        sparse_depth=lidar_maps[0],
        sparse_mask=lidar_maps[5] > 0.0,
        config=splat_config or SplattingConfig(radius_px=2, sigma_px=1.0),
    )

    image = np.zeros((*image_shape, 3), dtype=np.uint8)
    overlay = render_lidar_overlay(image, projected, max_depth_m=80.0, point_radius=2)

    overlay_path = overlay_dir / "synthetic_overlay.ppm"
    sparse_path = sparse_dir / "synthetic_sparse_maps.npz"
    splat_path = splat_dir / "synthetic_splatted_maps.npz"
    manifest_path = output_root / "manifest.json"

    write_ppm_image(overlay_path, overlay)
    np.savez_compressed(sparse_path, lidar_maps=lidar_maps)
    np.savez_compressed(
        splat_path,
        depth_expanded=splat.depth_expanded,
        confidence=splat.confidence,
        sparse_depth=splat.sparse_depth,
        sparse_mask=splat.sparse_mask,
    )

    manifest: dict[str, Any] = {
        "mode": "synthetic",
        "image_shape": [int(image_shape[0]), int(image_shape[1])],
        "projected_points": int(projected.pixels.shape[0]),
        "occupied_pixels": int(lidar_maps[5].sum()),
        "splatted_pixels": int((splat.confidence > 0.0).sum()),
        "outputs": {
            "overlay": overlay_path.relative_to(output_root).as_posix(),
            "sparse_maps": sparse_path.relative_to(output_root).as_posix(),
            "splatted_maps": splat_path.relative_to(output_root).as_posix(),
        },
        "artifact_integrity": {
            "overlay": _artifact_integrity(overlay_path),
            "sparse_maps": _artifact_integrity(sparse_path),
            "splatted_maps": _artifact_integrity(splat_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic demo artifacts.")
    parser.add_argument(
        "--output-dir",
        default="results/demo_artifacts",
        help="Output directory for ignored generated demo artifacts.",
    )
    args = parser.parse_args(argv)
    manifest = generate_synthetic_demo_artifacts(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
