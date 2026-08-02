#!/usr/bin/env python3
"""Render RGB-LiDAR splatting NPZ artifacts as simple PPM images.

Demo usage:
  pdm run demo-artifacts
  pdm run python docs/onboarding/scripts/render_splatting_ppm.py

KITTI usage:
  pdm run python docs/onboarding/scripts/render_splatting_ppm.py \
    --sparse-npz results/onboarding/kitti_000000/sparse_maps_000000.npz \
    --output-dir results/onboarding/kitti_000000/visualizations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("results/demo_artifacts")


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    positive = arr[arr > 0]
    if positive.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo = float(positive.min())
    hi = float(positive.max())
    if hi <= lo:
        scaled = np.where(arr > 0, 255, 0)
    else:
        scaled = np.where(arr > 0, (arr - lo) / (hi - lo) * 255.0, 0)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def heatmap(values: np.ndarray) -> np.ndarray:
    v = normalize(values)
    rgb = np.zeros((*v.shape, 3), dtype=np.uint8)
    rgb[..., 0] = v
    rgb[..., 1] = np.clip(255 - np.abs(v.astype(np.int16) - 128) * 2, 0, 255).astype(np.uint8)
    rgb[..., 2] = 255 - v
    rgb[v == 0] = 0
    return rgb


def write_ppm(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = image.shape
    with path.open("w", encoding="ascii") as f:
        f.write(f"P3\n{w} {h}\n255\n")
        for row in image:
            f.write(" ".join(f"{r} {g} {b}" for r, g, b in row) + "\n")


def load_splat_from_demo(root: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    sparse_npz = root / "sparse_maps" / "synthetic_sparse_maps.npz"
    splat_npz = root / "splatted_maps" / "synthetic_splatted_maps.npz"
    sparse = np.load(sparse_npz)["lidar_maps"]
    splatted = np.load(splat_npz)
    return sparse, np.stack([splatted["depth_expanded"], splatted["confidence"]]), root / "visualizations"


def compute_splat_from_sparse(sparse_npz: Path, output_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    from rgb_lidar_fusion.lidar_splatting import SplattingConfig, splat_sparse_depth

    sparse = np.load(sparse_npz)["lidar_maps"]
    splat = splat_sparse_depth(
        sparse_depth=sparse[0],
        sparse_mask=sparse[5] > 0.0,
        config=SplattingConfig(radius_px=2, sigma_px=1.0),
    )
    return sparse, np.stack([splat.depth_expanded, splat.confidence]), output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-npz", type=Path, help="Sparse LiDAR NPZ containing lidar_maps")
    parser.add_argument("--output-dir", type=Path, help="Directory for visualization PPM files")
    args = parser.parse_args()

    if args.sparse_npz:
        if not args.output_dir:
            parser.error("--output-dir is required with --sparse-npz")
        sparse, splatted, out = compute_splat_from_sparse(args.sparse_npz, args.output_dir)
    else:
        sparse, splatted, out = load_splat_from_demo(DEFAULT_ROOT)

    sparse_depth = sparse[0]
    sparse_mask = sparse[5]
    depth_expanded = splatted[0]
    confidence = splatted[1]

    outputs = {
        "sparse_depth.ppm": heatmap(sparse_depth),
        "sparse_mask.ppm": heatmap(sparse_mask),
        "splatted_depth.ppm": heatmap(depth_expanded),
        "splatted_confidence.ppm": heatmap(confidence),
    }
    for name, image in outputs.items():
        path = out / name
        write_ppm(path, image)
        print(path)

    print(f"sparse_pixels={int((sparse_mask > 0).sum())}")
    print(f"splatted_pixels={int((confidence > 0).sum())}")


if __name__ == "__main__":
    main()
