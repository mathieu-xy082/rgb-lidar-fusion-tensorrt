#!/usr/bin/env python3
"""Render RGB-LiDAR demo splatting NPZ artifacts as simple PPM images.

Usage:
  cd .
  pdm run python docs/onboarding/scripts/render_splatting_ppm.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

ROOT = Path("results/demo_artifacts")
SPARSE = ROOT / "sparse_maps" / "synthetic_sparse_maps.npz"
SPLAT = ROOT / "splatted_maps" / "synthetic_splatted_maps.npz"
OUT = ROOT / "visualizations"


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


def main() -> None:
    sparse = np.load(SPARSE)["lidar_maps"]
    splatted = np.load(SPLAT)

    sparse_depth = sparse[0]
    sparse_mask = sparse[5]
    depth_expanded = splatted["depth_expanded"]
    confidence = splatted["confidence"]

    outputs = {
        "sparse_depth.ppm": heatmap(sparse_depth),
        "sparse_mask.ppm": heatmap(sparse_mask),
        "splatted_depth.ppm": heatmap(depth_expanded),
        "splatted_confidence.ppm": heatmap(confidence),
    }
    for name, image in outputs.items():
        path = OUT / name
        write_ppm(path, image)
        print(path)

    print(f"sparse_pixels={int((sparse_mask > 0).sum())}")
    print(f"splatted_pixels={int((confidence > 0).sum())}")


if __name__ == "__main__":
    main()
