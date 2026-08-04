#!/usr/bin/env python3
"""Render RGB-LiDAR splatting NPZ artifacts as simple PPM images.

Demo usage:
  pdm run demo-artifacts
  pdm run python docs/onboarding/scripts/render_splatting_ppm.py

KITTI usage:
  pdm run python docs/onboarding/scripts/render_splatting_ppm.py \
    --sparse-npz results/onboarding/kitti_000000/sparse_maps_000000.npz \
    --background-ppm results/onboarding/kitti_000000/image_000000.ppm \
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


def depth_to_color(depth: np.ndarray) -> np.ndarray:
    """Colorize positive depth values: near=red/yellow, far=cyan/blue."""
    v = normalize(depth).astype(np.float32) / 255.0
    # Invert so close depths are warm and far depths are cool.
    t = np.where(depth > 0, 1.0 - v, 0.0)
    rgb = np.zeros((*depth.shape, 3), dtype=np.float32)
    # Piecewise colormap: blue -> cyan -> green -> yellow -> red as depth gets closer.
    rgb[..., 0] = np.clip(1.5 * t - 0.25, 0.0, 1.0)
    rgb[..., 1] = np.clip(1.5 - np.abs(2.0 * t - 1.0) * 1.5, 0.0, 1.0)
    rgb[..., 2] = np.clip(1.25 - 1.5 * t, 0.0, 1.0)
    rgb[depth <= 0] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def confidence_to_gray(confidence: np.ndarray) -> np.ndarray:
    v = normalize(confidence)
    return np.repeat(v[..., np.newaxis], 3, axis=2)


def read_ppm(path: Path) -> np.ndarray:
    """Read simple P3 or P6 PPM files into uint8 RGB."""
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic not in {b"P3", b"P6"}:
            raise ValueError(f"{path} is not a P3/P6 PPM image")

        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} ended before PPM header was complete")
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
        width, height, maxval = map(int, tokens[:3])
        if maxval != 255:
            raise ValueError(f"only maxval=255 PPM files are supported, got {maxval}")
        if magic == b"P6":
            data = np.frombuffer(f.read(width * height * 3), dtype=np.uint8)
        else:
            rest = b" ".join(tokens[3:] + f.read().split())
            data = np.fromstring(rest.decode("ascii"), sep=" ", dtype=np.uint8)
        if data.size != width * height * 3:
            raise ValueError(f"{path} has {data.size} values, expected {width * height * 3}")
        return data.reshape((height, width, 3))


def blend_overlay(background: np.ndarray, color: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if background.shape != color.shape:
        raise ValueError(f"background shape {background.shape} != overlay shape {color.shape}")
    alpha_map = np.where(mask, alpha, 0.0).astype(np.float32)[..., np.newaxis]
    blended = background.astype(np.float32) * (1.0 - alpha_map) + color.astype(np.float32) * alpha_map
    return np.clip(blended, 0, 255).astype(np.uint8)


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


def load_splat_from_splatted_npz(splatted_npz: Path, output_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    from rgb_lidar_fusion.lidar_splatting import load_splatted_lidar_maps_npz

    loaded = load_splatted_lidar_maps_npz(splatted_npz)
    return loaded["sparse_lidar_maps"], loaded["splatted_lidar_maps"][6:8], output_dir


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
    parser.add_argument("--splatted-npz", type=Path, help="Precomputed stable splatted_maps_<id>.npz artifact")
    parser.add_argument("--output-dir", type=Path, help="Directory for visualization PPM files")
    parser.add_argument("--background-ppm", type=Path, help="Optional camera PPM used for transparent overlays")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity in [0, 1], default: 0.45")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")

    if args.splatted_npz:
        if not args.output_dir:
            parser.error("--output-dir is required with --splatted-npz")
        sparse, splatted, out = load_splat_from_splatted_npz(args.splatted_npz, args.output_dir)
    elif args.sparse_npz:
        if not args.output_dir:
            parser.error("--output-dir is required with --sparse-npz")
        sparse, splatted, out = compute_splat_from_sparse(args.sparse_npz, args.output_dir)
    else:
        sparse, splatted, out = load_splat_from_demo(DEFAULT_ROOT)

    sparse_depth = sparse[0]
    sparse_mask = sparse[5] > 0.0
    depth_expanded = splatted[0]
    confidence = splatted[1]
    splat_mask = confidence > 0.0

    sparse_color = depth_to_color(sparse_depth)
    splatted_color = depth_to_color(depth_expanded)

    outputs = {
        "sparse_depth.ppm": sparse_color,
        "sparse_mask.ppm": confidence_to_gray(sparse_mask.astype(np.float32)),
        "splatted_depth.ppm": splatted_color,
        "splatted_confidence.ppm": confidence_to_gray(confidence),
    }

    if args.background_ppm:
        background = read_ppm(args.background_ppm)
        outputs["sparse_depth_overlay.ppm"] = blend_overlay(
            background=background,
            color=sparse_color,
            mask=sparse_mask,
            alpha=args.alpha,
        )
        outputs["splatted_depth_overlay.ppm"] = blend_overlay(
            background=background,
            color=splatted_color,
            mask=splat_mask,
            alpha=args.alpha,
        )

    for name, image in outputs.items():
        path = out / name
        write_ppm(path, image)
        print(path)

    print(f"sparse_pixels={int(sparse_mask.sum())}")
    print(f"splatted_pixels={int(splat_mask.sum())}")
    if args.background_ppm:
        print(f"background={args.background_ppm}")
        print(f"alpha={args.alpha:.2f}")


if __name__ == "__main__":
    main()
