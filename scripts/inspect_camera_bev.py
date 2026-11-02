"""Inspect one camera-feature BEV on CPU against camera-FOV raw KITTI LiDAR."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from rgb_lidar_fusion.calibration import CameraCalibration
from rgb_lidar_fusion.camera_bev import BEVGridConfig, lift_camera_features_to_bev
from rgb_lidar_fusion.camera_depth_analysis import load_camera_depth_checkpoint
from rgb_lidar_fusion.camera_depth_support import prepare_camera_depth_support
from rgb_lidar_fusion.dataset import KittiObjectDepthDataset
from rgb_lidar_fusion.lidar_splatting import SplattingConfig
from rgb_lidar_fusion.model_batch import dataset_items_to_model_batch, model_batch_to_baseline_inputs
from rgb_lidar_fusion.project_lidar import load_velodyne_bin, project_lidar_to_image
from rgb_lidar_fusion.training import select_camera_depth_lidar_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/kitti/training"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--source", choices=("sparse", "splatted"), default="sparse")
    parser.add_argument("--x-range", type=float, nargs=2, default=(0., 50.))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-25., 25.))
    parser.add_argument("--z-range", type=float, nargs=2, default=(-3., 3.))
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--max-valid-depth", type=float, default=80.)
    parser.add_argument("--min-splat-confidence", type=float, default=0.)
    parser.add_argument("--weight-by-splat-confidence", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/analysis/camera_bev"))
    return parser.parse_args()


def inspect(args: argparse.Namespace) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative.")
    loaded = load_camera_depth_checkpoint(args.checkpoint, device="cpu")
    if not {"height", "width", "max_depth_m"}.issubset(loaded.config):
        raise ValueError("checkpoint config must specify height, width, and max_depth_m.")
    grid = BEVGridConfig(tuple(args.x_range), tuple(args.y_range), tuple(args.z_range), args.resolution)
    dataset = KittiObjectDepthDataset(
        args.data_root, image_shape=(int(loaded.config["height"]), int(loaded.config["width"])),
        max_depth_m=float(loaded.config["max_depth_m"]),
    )
    item = dataset[args.sample_index]
    # Inference-style inspection: all measured inputs, no hidden targets,
    # validation scores, or learned-depth reliability claims.
    batch = dataset_items_to_model_batch([item], splatting_config=SplattingConfig(
        radius_px=int(loaded.config.get("splat_radius_px", 2)),
        sigma_px=float(loaded.config.get("splat_sigma_px", 1.)),
    ))
    arrays = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
    with torch.no_grad():
        image_features = loaded.model.forward_features(
            torch.from_numpy(arrays["rgb"]),
            torch.from_numpy(select_camera_depth_lidar_maps(arrays["lidar_maps"], lidar_mode=loaded.model.lidar_mode)),
        )
        support = prepare_camera_depth_support(
            torch.from_numpy(arrays["lidar_maps"]), source=args.source,
            max_depth_m=loaded.config["max_depth_m"], max_valid_depth_m=args.max_valid_depth,
            min_splat_confidence=args.min_splat_confidence,
        )
        bev = lift_camera_features_to_bev(
            image_features, depth_support=support,
            camera_projection=torch.from_numpy(batch["camera_projection"]),
            t_vehicle_to_camera=torch.from_numpy(batch["t_vehicle_to_camera"]),
            grid=grid, weight_by_splat_confidence=args.weight_by_splat_confidence,
        )

    calibration = CameraCalibration(
        k=item["camera_projection"][:, :3], projection_matrix=item["camera_projection"],
        t_vehicle_to_camera=item["t_vehicle_to_camera"],
    )
    reference = project_lidar_to_image(
        load_velodyne_bin(item["meta"]["lidar_path"]), calibration, tuple(item["meta"]["image_shape"]),
    ).vehicle_points
    inside = np.ones(len(reference), dtype=bool)
    for axis, bounds in enumerate((grid.x_range_m, grid.y_range_m, grid.z_range_m)):
        inside &= (reference[:, axis] >= bounds[0]) & (reference[:, axis] < bounds[1])
    reference = reference[inside]
    count = bev.pixel_count[0, 0].numpy()
    rms = bev.features[0].square().mean(dim=0).sqrt().numpy()
    extent = (*grid.y_range_m, *grid.x_range_m)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(np.moveaxis(item["image"], 0, -1))
    axes[0, 0].set(title=f"RGB {item['meta']['sample_id']}")
    axes[0, 0].axis("off")
    axes[0, 1].scatter(reference[:, 1], reference[:, 0], c=reference[:, 2], s=1., cmap="viridis")
    axes[0, 1].set(title="Raw LiDAR reference (camera FOV)")
    artists = [
        axes[1, 0].imshow(np.ma.masked_where(count == 0, count), origin="lower", extent=extent, interpolation="nearest"),
        axes[1, 1].imshow(np.ma.masked_where(count == 0, rms), origin="lower", extent=extent, interpolation="nearest"),
    ]
    axes[1, 0].set(title=f"Contributing pixels per cell ({args.source})")
    axes[1, 1].set(title="Feature RMS (not a semantic/object score)")
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.set(xlabel="LiDAR y left (m)", ylabel="LiDAR x forward (m)",
                 xlim=grid.y_range_m, ylim=grid.x_range_m, aspect="equal")
        axis.invert_xaxis()  # Positive vehicle-left is displayed on the left.
    for axis, artist in zip(axes[1], artists):
        fig.colorbar(artist, ax=axis, shrink=0.75)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"camera_bev_{item['meta']['sample_id']}_{args.source}"
    if args.weight_by_splat_confidence:
        stem += "_weighted"
    fig.savefig(args.output_dir / f"{stem}.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(
        args.output_dir / f"{stem}.npz", camera_bev=bev.features.numpy(),
        pixel_count=bev.pixel_count.numpy(), weight_sum=bev.weight_sum.numpy(),
        pixel_mask=bev.pixel_mask.numpy(), camera_projection=batch["camera_projection"],
        t_vehicle_to_camera=batch["t_vehicle_to_camera"],
    )
    summary = {
        "sample_id": item["meta"]["sample_id"], "dataset_index": args.sample_index,
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": loaded.epoch,
        "checkpoint_step": loaded.step, "device": "cpu", "holdout": False,
        "source": args.source, "weight_by_splat_confidence": args.weight_by_splat_confidence,
        "max_depth_m": loaded.config["max_depth_m"], "max_valid_depth_m": args.max_valid_depth,
        "min_splat_confidence": args.min_splat_confidence,
        "grid": asdict(grid), "feature_shape": list(bev.features.shape),
        "accepted_pixels": int(bev.pixel_count.sum()), "occupied_cells": int((count > 0).sum()),
        "reference_points_in_roi": len(reference),
        "png": str(args.output_dir / f"{stem}.png"),
        "npz": str(args.output_dir / f"{stem}.npz"),
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    # Bound CPU work even when the host exposes many logical processors.
    torch.set_num_threads(2)
    try:
        summary = inspect(parse_args())
    except (FileNotFoundError, IndexError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
