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
from rgb_lidar_fusion.lidar_bev import LidarBEVContext, PillarFeatureNet, scatter_pillar_features
from rgb_lidar_fusion.lidar_pillars import build_lidar_pillars
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
    parser.add_argument("--with-lidar", action="store_true", help="Include random-weight LiDAR features on the same grid.")
    parser.add_argument("--lidar-seed", type=int, default=13)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("results/analysis/camera_bev")
        if args.with_lidar:
            args.output_dir = Path("results/analysis/camera_lidar_bev") / f"res_{args.resolution}m"
    return args


def _lidar_features(raw_points, grid, seed):
    pillars = build_lidar_pillars([raw_points], grid=grid, seed=seed)
    # Isolate CPU initialization without seeding or initializing CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        pfn, context = PillarFeatureNet().eval(), LidarBEVContext().eval()
    with torch.no_grad():
        scattered = scatter_pillar_features(
            pfn(torch.from_numpy(pillars.features), torch.from_numpy(pillars.point_mask)),
            indices=torch.from_numpy(pillars.indices), point_count=torch.from_numpy(pillars.point_count),
        )
        contextual = context(scattered)
    arrays = {
        "lidar_pillar_bev": scattered.features.numpy(),
        "lidar_context_bev": contextual.features.numpy(),
        "lidar_point_count": contextual.point_count.numpy(),
        "lidar_observed_mask": contextual.observed_mask.numpy(),
        "pillar_indices": pillars.indices,
        "retained_count": pillars.retained_count,
        "dropped_count": pillars.dropped_count,
    }
    summary = {
        "weights": "random", "seed": seed, "feature_shape": list(contextual.features.shape),
        "occupied_cells": len(pillars.indices), "pre_cap_points": int(pillars.point_count.sum()),
        "retained_points": int(pillars.retained_count.sum()), "dropped_points": int(pillars.dropped_count.sum()),
    }
    return arrays, summary


def _joint_figure(image, sample_id, grid, camera_bev, reference, raw_points, arrays):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    axes[0].imshow(np.moveaxis(image, 0, -1))
    axes[0].set_title(f"RGB {sample_id}")
    axes[0].axis("off")
    inside = np.ones(len(raw_points), dtype=bool)
    for axis, bounds in enumerate((grid.x_range_m, grid.y_range_m, grid.z_range_m)):
        inside &= (raw_points[:, axis] >= bounds[0]) & (raw_points[:, axis] < bounds[1])
    for ax, points, title in ((axes[1], raw_points[inside], "Raw LiDAR (full ROI)"),
                              (axes[2], reference, "Raw LiDAR (camera FOV)")):
        ax.scatter(points[:, 1], points[:, 0], c=points[:, 2], s=.5, cmap="viridis",
                   vmin=grid.z_range_m[0], vmax=grid.z_range_m[1])
        ax.set_title(title)
    extent = (*grid.y_range_m, *grid.x_range_m)
    camera_count = camera_bev.pixel_count[0, 0].numpy()
    lidar_count = arrays["lidar_point_count"][0, 0]
    overlap = arrays["coverage_overlap"][0, 0]
    panels = [
        (3, camera_count, "Camera pixel contributions", camera_count > 0),
        (4, lidar_count, "Raw LiDAR points (before cap)", lidar_count > 0),
        (6, camera_bev.features[0].square().mean(dim=0).sqrt().numpy(), "Camera feature RMS (not object score)", camera_count > 0),
    ]
    for index, key, title in ((7, "lidar_pillar_bev", "LiDAR PFN RMS (random weights)"),
                              (8, "lidar_context_bev", "LiDAR context RMS (random weights)")):
        rms = np.sqrt(np.square(arrays[key][0]).mean(axis=0))
        panels.append((index, rms, title, rms > 0))
    for index, values, title, mask in panels:
        artist = axes[index].imshow(np.ma.masked_where(~mask, values), origin="lower", extent=extent, interpolation="nearest")
        axes[index].set_title(title, fontsize=10)
        fig.colorbar(artist, ax=axes[index], shrink=.75)
    artist = axes[5].imshow(overlap, origin="lower", extent=extent, interpolation="nearest",
                            cmap=ListedColormap(["white", "#efab52", "#4aafc3", "#408a54"]),
                            norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4))
    axes[5].set_title("Support overlap (not an accuracy score)", fontsize=10)
    bar = fig.colorbar(artist, ax=axes[5], shrink=.75, ticks=[0, 1, 2, 3])
    bar.ax.set_yticklabels(["None", "Camera", "LiDAR", "Both"])
    for ax in axes[1:]:
        ax.set(xlabel="LiDAR y left (m)", ylabel="LiDAR x forward (m)",
               xlim=grid.y_range_m, ylim=grid.x_range_m, aspect="equal")
        ax.invert_xaxis()
    fig.tight_layout()
    return fig


def inspect(args: argparse.Namespace) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.sample_index < 0:
        raise ValueError("sample-index must be non-negative.")
    if args.lidar_seed < 0:
        raise ValueError("lidar-seed must be non-negative.")
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
    raw_points = load_velodyne_bin(item["meta"]["lidar_path"])
    reference = project_lidar_to_image(raw_points, calibration, tuple(item["meta"]["image_shape"])).vehicle_points
    inside = np.ones(len(reference), dtype=bool)
    for axis, bounds in enumerate((grid.x_range_m, grid.y_range_m, grid.z_range_m)):
        inside &= (reference[:, axis] >= bounds[0]) & (reference[:, axis] < bounds[1])
    reference = reference[inside]
    count = bev.pixel_count[0, 0].numpy()
    lidar_arrays, lidar_summary = {}, {}
    if args.with_lidar:
        lidar_arrays, lidar_summary = _lidar_features(raw_points, grid, args.lidar_seed)
        lidar_arrays["coverage_overlap"] = (
            (bev.pixel_count.numpy() > 0).astype(np.uint8)
            + 2 * lidar_arrays["lidar_observed_mask"].astype(np.uint8)
        )
        fig = _joint_figure(item["image"], item["meta"]["sample_id"], grid, bev, reference, raw_points, lidar_arrays)
    else:
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
            axis.invert_xaxis()
        for axis, artist in zip(axes[1], artists):
            fig.colorbar(artist, ax=axis, shrink=0.75)
        fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"camera_bev_{item['meta']['sample_id']}_{args.source}"
    if args.weight_by_splat_confidence:
        stem += "_weighted"
    if args.with_lidar:
        stem += f"_lidar_res{str(args.resolution).replace('.', 'p')}m"
    fig.savefig(args.output_dir / f"{stem}.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(
        args.output_dir / f"{stem}.npz", camera_bev=bev.features.numpy(),
        pixel_count=bev.pixel_count.numpy(), weight_sum=bev.weight_sum.numpy(),
        pixel_mask=bev.pixel_mask.numpy(), camera_projection=batch["camera_projection"],
        t_vehicle_to_camera=batch["t_vehicle_to_camera"],
        **lidar_arrays,
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
    if args.with_lidar:
        summary["lidar"] = lidar_summary
        overlap = lidar_arrays["coverage_overlap"]
        summary["coverage_overlap_cells"] = {name: int((overlap == code).sum()) for code, name in
                                             enumerate(("none", "camera_only", "lidar_only", "both"))}
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
