"""CPU frame inspection for the optional, initially untrained LiDAR center head."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .calibration import parse_kitti_calibration_file
from .camera_bev import BEVGridConfig
from .kitti_labels import KittiLabel, load_kitti_labels
from .lidar_center_model import LidarCenterDiagnostic, decode_lidar_centers, lidar_center_loss
from .lidar_center_targets import LidarCenterTargets, build_lidar_center_targets
from .lidar_pillars import build_lidar_pillars
from .project_lidar import load_velodyne_bin


@dataclass(frozen=True)
class LidarBEVSample:
    sample_id: str
    grid: BEVGridConfig
    image: np.ndarray
    raw_points: np.ndarray
    labels: list[KittiLabel] | None
    targets: LidarCenterTargets | None
    point_count: np.ndarray
    pillar_rms: np.ndarray
    context_rms: np.ndarray
    center_scores: np.ndarray
    decoded_centers: np.ndarray
    loss: dict[str, float] | None
    retained_points: int
    dropped_points: int


def analyze_lidar_bev_sample(
    root: str | Path, sample_id: str, *, grid: BEVGridConfig,
    model: LidarCenterDiagnostic, score_threshold: float = .3,
    max_points_per_pillar: int = 64, pillar_seed: int = 13, top_k: int = 50,
) -> LidarBEVSample:
    """Inspect native-resolution RGB/calibration and full-ROI raw LiDAR on CPU.

    Missing labels permit geometry inspection but produce no target/loss. They
    are not interpreted as an empty, fully annotated negative frame. Inference
    leaves parameters and BN buffers unchanged; training mode is restored.
    Supplying targets never changes the encoder input. Decoding, when labels
    exist, is restricted to the diagnostic evaluation domain only.
    """

    from PIL import Image

    if not isinstance(sample_id, str) or len(sample_id) != 6 or not sample_id.isascii() or not sample_id.isdigit():
        raise ValueError("sample_id must be a six-digit KITTI ID.")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("frame inspection uses a CPU model; GPU probing belongs in a separate process.")
    root = Path(root)
    with Image.open(root / "image_2" / f"{sample_id}.png") as rgb:
        image = np.asarray(rgb.convert("RGB")).copy()
    calibration = parse_kitti_calibration_file(root / "calib" / f"{sample_id}.txt")
    raw = load_velodyne_bin(root / "velodyne" / f"{sample_id}.bin")
    label_path = root / "label_2" / f"{sample_id}.txt"
    labels = load_kitti_labels(label_path) if label_path.is_file() else None
    target = None if labels is None else build_lidar_center_targets(
        labels, calibration=calibration, image_shape=image.shape[:2], grid=grid,
    )
    pillars = build_lidar_pillars(
        [raw], grid=grid, max_points_per_pillar=max_points_per_pillar, seed=pillar_seed,
    )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            scattered, contextual = model.encode(
                torch.from_numpy(pillars.features), torch.from_numpy(pillars.point_mask),
                torch.from_numpy(pillars.indices), torch.from_numpy(pillars.point_count),
            )
            prediction = model.head(contextual.features)
            valid_mask = None if target is None else torch.from_numpy(target.supervision_mask[None])
            decoded = decode_lidar_centers(
                prediction, grid=grid, score_threshold=score_threshold, top_k=top_k, valid_mask=valid_mask,
            )[0]
            loss = None
            if target is not None:
                components = lidar_center_loss(
                    prediction, heatmap=torch.from_numpy(target.heatmap[None]),
                    offset_xy=torch.from_numpy(target.offset_xy[None]),
                    center_mask=torch.from_numpy(target.center_mask[None]), supervision_mask=valid_mask,
                )
                loss = {name: float(getattr(components, name)) for name in ("total", "heatmap", "offset")}
            return LidarBEVSample(
                sample_id, grid, image, raw, labels, target, pillars.point_count[0, 0].copy(),
                scattered.features[0].square().mean(dim=0).sqrt().numpy(),
                contextual.features[0].square().mean(dim=0).sqrt().numpy(),
                prediction.heatmap_logits[0, 0].sigmoid().numpy(), decoded.numpy(), loss,
                int(pillars.retained_count.sum()), int(pillars.dropped_count.sum()),
            )
    finally:
        model.train(was_training)


def plot_lidar_bev_sample(sample: LidarBEVSample, *, weights_label: str = "random, untrained"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.ravel()
    axes[0].imshow(sample.image)
    axes[0].set_title(f"RGB + KITTI labels {sample.sample_id}", fontsize=10)
    axes[0].axis("off")
    for label in sample.labels or []:
        left, top, right, bottom = label.bbox_xyxy
        axes[0].add_patch(Rectangle((left, top), right - left, bottom - top, fill=False,
                                     edgecolor="lime" if label.category == "Car" else "gray", linewidth=1))
    inside = np.ones(len(sample.raw_points), dtype=bool)
    for axis, bounds in enumerate((sample.grid.x_range_m, sample.grid.y_range_m, sample.grid.z_range_m)):
        inside &= (sample.raw_points[:, axis] >= bounds[0]) & (sample.raw_points[:, axis] < bounds[1])
    points = sample.raw_points[inside]
    axes[1].scatter(points[:, 1], points[:, 0], c=points[:, 2], cmap="viridis", s=.5)
    axes[1].set_title("Native ROI LiDAR + Car boxes", fontsize=10)
    extent = (*sample.grid.y_range_m, *sample.grid.x_range_m)
    panels = [(2, sample.point_count, "Raw points before cap", None),
              (4, sample.pillar_rms, "PFN feature RMS (not object score)", None),
              (5, sample.context_rms, "Context RMS (not object score)", None)]
    if sample.targets is not None:
        target = sample.targets
        for corners in target.corners_vehicle:
            footprint = corners[[0, 1, 2, 3, 0]]
            axes[1].plot(footprint[:, 1], footprint[:, 0], color="lime", linewidth=.8)
        panels.extend([(3, target.supervision_mask[0], "Supervision domain (white = ignored)", (0, 1)),
                       (6, np.ma.masked_where(~target.supervision_mask[0], target.heatmap[0]), "Car-center target (ignored cells masked)", (0, 1))])
        predicted = np.ma.masked_where(~target.supervision_mask[0], sample.center_scores)
    else:
        for index in (3, 6):
            axes[index].set_title("Labels missing: no supervision", fontsize=10)
        predicted = sample.center_scores
    panels.append((7, predicted, f"Car score ({weights_label})", (0, 1)))
    for index, values, title, limits in panels:
        if limits is None:
            values = np.ma.masked_where(values == 0, values)
        options = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
        if index == 3:
            values = np.ma.masked_where(values == 0, values)
        artist = axes[index].imshow(values, origin="lower", extent=extent, interpolation="nearest", **options)
        axes[index].set_title(title, fontsize=10)
        fig.colorbar(artist, ax=axes[index], shrink=.7)
    if sample.targets is not None and len(sample.targets.centers_vehicle):
        centers = sample.targets.centers_vehicle
        for index in (6, 7):
            axes[index].scatter(centers[:, 1], centers[:, 0], facecolors="none", edgecolors="lime",
                                 s=50, linewidths=1, label="GT Car center")
    if len(sample.decoded_centers):
        axes[7].scatter(sample.decoded_centers[:, 1], sample.decoded_centers[:, 0], marker="x", c="red", s=12,
                         label="Decoded center")
    if axes[7].get_legend_handles_labels()[0]:
        axes[7].legend(loc="lower left", fontsize=7)
    for ax in axes[1:]:
        ax.set(xlabel="LiDAR y left (m)", ylabel="LiDAR x forward (m)",
               xlim=sample.grid.y_range_m, ylim=sample.grid.x_range_m, aspect="equal")
        ax.invert_xaxis()
    fig.tight_layout(pad=1.5)
    return fig
