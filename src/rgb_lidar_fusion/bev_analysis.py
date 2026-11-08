"""CPU inspection of separate supports and features in the assembled BEV model."""

from dataclasses import dataclass

import numpy as np
import torch

from .bev_model import BEVFusionDiagnostic
from .lidar_center_model import decode_lidar_centers, lidar_center_loss
from .lidar_center_training import match_lidar_centers


@dataclass(frozen=True)
class BEVFusionSample:
    sample_id: str
    image: np.ndarray
    maps: dict[str, np.ndarray]
    centers_xy: np.ndarray
    decoded_centers: np.ndarray
    loss: dict
    matching: dict


def analyze_bev_fusion_frame(model: BEVFusionDiagnostic, frame, *, score_threshold=.3,
                             top_k=50, match_radius_m=1.) -> BEVFusionSample:
    """Inspect one prepared CPU frame, without updating parameters or BN buffers."""

    if next(model.parameters()).device.type != "cpu":
        raise ValueError("inspection requires a CPU model.")
    if any(value.device.type != "cpu" for value in (*frame.inputs.values(), *frame.targets.values())):
        raise ValueError("inspection requires CPU inputs and targets.")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            representations = model.encode(**frame.inputs)
            prediction = model.head(representations.shared_bev.features)
            loss = lidar_center_loss(prediction, **frame.targets)
            decoded = decode_lidar_centers(
                prediction, grid=model.config.grid, score_threshold=score_threshold,
                top_k=top_k, valid_mask=frame.targets["supervision_mask"],
            )[0].numpy().copy()
            maps = {
                "camera_count": representations.camera_bev.pixel_count[0, 0].numpy().copy(),
                "camera_mass": representations.camera_bev.weight_sum[0, 0].numpy().copy(),
                "lidar_count": representations.lidar_bev.point_count[0, 0].numpy().copy(),
                "supervision": frame.targets["supervision_mask"][0, 0].numpy().copy(),
                "target": frame.targets["heatmap"][0, 0].numpy().copy(),
                "scores": prediction.heatmap_logits[0, 0].sigmoid().numpy().copy(),
            }
            for name, bundle in (("camera_rms", representations.camera_bev),
                                 ("lidar_rms", representations.lidar_bev),
                                 ("fusion_rms", representations.fused_bev),
                                 ("shared_rms", representations.shared_bev)):
                maps[name] = bundle.features[0].square().mean(0).sqrt().numpy().copy()
            return BEVFusionSample(
                frame.sample_id, frame.inputs["rgb"][0].permute(1, 2, 0).numpy().copy(), maps,
                frame.centers_xy.copy(), decoded,
                {name: float(getattr(loss, name)) for name in ("total", "heatmap", "offset")},
                match_lidar_centers(decoded, frame.centers_xy, radius_m=match_radius_m),
            )
    finally:
        model.train(was_training)


def plot_bev_fusion_sample(sample: BEVFusionSample, *, grid, weights_label="fresh fusion/head"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    axes = axes.ravel()
    axes[0].imshow(np.clip(sample.image, 0, 1))
    axes[0].set_title(f"RGB {sample.sample_id}", fontsize=10)
    axes[0].axis("off")
    panels = (
        (1, "camera_count", "Camera contributing pixels (support > 0)", False),
        (2, "lidar_count", "Raw LiDAR count before cap (observed > 0)", False),
        (3, "supervision", "Supervision (white = ignored)", True),
        (4, "camera_rms", "Camera RMS (not object score)", False),
        (5, "lidar_rms", "LiDAR context RMS (not object score)", False),
        (6, "fusion_rms", "Fusion RMS (not object score)", False),
        (7, "shared_rms", "Shared context RMS (not object score)", False),
        (8, "camera_mass", "Camera splat weight sum (heuristic)", False),
        (9, "target", "Car-center target", True),
        (10, "scores", f"Car score ({weights_label})", True),
        (11, "lidar_count", "Decoded centers on LiDAR support", False),
    )
    extent = (*grid.y_range_m, *grid.x_range_m)
    for index, key, title, probability in panels:
        values = sample.maps[key]
        mask = ~sample.maps["supervision"] if probability else values == 0
        options = {"vmin": 0, "vmax": 1} if probability else {}
        artist = axes[index].imshow(np.ma.masked_where(mask, values), origin="lower", extent=extent,
                                    interpolation="nearest", **options)
        fig.colorbar(artist, ax=axes[index], shrink=.7)
        axes[index].set_title(title, fontsize=10)
        axes[index].set(xlabel="LiDAR y left (m)", ylabel="LiDAR x forward (m)",
                        xlim=grid.y_range_m, ylim=grid.x_range_m, aspect="equal")
        axes[index].invert_xaxis()
    for index in (9, 10, 11):
        centers = sample.centers_xy
        axes[index].scatter(centers[:, 1], centers[:, 0], facecolors="none", edgecolors="lime",
                            s=45, linewidths=1, label="GT Car center")
    for index in (10, 11):
        centers = sample.decoded_centers
        axes[index].scatter(centers[:, 1], centers[:, 0], marker="x", c="red", s=18,
                            label="Decoded center")
        axes[index].legend(loc="lower left", fontsize=7)
    fig.tight_layout(pad=1.5)
    return fig
