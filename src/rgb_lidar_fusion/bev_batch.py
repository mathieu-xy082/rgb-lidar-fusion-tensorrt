"""CPU-only assembly of existing dataset items and same-ID raw XYZI clouds."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from .bev_model import BEVFusionConfig
from .lidar_pillars import build_lidar_pillars
from .lidar_splatting import SplattingConfig
from .model_batch import dataset_items_to_model_batch, model_batch_to_baseline_inputs


@dataclass(frozen=True)
class BEVFusionBatch:
    sample_ids: tuple[str, ...]
    inputs: dict[str, torch.Tensor]


def prepare_bev_fusion_batch(
    items: Sequence[dict], raw_clouds: Mapping[str, np.ndarray], *, config: BEVFusionConfig,
) -> BEVFusionBatch:
    """Prepare CPU model kwargs without reading annotations or holdout targets.

    Items follow KittiObjectDepthDataset's resized RGB/six-channel sparse
    geometry contract. Raw clouds are keyed by sample ID, not blindly zipped.
    Callers must supply the matching scan/calibration and remove any actual
    held-out returns before building BOTH branches; this adapter adds no dropout.
    Device transfer is explicit and belongs to the future runner.
    """

    if not items or not isinstance(raw_clouds, Mapping):
        raise ValueError("non-empty items and raw_clouds keyed by sample ID are required.")
    ids = []
    clean_items = []
    for item in items:
        sample_id = item.get("meta", {}).get("sample_id")
        if not isinstance(sample_id, str) or len(sample_id) != 6 or not sample_id.isascii() or not sample_id.isdigit():
            raise ValueError("each item must have a six-digit sample_id.")
        if sample_id not in raw_clouds:
            raise ValueError(f"missing raw cloud for sample_id {sample_id}.")
        if item.get("meta", {}).get("lidar_representation", "sparse") != "sparse":
            raise ValueError("items must contain sparse maps; splatting is rebuilt from config.")
        if np.asarray(item["image"]).shape != (3, *config.image_shape):
            raise ValueError("item image shape differs from config.image_shape.")
        scale = np.asarray(item.get("max_depth_m", float("nan")), dtype=np.float32)
        if scale.shape != () or not np.isfinite(scale) or scale != np.float32(config.max_depth_m):
            raise ValueError("item max_depth_m must match the model normalization scale.")
        ids.append(sample_id)
        clean_items.append({key: item[key] for key in (
            "image", "lidar_maps", "camera_projection", "t_vehicle_to_camera", "max_depth_m", "meta",
        )})
    batch = dataset_items_to_model_batch(clean_items, splatting_config=SplattingConfig(
        radius_px=config.splat_radius_px, sigma_px=config.splat_sigma_px,
    ))
    camera = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
    pillars = build_lidar_pillars(
        [raw_clouds[sample_id] for sample_id in ids], grid=config.grid,
        max_points_per_pillar=config.max_points_per_pillar, seed=config.pillar_seed,
    )
    inputs = {name: torch.from_numpy(camera[name]) for name in ("rgb", "lidar_maps")}
    inputs.update({name: torch.from_numpy(batch[name])
                   for name in ("camera_projection", "t_vehicle_to_camera")})
    inputs.update({name: torch.from_numpy(getattr(pillars, name))
                   for name in ("point_mask", "indices", "point_count")})
    inputs["pillar_features"] = torch.from_numpy(pillars.features)
    return BEVFusionBatch(tuple(ids), inputs)
