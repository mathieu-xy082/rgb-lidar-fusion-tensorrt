"""NumPy-first adapter from dataset items to model-ready batches."""

from __future__ import annotations

from typing import Any

import numpy as np

from .lidar_splatting import SplattingConfig, splat_sparse_depth
from .project_lidar import LIDAR_MAP_CHANNELS

RGB_CHANNELS = ("rgb_red", "rgb_green", "rgb_blue")
SPLATTED_CHANNELS = ("depth_expanded", "confidence")


def dataset_item_to_model_batch(
    item: dict[str, Any],
    *,
    include_splatted_depth: bool = False,
    splatting_config: SplattingConfig | None = None,
) -> dict[str, Any]:
    """Convert one ``KittiSparseLidarDataset`` item into a batch dictionary.

    The model input contract is channel-first NumPy data with an explicit batch
    dimension: RGB channels first, followed by the dataset's sparse LiDAR maps.
    """

    image = np.asarray(item["image"], dtype=np.float32)
    lidar_maps = np.asarray(item["lidar_maps"], dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W].")
    if lidar_maps.ndim != 3 or lidar_maps.shape[0] != len(LIDAR_MAP_CHANNELS):
        raise ValueError(
            f"lidar_maps must have shape [{len(LIDAR_MAP_CHANNELS)}, H, W]."
        )
    if image.shape[1:] != lidar_maps.shape[1:]:
        raise ValueError("image and lidar_maps must share height and width.")
    meta = item.get("meta", {})
    if (
        "lidar_map_channels" in meta
        and tuple(meta["lidar_map_channels"]) != LIDAR_MAP_CHANNELS
    ):
        raise ValueError("lidar_map_channels metadata must match LIDAR_MAP_CHANNELS.")
    channels = [image, lidar_maps]
    input_channels = [*RGB_CHANNELS, *LIDAR_MAP_CHANNELS]
    if include_splatted_depth:
        splat = splat_sparse_depth(
            sparse_depth=lidar_maps[0],
            sparse_mask=lidar_maps[5] > 0.0,
            config=splatting_config,
        )
        channels.append(
            np.stack([splat.depth_expanded, splat.confidence]).astype(
                np.float32,
                copy=False,
            )
        )
        input_channels.extend(SPLATTED_CHANNELS)

    inputs = np.concatenate(channels, axis=0)[np.newaxis, ...]
    return {
        "inputs": inputs.astype(np.float32, copy=False),
        "input_channels": input_channels,
        "sparse_lidar_maps": lidar_maps[np.newaxis, ...].copy(),
        "target": item.get("target"),
        "meta": item.get("meta", {}),
    }
