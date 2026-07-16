"""NumPy-first adapter from dataset items to model-ready batches."""

from __future__ import annotations

from typing import Any

import numpy as np

from .lidar_splatting import SplattingConfig, splat_sparse_depth
from .project_lidar import ENRICHED_LIDAR_MAP_CHANNELS, LIDAR_MAP_CHANNELS

RGB_CHANNELS = ("rgb_red", "rgb_green", "rgb_blue")
SPLATTED_CHANNELS = ("depth_expanded", "confidence")
BASELINE_LIDAR_CHANNELS_BY_MODE = {
    "sparse": LIDAR_MAP_CHANNELS,
    "enriched": ENRICHED_LIDAR_MAP_CHANNELS,
}


def dataset_items_to_model_batch(
    items: list[dict[str, Any]],
    *,
    include_splatted_depth: bool = False,
    splatting_config: SplattingConfig | None = None,
) -> dict[str, Any]:
    """Stack multiple dataset items into one model-ready NumPy batch."""

    batches = [
        dataset_item_to_model_batch(
            item,
            include_splatted_depth=include_splatted_depth,
            splatting_config=splatting_config,
        )
        for item in items
    ]
    if not batches:
        raise ValueError("items must contain at least one dataset item.")
    input_channels = batches[0]["input_channels"]
    if any(batch["input_channels"] != input_channels for batch in batches):
        raise ValueError("all dataset items must produce the same input channel order.")
    input_shape = batches[0]["inputs"].shape[1:]
    if any(batch["inputs"].shape[1:] != input_shape for batch in batches):
        raise ValueError("all dataset items must share batch input shape.")
    return {
        "inputs": np.concatenate([batch["inputs"] for batch in batches], axis=0),
        "input_channels": input_channels,
        "sparse_lidar_maps": np.concatenate(
            [batch["sparse_lidar_maps"] for batch in batches],
            axis=0,
        ),
        "targets": [batch["target"] for batch in batches],
        "metas": [batch["meta"] for batch in batches],
    }


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


def _baseline_channel_indices(
    input_channels: list[str] | tuple[str, ...],
    required_channels: tuple[str, ...],
) -> list[int]:
    indices = []
    missing = []
    for channel in required_channels:
        try:
            indices.append(input_channels.index(channel))
        except ValueError:
            missing.append(channel)
    if missing:
        raise ValueError(
            "missing required baseline input channels: " + ", ".join(missing)
        )
    return indices


def model_batch_to_baseline_inputs(
    batch: dict[str, Any],
    *,
    lidar_mode: str = "enriched",
) -> dict[str, np.ndarray]:
    """Adapt a generic dataset batch to ``BaselineFusionModel`` inputs.

    ``dataset_items_to_model_batch`` intentionally keeps a generic concatenated
    ``inputs`` array for NumPy-first downstream tooling. The PyTorch baseline,
    ONNX export, and demos should use this adapter instead of hand-slicing that
    array: channel metadata is validated before the function returns separate
    ``rgb`` and ``lidar_maps`` arrays.

    Args:
        batch: Batch dictionary produced by ``dataset_items_to_model_batch``.
        lidar_mode: ``"sparse"`` for 6 LiDAR channels, or ``"enriched"`` for
            the same 6 sparse channels followed by ``depth_expanded`` and
            ``confidence``. Enriched mode requires those derived channels to be
            present in ``batch["inputs"]``; they are not reconstructed here.
    """

    if lidar_mode not in BASELINE_LIDAR_CHANNELS_BY_MODE:
        valid_modes = ", ".join(sorted(BASELINE_LIDAR_CHANNELS_BY_MODE))
        raise ValueError(f"lidar_mode must be one of: {valid_modes}.")
    if "input_channels" not in batch:
        raise ValueError("input_channels metadata is required for baseline slicing.")

    inputs = np.asarray(batch.get("inputs"), dtype=np.float32)
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [B, C, H, W].")

    input_channels = list(batch["input_channels"])
    if len(input_channels) != inputs.shape[1]:
        raise ValueError("input_channels length must match inputs channel dimension.")

    rgb_indices = _baseline_channel_indices(input_channels, RGB_CHANNELS)
    lidar_indices = _baseline_channel_indices(
        input_channels,
        BASELINE_LIDAR_CHANNELS_BY_MODE[lidar_mode],
    )
    if rgb_indices != sorted(rgb_indices):
        raise ValueError("RGB baseline input channels must appear in canonical order.")
    if lidar_indices != sorted(lidar_indices):
        raise ValueError("LiDAR baseline input channels must appear in canonical order.")

    rgb = inputs[:, rgb_indices, :, :]
    lidar_maps = inputs[:, lidar_indices, :, :]
    return {
        "rgb": rgb.astype(np.float32, copy=False),
        "lidar_maps": lidar_maps.astype(np.float32, copy=False),
    }


def model_batch_to_torch_tensors(
    baseline_inputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Convert baseline NumPy input arrays to torch tensors on demand.

    Torch is imported lazily so the default NumPy batching path remains usable
    without installing the optional ``ml`` dependency group.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised when ml is absent.
        raise ImportError(
            "model_batch_to_torch_tensors requires the optional ml dependencies; "
            "install with `pdm install -G ml`."
        ) from exc
    return {
        "rgb": torch.as_tensor(baseline_inputs["rgb"], dtype=torch.float32),
        "lidar_maps": torch.as_tensor(
            baseline_inputs["lidar_maps"],
            dtype=torch.float32,
        ),
    }
