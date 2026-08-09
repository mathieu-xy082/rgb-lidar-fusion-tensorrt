"""NumPy-first adapter from dataset items to model-ready batches."""

from __future__ import annotations

from typing import Any

import numpy as np

from .lidar_splatting import (
    IDENTITY_SPLATTING_CONFIG,
    SPLATTED_LIDAR_MAP_CHANNELS,
    SplattingConfig,
    splat_sparse_depth,
)
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
    include_splatted_depth: bool = True,
    splatting_config: SplattingConfig | None = None,
) -> dict[str, Any]:
    """Stack multiple dataset items into one model-ready NumPy batch.

    The nominal project contract is enriched LiDAR: RGB plus the six sparse
    LiDAR channels plus ``depth_expanded``/``confidence``. Pass
    ``include_splatted_depth=False`` only for explicit sparse-only ablations.
    """

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
    include_splatted_depth: bool = True,
    splatting_config: SplattingConfig | None = None,
) -> dict[str, Any]:
    """Convert one ``KittiSparseLidarDataset`` item into a batch dictionary.

    The nominal model input contract is channel-first NumPy data with an
    explicit batch dimension: RGB channels first, followed by the dataset's
    sparse LiDAR maps and enriched ``depth_expanded``/``confidence`` channels.
    Use ``include_splatted_depth=False`` only for sparse-only ablations.
    """

    image = np.asarray(item["image"], dtype=np.float32)
    lidar_maps = np.asarray(item["lidar_maps"], dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W].")
    meta = item.get("meta", {})
    lidar_representation = meta.get("lidar_representation", "sparse")
    if lidar_representation == "splatted":
        expected_channels = SPLATTED_LIDAR_MAP_CHANNELS
        expected_count = len(SPLATTED_LIDAR_MAP_CHANNELS)
    else:
        expected_channels = LIDAR_MAP_CHANNELS
        expected_count = len(LIDAR_MAP_CHANNELS)
    if lidar_maps.ndim != 3 or lidar_maps.shape[0] != expected_count:
        raise ValueError(
            f"lidar_maps must have shape [{expected_count}, H, W]."
        )
    if image.shape[1:] != lidar_maps.shape[1:]:
        raise ValueError("image and lidar_maps must share height and width.")
    if (
        "lidar_map_channels" in meta
        and tuple(meta["lidar_map_channels"]) != expected_channels
    ):
        raise ValueError("lidar_map_channels metadata must match the lidar representation.")
    sparse_lidar_maps = lidar_maps[:6]
    channels = [image, sparse_lidar_maps]
    input_channels = [*RGB_CHANNELS, *LIDAR_MAP_CHANNELS]
    if include_splatted_depth:
        if lidar_representation == "splatted":
            channels.append(lidar_maps[6:8])
        else:
            splat = splat_sparse_depth(
                sparse_depth=sparse_lidar_maps[0],
                sparse_mask=sparse_lidar_maps[5] > 0.0,
                config=splatting_config or IDENTITY_SPLATTING_CONFIG,
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
        "sparse_lidar_maps": sparse_lidar_maps[np.newaxis, ...].copy(),
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


def model_batch_to_camera_depth_training_batch(
    batch: dict[str, Any],
    *,
    holdout_fraction: float = 0.2,
    seed: int = 0,
    include_splatted_depth: bool = True,
    splatting_config: SplattingConfig | None = None,
) -> dict[str, np.ndarray]:
    """Create leakage-safe camera-depth inputs and held-out sparse targets.

    Holdout pixels are removed from all six sparse LiDAR channels before the
    optional splatted depth and confidence channels are recomputed. A splat from
    another, retained point may still cover a holdout location; the held-out
    measurement itself never contributes to any model input.
    """

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if "input_channels" not in batch:
        raise ValueError("input_channels metadata is required for camera-depth slicing.")

    inputs = np.asarray(batch.get("inputs"), dtype=np.float32)
    sparse_maps = np.asarray(batch.get("sparse_lidar_maps"), dtype=np.float32)
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [B, C, H, W].")
    if sparse_maps.ndim != 4 or sparse_maps.shape[1] != len(LIDAR_MAP_CHANNELS):
        raise ValueError("sparse_lidar_maps must have shape [B, 6, H, W].")
    if inputs.shape[0] != sparse_maps.shape[0] or inputs.shape[2:] != sparse_maps.shape[2:]:
        raise ValueError("inputs and sparse_lidar_maps must share B, H, and W.")

    input_channels = list(batch["input_channels"])
    if len(input_channels) != inputs.shape[1]:
        raise ValueError("input_channels length must match inputs channel dimension.")
    rgb_indices = _baseline_channel_indices(input_channels, RGB_CHANNELS)
    rgb = inputs[:, rgb_indices, :, :].copy()

    rng = np.random.default_rng(seed)
    kept_sparse_maps = sparse_maps.copy()
    holdout_mask = np.zeros(
        (sparse_maps.shape[0], 1, sparse_maps.shape[2], sparse_maps.shape[3]),
        dtype=bool,
    )
    for batch_index in range(sparse_maps.shape[0]):
        valid = (sparse_maps[batch_index, 5] > 0.0) & (
            sparse_maps[batch_index, 0] > 0.0
        )
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            raise ValueError(
                f"batch sample {batch_index} has no valid sparse depth pixels."
            )
        holdout_count = max(1, int(round(valid_indices.size * holdout_fraction)))
        if valid_indices.size > 1:
            holdout_count = min(holdout_count, valid_indices.size - 1)
        held_out = rng.choice(valid_indices, size=holdout_count, replace=False)
        sample_mask = holdout_mask[batch_index, 0].reshape(-1)
        sample_mask[held_out] = True
        kept_sparse_maps[batch_index, :, holdout_mask[batch_index, 0]] = 0.0

    lidar_channels = [kept_sparse_maps]
    if include_splatted_depth:
        config = splatting_config or SplattingConfig()
        splatted = np.zeros(
            (sparse_maps.shape[0], 2, sparse_maps.shape[2], sparse_maps.shape[3]),
            dtype=np.float32,
        )
        for batch_index in range(sparse_maps.shape[0]):
            result = splat_sparse_depth(
                sparse_depth=kept_sparse_maps[batch_index, 0],
                sparse_mask=kept_sparse_maps[batch_index, 5] > 0.0,
                config=config,
            )
            splatted[batch_index, 0] = result.depth_expanded
            splatted[batch_index, 1] = result.confidence
        lidar_channels.append(splatted)

    return {
        "rgb": rgb.astype(np.float32, copy=False),
        "lidar_maps": np.concatenate(lidar_channels, axis=1),
        "depth_target": sparse_maps[:, 0:1].copy(),
        "loss_mask": holdout_mask,
        "kept_sparse_lidar_maps": kept_sparse_maps,
    }
