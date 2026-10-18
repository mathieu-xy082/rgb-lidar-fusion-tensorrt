# Workstream — Dataset to model contract

> Workstream terminé et archivé le 2026-08-17. Voir
> [l'index documentaire actif](../../README.md).

Branch: `feature/dataset-model-contract`

Parent branch: `main`

## Goal

Create the minimal, tested bridge between `KittiSparseLidarDataset`, LiDAR splatting, and the future PyTorch baseline model.

## Scope

- Convert one dataset item into a model-ready batch representation.
- Keep implementation NumPy-first unless the branch explicitly depends on the `ml` group.
- Add an explicit option for derived splatting channels (`depth_expanded`, `confidence`).
- Preserve raw sparse channels so the experimental splatting representation remains comparable.
- Use synthetic fixtures only; do not require downloaded KITTI data.

## Acceptance criteria

- Tests cover batch dimension, channel order, shape validation, and optional splatting channels.
- The contract is documented in README or a focused docs file.
- Generated artifacts remain outside Git.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev` succeeds.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate` succeeds.
- The branch reports `BRANCHE PRÊTE POUR REVIEW: feature/dataset-model-contract` only when the adapter and docs are tested.

## Implemented contract

`rgb_lidar_fusion.model_batch.dataset_item_to_model_batch` converts one
`KittiSparseLidarDataset`-style item into a NumPy-only model batch:

- `inputs`: `float32` tensor-shaped array `[1, C, H, W]` with the batch dimension
  inserted first.
- Default channel order: `rgb_red`, `rgb_green`, `rgb_blue`, then the six sparse
  LiDAR channels from `LIDAR_MAP_CHANNELS` (`normalized_camera_depth`, vehicle
  xyz, `intensity`, `point_mask`).
- With `include_splatted_depth=True`, `depth_expanded` and `confidence` are
  appended after the sparse LiDAR channels using `splat_sparse_depth`.
- `sparse_lidar_maps`: preserved copy of the original six sparse maps as
  `[1, 6, H, W]`, so splatting never replaces the sparse representation.
- `dataset_items_to_model_batch([...])` stacks validated single-item batches into
  `[N, C, H, W]` inputs plus `[N, 6, H, W]` sparse maps while carrying per-item
  `targets` and `metas` lists.
- The adapter validates RGB shape `[3, H, W]`, LiDAR-map shape `[6, H, W]`,
  shared image/LiDAR spatial dimensions, and any dataset-provided
  `meta.lidar_map_channels` declaration against `LIDAR_MAP_CHANNELS`.

The contract intentionally remains NumPy-first; no `ml` dependency group is
required for this bridge.
