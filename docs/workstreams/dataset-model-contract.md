# Workstream — Dataset to model contract

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
