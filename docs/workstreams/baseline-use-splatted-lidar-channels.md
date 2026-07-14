# Workstream — Baseline with splatted LiDAR channels

Branch: `feature/baseline-use-splatted-lidar-channels`

Parent branch: `feature/baseline-fusion-model`

## Goal

Decide and implement how the first PyTorch baseline should consume the integrated LiDAR surface splatting representation.

The expected outcome is a reviewable model input contract, not a large model architecture.

## Scope

- Keep the baseline small and synthetic-test friendly.
- Compare/choose between:
  - the existing 6-channel sparse LiDAR maps; or
  - an enriched LiDAR tensor that appends `depth_expanded` and `confidence`.
- Keep raw sparse maps available; splatting must complement them, not replace them.
- Avoid torchvision/pretrained backbones until the input contract is stable.

## Acceptance criteria

- Unit tests lock expected channel counts and shape validation.
- README or docs explain which LiDAR channels the baseline consumes.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml` succeeds.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate` succeeds.
- The branch reports `BRANCHE PRÊTE POUR REVIEW: feature/baseline-use-splatted-lidar-channels` only when the contract is documented and tested.
