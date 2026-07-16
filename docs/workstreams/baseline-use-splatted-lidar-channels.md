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

- Unit tests lock expected channel counts and shape validation. ✅
- README or docs explain which LiDAR channels the baseline consumes. ✅
- `PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml` succeeds. ✅
- `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate` succeeds. ✅
- The branch reports `BRANCHE PRÊTE POUR REVIEW: feature/baseline-use-splatted-lidar-channels` only when the contract is documented and tested. ✅

## Decision

`BaselineFusionModel` now defaults to `lidar_mode="enriched"`, consuming
`Tensor[B, 8, H, W]` LiDAR inputs:

1. the six raw sparse maps from `build_sparse_lidar_maps` are preserved as
   channels 0–5 (`normalized_camera_depth`, vehicle xyz, intensity, point mask);
2. `depth_expanded` from local surface splatting is appended as channel 6;
3. `confidence` from local surface splatting is appended as channel 7.

The sparse-only baseline remains available as `lidar_mode="sparse"` with
`Tensor[B, 6, H, W]` to keep ablation comparisons possible. `build_enriched_lidar_maps`
now provides the canonical handoff from projection/splatting to the baseline: it
copies sparse channels 0–5 unchanged, then appends `depth_expanded` and
`confidence` as channels 6–7. Shape checks reject missing batch dimensions, RGB
tensors that are not `B x 3 x H x W`, unaligned batch/spatial dimensions, and
wrong LiDAR channel counts before concatenation.
