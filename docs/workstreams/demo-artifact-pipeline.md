# Workstream — Demo artifact pipeline

Branch: `feature/demo-artifact-pipeline`

Parent branch: `main`

## Goal

Prepare a professional demonstration path that generates visual artifacts locally without committing heavy generated files.

## Scope

- Generate overlays / sparse LiDAR maps / splatted LiDAR maps under an ignored output directory such as `results/`.
- Keep synthetic fixtures usable when real KITTI data is unavailable.
- Keep all generated images, videos, ONNX files, TensorRT engines, checkpoints, logs, and benchmark outputs out of Git.
- Prefer deterministic, scriptable artifacts over manual notebook-only steps.

## Acceptance criteria

- A script or documented command creates a small demo artifact from synthetic inputs.
- `.gitignore` protects generated outputs.
- README or docs explain how to regenerate artifacts locally.
- Validation remains CPU-only and lightweight.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev` succeeds.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate` succeeds.
- The branch reports `BRANCHE PRÊTE POUR REVIEW: feature/demo-artifact-pipeline` only once artifacts are reproducibly generated and ignored.
