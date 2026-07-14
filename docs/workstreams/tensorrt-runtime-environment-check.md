# Workstream — TensorRT runtime environment check

Branch: `feature/tensorrt-runtime-environment-check`

Parent branch: `feature/tensorrt-benchmark-plan`

## Goal

Add a reproducible runtime diagnostic for the future ONNX → TensorRT FP16 path.

This branch should make missing CUDA/TensorRT dependencies explicit and actionable instead of failing with opaque import or driver errors.

## Scope

- Detect Python TensorRT bindings if available.
- Detect CUDA/GPU availability if available, without requiring it in CI.
- Report versions and capability gaps in a structured way.
- Keep default validation green on CPU-only machines.
- Do not require building a real engine until ONNX export is integrated.

## Acceptance criteria

- A CLI or helper returns a machine-readable diagnostic object.
- Unit tests cover CPU-only / missing TensorRT behavior.
- Docs explain the expected local/container environment for future TensorRT work.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev` succeeds.
- `PDM_IGNORE_ACTIVE_VENV=1 pdm run validate` succeeds.
- The branch reports `BRANCHE PRÊTE POUR REVIEW: feature/tensorrt-runtime-environment-check` only once diagnostics are implemented and tested.
