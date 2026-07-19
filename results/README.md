# Results

Generated artifacts live here during local experiments, but binary outputs are ignored by Git.

Expected subfolders:

```text
projected_lidar_examples/  PNG/JPG overlays for Milestone 1
sparse_maps/               Compressed sparse LiDAR map arrays
splatted_maps/             Compressed local depth splat arrays
demo_artifacts/            Reproducible synthetic demo bundle
onnx/                      Local ONNX exports for validation
benchmarks/                Markdown/CSV latency summaries
training/                  Checkpoints and JSON/CSV metrics from baseline training
```

Everything under `results/` is ignored except this README. Regenerate local artifacts with:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm run demo-artifacts
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate_onnx
PDM_IGNORE_ACTIVE_VENV=1 pdm run train-baseline -- --config configs/training/synthetic_smoke.yaml
```

Do not commit generated `.onnx`, `.onnx.data`, TensorRT engine, plan, image, video, array, metrics, or checkpoint artifacts.
