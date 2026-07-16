# ONNX export and parity validation plan

Milestone 4 starts from the reviewed baseline model now available on `main`.

Current status:

- `main` exposes `rgb_lidar_fusion.baseline_model.BaselineFusionModel` with a documented RGB + LiDAR-map input contract.
- The default LiDAR contract is the enriched 8-channel representation; `lidar_mode="sparse"` remains available for 6-channel sparse-only comparison.
- No generated `.onnx` artifacts are committed; exports should be written to an ignored output path such as `results/onnx/`.

## Initial scope once the baseline is available

Keep the first graph small and deployment-friendly:

1. Export the baseline forward pass with synthetic RGB + LiDAR-map inputs.
2. Validate input and output shapes after loading the ONNX graph.
3. Compare PyTorch output against ONNX Runtime output on deterministic synthetic tensors.
4. Keep NMS, decoding, visualization, TensorRT-specific plugins, and other complex post-processing outside the ONNX graph.

## Baseline contract required before scripts

Do not add `pdm run export_onnx` or `pdm run validate_onnx` until `main` exposes a reviewed model interface with:

- deterministic construction for a tiny synthetic fixture, without requiring a checkpoint download;
- named tensor inputs for `rgb` and `lidar_maps`, or a documented fused tensor replacement if the baseline standardizes on concatenation;
- a stable `prediction` output tensor or mapping whose shape can be asserted before any post-processing;
- no NMS, box decoding, visualization, TensorRT plugins, or dataset I/O inside the exported graph.

The first parity script should instantiate that reviewed interface directly, feed fixed synthetic tensors, write the generated `.onnx` under `results/onnx/`, and remove/recreate only that ignored artifact path during validation.

## Expected PDM shape

Add an `onnx` dependency group when the export scripts are implemented and exercised against the reviewed baseline:

```toml
[dependency-groups]
onnx = [
  "onnx",
  "onnxruntime",
]
```

Candidate scripts once useful:

```toml
[tool.pdm.scripts]
export_onnx = "python scripts/export_onnx.py"
validate_onnx = "python scripts/validate_onnx.py"
```

PyTorch should remain in the `ml` group or whichever reviewed group owns the baseline model; do not duplicate it into `onnx` unless the dependency strategy changes.

## Validation ladder

- **Prepared:** this document exists, the dependency gate is explicit, and no heavy artifacts are versioned.
- **Export smoke:** `pdm run export_onnx` creates an ONNX file under an ignored output directory.
- **Shape validation:** `pdm run validate_onnx` loads the graph and confirms input/output names and shapes.
- **Parity validation:** deterministic PyTorch and ONNX Runtime outputs match within a documented tolerance.
- **Review-ready:** base validation plus ONNX validation pass on `feature/onnx-export-validation`, with docs updated and no generated model artifacts in git.

## Open limitations

- Dynamic axes are deferred until the baseline input contract is stable.
- Quantization, FP16, TensorRT engine generation, and latency benchmarking belong to the TensorRT milestone, not the first ONNX export.
- KITTI-backed examples are deferred until dataset/model branches are reviewed; synthetic inputs are sufficient for the first parity gate.
