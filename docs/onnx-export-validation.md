# ONNX export and parity validation plan

Status: **legacy baseline deployment guide**.

This document covers `BaselineFusionModel`. It validates the PyTorch-to-ONNX
toolchain, but it does not yet describe export of `CameraDepthModel`, a BEV
lift, or a 3D detector.

Current status:

- `main` exposes `rgb_lidar_fusion.baseline_model.BaselineFusionModel` with a documented RGB + LiDAR-map input contract.
- The default LiDAR contract is the enriched 8-channel representation; `lidar_mode="sparse"` remains available for 6-channel sparse-only comparison.
- No generated `.onnx` or `.onnx.data` artifacts are committed; exports should be written to an ignored output path such as `results/onnx/`.

## Initial scope once the baseline is available

Keep the first graph small and deployment-friendly:

1. Export the baseline forward pass with synthetic RGB + LiDAR-map inputs.
2. Validate input and output shapes after loading the ONNX graph.
3. Compare PyTorch output against ONNX Runtime output on deterministic synthetic tensors.
4. Keep NMS, decoding, visualization, TensorRT-specific plugins, and other complex post-processing outside the ONNX graph.

## Reviewed baseline contract used by scripts

`pdm run export_onnx` and `pdm run validate_onnx` now target the reviewed model interface on `main`:

- deterministic construction via `BaselineFusionModel(output_dim=4, lidar_mode="enriched")`, without requiring a checkpoint download;
- named tensor inputs `rgb: [2, 3, 32, 48]` and `lidar_maps: [2, 8, 32, 48]`;
- stable `prediction: [2, 4]` output asserted before any post-processing;
- no NMS, box decoding, visualization, TensorRT plugins, or dataset I/O inside the exported graph.

The parity scripts instantiate that reviewed interface directly, feed fixed synthetic tensors, write the generated `.onnx` under `results/onnx/`, and recreate only that ignored artifact path during validation.

## Expected PDM shape

The `onnx` dependency group is opt-in and must be installed together with `ml` for export/parity validation:

```toml
[dependency-groups]
onnx = [
  "onnx>=1.16",
  "onnxruntime>=1.18",
  "onnxscript>=0.1",
]
```

Enabled scripts:

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
- **Validated baseline:** base validation plus ONNX validation pass, with no
  generated model artifacts in Git.

## Open limitations

- Dynamic axes are deferred until the baseline input contract is stable.
- Quantization, FP16, TensorRT engine generation, and latency benchmarking belong to the TensorRT milestone, not the first ONNX export.
- KITTI-backed examples are deferred until dataset/model branches are reviewed; synthetic inputs are sufficient for the first parity gate.
