# Dependency roadmap

This repository starts with a deliberately small PDM environment so the geometry/projection milestone remains fast and reproducible.

## Active now

### Default dependencies

- `numpy` — geometry, projection, sparse maps.

### `dev` group

- `pytest` — unit and smoke validation.

Commands:

```bash
pdm install -G dev
pdm run validate
```

## Add when the relevant milestone starts

### `ml` group — PyTorch model milestone

Add when Milestone 2/3 starts: dataset + baseline model.

Candidate dependencies:

```toml
[dependency-groups]
ml = [
  "torch",
  "torchvision",
]
```

Keep this separate from the default install because PyTorch is heavy and may need CPU/CUDA-specific wheel selection.

### `onnx` group — export and validation milestone

Add when Milestone 4 starts.

Candidate dependencies:

```toml
onnx = [
  "onnx",
  "onnxruntime",
]
```

Use this group for export scripts and PyTorch vs ONNX Runtime parity checks.

### `visualization` group — overlays and notebooks

Add when real KITTI samples are available and visual artifacts are needed.

Candidate dependencies:

```toml
visualization = [
  "opencv-python",
  "matplotlib",
]
```

### `notebook` group — exploration

Add when interactive exploration becomes useful.

Candidate dependencies:

```toml
notebook = [
  "jupyterlab",
  "ipykernel",
]
```

### TensorRT milestone

Do not add TensorRT blindly to `pyproject.toml` until the target runtime is known.

TensorRT installation is usually platform/CUDA-specific and may involve NVIDIA package indexes, system libraries, or container images. For Milestone 5, document the selected target first:

- local machine vs container;
- CUDA version;
- TensorRT version;
- Python bindings availability;
- build path from ONNX to engine.

Then add either a dedicated PDM group or an external setup document depending on what is reproducible.
