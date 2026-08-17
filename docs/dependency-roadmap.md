# Dependency roadmap

Status: **current dependency guide**.

This repository keeps the default environment deliberately small so geometry, projection, CI, and documentation stay fast and reproducible. Heavy ML/runtime dependencies live in explicit PDM groups or external GPU/TensorRT setup notes.

## Active now

### Default dependencies

- `numpy` — geometry, projection, sparse/enriched LiDAR maps, lightweight synthetic dataset fixtures, and NumPy-first dataset/model batching.

The dataset layer remains NumPy-first. PyTorch conversion happens through the model/training adapter when the `ml` group is installed.

### `dev` group

- `pytest` — unit and smoke validation.

Commands:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

### `ml` group

- `torch==2.13.0+cu129` — models, training loops, checkpointing and
  forward/backward validation on CPU or a compatible NVIDIA runtime;
- `pillow` — loading and resizing local KITTI images for camera-depth training.

Commands:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
```

Training tests remain CPU-safe. CUDA is selected at runtime only when available.

### `onnx` group

- `onnx` — graph serialization/checking;
- `onnxruntime` — CPU parity validation;
- `onnxscript` — supports current PyTorch ONNX export stack.

Commands:

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml -G onnx
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate_onnx
```

Use this group for the legacy synthetic export today, then for trained models
once their export contracts are stable.

## Optional additions when justified

### Training configuration utilities

Do not add Hydra, Lightning, WandB, MLflow, or similar frameworks for the first training loop unless the manual loop becomes too cumbersome. The first branch should prefer a transparent small PyTorch loop with standard-library config parsing or a tiny YAML dependency if needed.

Potential later dependencies:

```toml
training = [
  "pyyaml",
  "tqdm",
]
```

Only add these after the script and tests justify them.

### `torchvision` — dataset/model utilities

Add to the existing `ml` group only when image transforms, pretrained backbones, or detection utilities are introduced. The current baseline should stay plain PyTorch until the training objective requires more.

### `visualization` group — overlays and notebooks

Add when real KITTI samples, plots, or richer visual artifacts are needed.

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

## GPU training environment

GPU access is an infrastructure requirement, not a default project dependency.

The repository should support:

```text
CPU CI smoke training
CUDA local/cloud training when available
```

Before renting or configuring a GPU, document:

- provider or local machine;
- GPU model;
- driver version;
- CUDA version;
- PyTorch CUDA build;
- dataset path/storage strategy;
- expected run duration/budget;
- checkpoint and metrics output path.

The current local GPU results and warnings are recorded in
[the camera-depth workstream](workstreams/camera-depth-backbone.md). Project
gates are tracked in [milestones](milestones.md).

## TensorRT milestone

Do not add TensorRT blindly to `pyproject.toml` until the target runtime is known.

TensorRT installation is usually platform/CUDA-specific and may involve NVIDIA package indexes, system libraries, or container images. For the TensorRT milestone, document the selected target first:

- local machine vs container;
- CUDA version;
- TensorRT version;
- Python bindings availability;
- build path from ONNX to engine;
- whether the ONNX input comes from random weights or a trained checkpoint.

The actionable runtime/benchmark plan lives in
[tensorrt-benchmark-plan.md](tensorrt-benchmark-plan.md). Current guarded scripts
intentionally work as stubs on hosts without TensorRT:

```bash
python scripts/tensorrt_benchmark.py --detect
python scripts/tensorrt_benchmark.py --schema
python scripts/tensorrt_build_engine.py --onnx results/model.onnx --engine results/model_fp16.engine
python scripts/tensorrt_infer.py --engine results/model_fp16.engine
```

Add either a dedicated PDM group or an external setup document only after the target CUDA/TensorRT runtime has been selected and verified.

## Artifact dependency policy

Never commit generated artifacts or datasets:

```text
data/** real datasets
results/**/*.onnx
results/**/*.onnx.data
results/**/*.engine
results/**/*.plan
results/**/*.pt
results/**/*.pth
results/**/*.ckpt
results/training/**
```

If a future branch needs a tiny fixture, it must be intentionally small, documented, and reviewed before versioning.
