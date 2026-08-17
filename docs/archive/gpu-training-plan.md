# GPU training plan

> Archive historique déplacée le 2026-08-17. Ce plan précède le backbone
> camera-depth et ses runs GPU. Voir les
> [milestones actuels](../milestones.md).

This document prepares the first real training phase for the RGB-LiDAR fusion prototype. The goal is not to chase state-of-the-art KITTI detection yet; the goal is to prove an end-to-end ML ownership story:

```text
RGB + enriched LiDAR dataset
→ PyTorch training loop
→ checkpoint + metrics
→ ONNX export of a trained checkpoint
→ TensorRT runtime/benchmark path
→ reproducible demo artifacts
```

## Why this phase matters

The repository already validates geometry, dataset contracts, enriched LiDAR batching, a baseline PyTorch model, ONNX parity scaffolding, TensorRT runtime planning, and demo artifact generation. Without a training phase, however, the project remains mostly a pipeline integration demonstrator. A modest but real training run turns it into a credible ML deployment project.

## Current input contract

The nominal model input contract is enriched LiDAR:

```text
rgb:        [B, 3, H, W]
lidar_maps: [B, 8, H, W]
```

LiDAR channels:

```text
0: normalized sparse depth
1: x_vehicle
2: y_vehicle
3: z_vehicle
4: intensity
5: sparse point mask
6: depth_expanded
7: confidence
```

Downstream training/export/demo code should build inputs through the adapter rather than manual slicing:

```python
from rgb_lidar_fusion.model_batch import (
    dataset_items_to_model_batch,
    model_batch_to_baseline_inputs,
    model_batch_to_torch_tensors,
)

batch = dataset_items_to_model_batch(items)
arrays = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
tensors = model_batch_to_torch_tensors(arrays)
```

Sparse-only mode remains an ablation/debug path, not the default training contract.

## Recommended training milestone

Create a dedicated branch after the ONNX and demo branches are integrated:

```text
feature/training-loop-baseline
```

The first training branch should stay deliberately small and CPU-safe in CI while being GPU-ready for real runs.

### Deliverables

```text
configs/training/synthetic_smoke.yaml
configs/training/kitti_tiny.yaml
docs/gpu-training-plan.md
scripts/train_baseline.py
src/rgb_lidar_fusion/training.py
tests/test_training_loop.py
```

### Required capabilities

- deterministic seed handling;
- `cuda` if available, otherwise `cpu`;
- clear device logging;
- tiny synthetic smoke training usable in CI;
- configurable batch size, epochs, learning rate, output directory;
- checkpoint save/resume;
- metrics JSON/CSV saved outside Git-tracked artifacts;
- run metadata JSON with seed, selected device diagnostic, resume start epoch, and checkpoint path;
- no committed datasets/checkpoints;
- failure messages that explain missing CUDA/GPU resources rather than crashing cryptically;
- CUDA training logs should include the selected GPU name for run traceability.

## Initial learning target

Do **not** jump directly to full KITTI 3D detection as the first training objective. That would add annotation parsing, detection heads, matching, NMS, evaluation metrics, and much more debugging surface at once.

Preferred first target:

```text
Train a small baseline on a simple dense or low-dimensional supervised proxy task that exercises both RGB and enriched LiDAR inputs.
```

Acceptable first proxy targets:

| Target | Why it is useful | Risk |
|---|---|---|
| Synthetic regression/classification from RGB+LiDAR patterns | Fast, deterministic, proves trainability and checkpointing | Less portfolio-impressive if left there |
| Dense depth/occupancy-style pseudo-target from LiDAR maps | Good fusion story, uses LiDAR semantics | Needs careful target definition |
| Small heatmap/objectness proxy | Closer to perception | Requires more target design |

Recommended path:

1. synthetic smoke target for CI;
2. tiny local KITTI subset path for real experiments;
3. only then consider a detection-like head/evaluation branch.

## GPU resource requirements

### Minimum viable GPU

A small NVIDIA GPU is enough for the first meaningful run:

| Use case | Suggested resource |
|---|---|
| smoke tests | CPU or any CUDA GPU |
| baseline experiments | RTX 3060/4060/4070, T4, L4 |
| faster iteration / larger images | RTX 4080/4090, A10 |
| serious TensorRT benchmarking | NVIDIA GPU with supported CUDA/TensorRT stack |

The key requirement is not only compute power. It is a stable NVIDIA CUDA environment with durable storage for datasets, checkpoints, logs, ONNX files, and TensorRT engines.

### Cloud options

Good candidates for short dedicated runs:

- RunPod;
- Lambda Labs;
- Vast.ai;
- Paperspace;
- AWS/GCP/Azure GPU instances;
- OVH GPU instances if availability/budget fits.

Colab/Kaggle can be useful for quick experiments but are not recommended as the main path for TensorRT or reproducible checkpoint generation because sessions and GPU availability are not stable enough.

## Artifact policy

Never commit:

```text
*.ckpt
*.pt
*.pth
*.onnx
*.engine
*.plan
results/** generated model artifacts
data/** real datasets
```

Recommended generated locations:

```text
results/training/<run-id>/metrics.json
results/training/<run-id>/checkpoint.pt
results/onnx/<run-id>/baseline_fusion.onnx
results/tensorrt/<run-id>/baseline_fusion_fp16.engine
```

These paths should stay ignored by Git. For scheduled/recurring jobs, prefer a temp/cache root outside the repository if outputs are not user-facing deliverables.

## Validation ladder

### CI / CPU gate

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run validate
PDM_IGNORE_ACTIVE_VENV=1 pdm run pytest tests/test_training_loop.py -q
```

Expected proof:

- one tiny training step or epoch runs on CPU;
- loss is finite;
- at least one parameter changes or receives a finite non-zero gradient;
- checkpoint write/read works using a temporary directory;
- no generated artifact is committed.

### Local/Cloud GPU gate

```bash
PDM_IGNORE_ACTIVE_VENV=1 pdm install -G dev -G ml
PDM_IGNORE_ACTIVE_VENV=1 pdm run python scripts/train_baseline.py \
  --config configs/training/kitti_tiny.yaml \
  --device cuda
```

Expected proof:

- script reports CUDA device name;
- training completes configured epochs;
- checkpoint and metrics are written;
- loss/metric trend is documented honestly;
- command and hardware are recorded in the run metadata.

### Deployment continuation gate

After a trained checkpoint exists:

1. export ONNX from checkpoint;
2. validate PyTorch vs ONNX Runtime parity;
3. build/run TensorRT only on a compatible NVIDIA target;
4. generate demo artifacts from the trained checkpoint.

## Open decisions before renting GPU

Before spending GPU budget, decide:

- exact first target: synthetic proxy, dense pseudo-depth, or heatmap/objectness;
- image resolution for first run;
- dataset source and storage path;
- cloud provider and GPU class;
- expected maximum budget and run duration;
- checkpoint format and metadata schema;
- whether TensorRT benchmarking is done on the same GPU instance or a separate runtime target.

## Recommended next branch order

```text
1. integrate feature/onnx-export-validation
2. integrate feature/demo-artifact-pipeline
3. create feature/training-loop-baseline
4. implement CPU-safe training smoke loop
5. run first dedicated GPU experiment
6. export trained checkpoint to ONNX
7. run TensorRT benchmark on verified CUDA/TensorRT target
```
