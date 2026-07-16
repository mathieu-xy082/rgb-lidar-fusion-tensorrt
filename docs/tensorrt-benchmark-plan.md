# TensorRT FP16 runtime and benchmark plan

Milestone 5 must stay reproducible without assuming that every development host
has CUDA or TensorRT installed. The repository therefore provides guarded scripts
that document the expected command line and fail clearly when the runtime is
missing.

## Current host diagnosis

The scheduled job that prepared this plan observed:

- `nvidia-smi`: not found
- `nvcc`: not found
- `trtexec`: not found
- Python package `tensorrt`: not importable

This host is therefore suitable for planning and validation of stubs only. It is
not a proven TensorRT execution target.

## Runtime options

### Preferred: NVIDIA TensorRT container

Use a pinned NVIDIA container once the ONNX milestone produces a model:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/tensorrt:<PINNED_VERSION>-py3 \
  bash -lc 'trtexec --version && python scripts/tensorrt_benchmark.py --detect'
```

Selection checklist:

1. Pick the image after confirming the target GPU driver supports its CUDA
   version.
2. Record the TensorRT, CUDA, cuDNN, GPU model, and driver versions in the
   benchmark output notes.
3. Build engines inside the same image/runtime used for inference. TensorRT
   engines are not portable across arbitrary GPU/TensorRT combinations.

### Alternative: local CUDA/TensorRT install

A local install is acceptable only after documenting:

- NVIDIA driver and GPU model (`nvidia-smi`);
- CUDA toolkit/runtime version (`nvcc --version` or container metadata);
- `trtexec` path and version;
- whether Python bindings (`import tensorrt`) are installed in the active PDM
  environment or intentionally avoided.

Do not add `tensorrt` to `pyproject.toml` until the target package index and CUDA
compatibility are known.

## Guarded scripts

The scripts are safe to run on hosts without TensorRT:

```bash
python scripts/tensorrt_benchmark.py --detect
python scripts/tensorrt_benchmark.py --schema
python scripts/tensorrt_build_engine.py --onnx results/model.onnx --engine results/model_fp16.engine
python scripts/tensorrt_infer.py --engine results/model_fp16.engine
```

Expected behavior before a runtime exists:

- detection/schema commands return successfully;
- build/infer/benchmark execution exits with code `2` and a message explaining
  that `trtexec`/`tensorrt` is unavailable or that ONNX tensor contracts are not
  defined yet.

## ONNX to FP16 engine path

Once Milestone 4 produces an ONNX model and parity validation:

```bash
python scripts/tensorrt_build_engine.py \
  --onnx results/<model>.onnx \
  --engine results/<model>_fp16.engine \
  --workspace-mb 2048
```

The initial backend is `trtexec` because it is easy to audit in CI/container logs.
Python TensorRT bindings can be added later for custom preprocessing, dynamic
shapes, or richer profiling.

## Benchmark metrics schema

Each benchmark run should emit JSON compatible with:

```json
{
  "batch_size": 1,
  "device": "GPU name + compute capability",
  "fps": 0.0,
  "input_shape": "1xCxHxW",
  "latency_ms": {
    "max": 0.0,
    "mean": 0.0,
    "min": 0.0,
    "p50": 0.0,
    "p95": 0.0
  },
  "measured_runs": 200,
  "model": "model_fp16.engine",
  "notes": "container/local runtime, CUDA/TensorRT versions, limitations",
  "precision": "fp16",
  "runtime": "trtexec or python-tensorrt",
  "warmup_runs": 20
}
```

Definitions:

- `p50`: median end-to-end inference latency in milliseconds after warmup;
- `p95`: 95th percentile latency, used to catch jitter/tail latency;
- `fps`: `batch_size * 1000 / latency_ms.mean`;
- warmup and measured run counts must be reported with every result.

The lightweight helper `rgb_lidar_fusion.tensorrt_runtime.latency_summary_ms`
implements these p50/p95/FPS calculations with no NumPy/TensorRT dependency, so
future runtime adapters can share one tested metric contract.

## Artifact policy

Do not commit generated deployment artifacts:

- `*.onnx`
- `*.engine`
- `*.plan`
- checkpoints (`*.pt`, `*.pth`, `*.ckpt`)

The existing `.gitignore` covers these under `results/` and for checkpoint
extensions. Engine paths should stay under `results/` or another ignored runtime
artifact directory.
