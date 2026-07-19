"""Export the reviewed baseline fusion model to ONNX.

The script intentionally exports only the small baseline forward pass with
synthetic tensors. NMS, decoding, visualization, TensorRT plugins, and dataset
I/O stay outside the graph for the first Milestone 4 validation gate.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path


def _default_output_path() -> Path:
    return Path("results/onnx/baseline_fusion.onnx")


def synthetic_baseline_inputs(*, batch_size: int = 2, height: int = 32, width: int = 48):
    """Build deterministic baseline inputs through the dataset-batch adapter.

    ONNX exports should exercise the same enriched-by-default contract as
    dataset/demo code: generic NumPy batch first, then explicit model-call
    tensors via ``model_batch_to_baseline_inputs``. This prevents export scripts
    from drifting into hand-built LiDAR channel layouts.
    """

    import numpy as np

    from rgb_lidar_fusion.model_batch import (
        dataset_items_to_model_batch,
        model_batch_to_baseline_inputs,
        model_batch_to_torch_tensors,
    )

    items = []
    for batch_index in range(batch_size):
        image = np.linspace(
            0.0,
            1.0,
            num=3 * height * width,
            dtype=np.float32,
        ).reshape(3, height, width)
        lidar_maps = np.zeros((6, height, width), dtype=np.float32)
        lidar_maps[0] = np.linspace(
            0.0,
            1.0,
            num=height * width,
            dtype=np.float32,
        ).reshape(height, width)
        lidar_maps[1] = batch_index / max(batch_size, 1)
        lidar_maps[2] = np.linspace(
            -1.0,
            1.0,
            num=height * width,
            dtype=np.float32,
        ).reshape(height, width)
        lidar_maps[3] = 0.25
        lidar_maps[4] = 0.5
        lidar_maps[5, batch_index % height :: 4, batch_index % width :: 4] = 1.0
        items.append({"image": image, "lidar_maps": lidar_maps})

    batch = dataset_items_to_model_batch(items)
    baseline_inputs = model_batch_to_baseline_inputs(batch, lidar_mode="enriched")
    return model_batch_to_torch_tensors(baseline_inputs)


def export_baseline_onnx(
    output: Path,
    *,
    batch_size: int = 2,
    height: int = 32,
    width: int = 48,
) -> Path:
    """Export ``BaselineFusionModel`` with deterministic adapter-built inputs."""

    import torch

    from rgb_lidar_fusion.baseline_model import BaselineFusionModel

    torch.manual_seed(0)
    model = BaselineFusionModel(output_dim=4, lidar_mode="enriched").eval()
    inputs = synthetic_baseline_inputs(batch_size=batch_size, height=height, width=width)

    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model,
            (inputs["rgb"], inputs["lidar_maps"]),
            output,
            input_names=["rgb", "lidar_maps"],
            output_names=["prediction"],
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=48)
    args = parser.parse_args()

    output = export_baseline_onnx(
        args.output,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
    )
    print(f"Exported ONNX baseline to {output}")


if __name__ == "__main__":
    main()
