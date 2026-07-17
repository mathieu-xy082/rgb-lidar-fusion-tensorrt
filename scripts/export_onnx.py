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


def export_baseline_onnx(output: Path, *, batch_size: int = 2, height: int = 32, width: int = 48) -> Path:
    """Export ``BaselineFusionModel`` with deterministic synthetic inputs."""

    import torch

    from rgb_lidar_fusion.baseline_model import BaselineFusionModel, ENRICHED_LIDAR_CHANNELS

    torch.manual_seed(0)
    model = BaselineFusionModel(output_dim=4, lidar_mode="enriched").eval()
    rgb = torch.linspace(0.0, 1.0, steps=batch_size * 3 * height * width, dtype=torch.float32).reshape(
        batch_size, 3, height, width
    )
    lidar_maps = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * ENRICHED_LIDAR_CHANNELS * height * width,
        dtype=torch.float32,
    ).reshape(batch_size, ENRICHED_LIDAR_CHANNELS, height, width)

    output.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model,
            (rgb, lidar_maps),
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
