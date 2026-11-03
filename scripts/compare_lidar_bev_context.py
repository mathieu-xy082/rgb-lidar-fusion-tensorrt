"""Compare untrained dilated and half-resolution BEV context blocks on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from rgb_lidar_fusion.lidar_bev import LidarBEVContext, LidarBEVFeatures


class _MultiResolutionBlocks(nn.Module):
    """Comparison only: fine conv, stride-2 conv, three coarse convs, upsample.

    All convolutions are bias-free, without normalization, at constant width.
    The stride-2 grid retains the input's index-zero center. Transposed 3x3
    convolution with stride 2/padding 1 and explicit output_size restores the
    original grid, including odd/non-square dimensions, without resizing by a
    non-integer ratio. The field is 17 cells at even output indices and 19 at
    odd ones, independently per axis. This is not a second production encoder.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.ReLU())
        coarse = [nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False), nn.ReLU()]
        for _ in range(3):
            coarse.extend([nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.ReLU()])
        self.coarse = nn.Sequential(*coarse)
        self.up = nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        fine = self.fine(features)
        up = self.up(self.coarse(fine), output_size=fine.shape)
        return F.relu(fine + up)


class _MultiResolutionContext(LidarBEVContext):
    """Reuse production input checks/evidence copying for a fair CPU comparison."""

    def __init__(self, channels: int) -> None:
        super().__init__(channels=channels)
        self.blocks = _MultiResolutionBlocks(channels)


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=_positive_int, default=250)
    parser.add_argument("--width", type=_positive_int, default=250)
    parser.add_argument("--channels", type=_positive_int, default=32)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--warmup", type=_positive_int, default=2)
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/lidar_bev/context_comparison_cpu.json"))
    return parser.parse_args()


def compare(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(13)
    shape = (args.batch_size, args.channels, args.height, args.width)
    inputs = torch.rand(shape, requires_grad=True)
    bev = LidarBEVFeatures(inputs, torch.ones(args.batch_size, 1, args.height, args.width, dtype=torch.int64))
    results = []
    for name, model in (
        ("dilated", LidarBEVContext(channels=args.channels)),
        ("multi_resolution_reference", _MultiResolutionContext(args.channels)),
    ):
        with torch.no_grad():
            output = model.eval()(bev)
            if output.features.shape != inputs.shape or not torch.isfinite(output.features).all():
                raise ValueError("comparison model must produce aligned, finite features.")

        def forward() -> None:
            with torch.no_grad():
                model(bev)

        def forward_backward() -> None:
            model.zero_grad(set_to_none=True)
            inputs.grad = None
            model(bev).features.square().mean().backward()

        timings = {}
        for label, operation in (("forward_ms", forward), ("forward_backward_ms", forward_backward)):
            model.train(label == "forward_backward_ms")
            for _ in range(args.warmup):
                operation()
            samples = []
            for _ in range(args.iterations):
                start = perf_counter()
                operation()
                samples.append((perf_counter() - start) * 1000)
            timings[label] = median(samples)
        results.append({
            "name": name,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "parameter_mib": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1024**2,
            **timings,
        })
    summary = {
        "schema_version": "lidar-bev-context-comparison-v1",
        "device": "cpu", "dtype": "float32", "torch_version": torch.__version__,
        "shape": list(shape), "threads": args.threads,
        "warmup": args.warmup, "iterations": args.iterations, "seed": 13,
        "scope": "Random weights; context only, no PFN, detector, GPU, or peak-memory measurement.",
        "receptive_field": {"dilated": "19x19", "multi_resolution_reference": "17 or 19 per axis, depending on output-index parity"},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(compare(parse_args()), indent=2))
