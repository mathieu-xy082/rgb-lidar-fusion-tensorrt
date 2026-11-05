"""Bounded synthetic CPU probe of the shared BEV backbone, without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from rgb_lidar_fusion.bev_backbone import SharedBEVBackbone
from rgb_lidar_fusion.bev_fusion import FusedBEVFeatures
from rgb_lidar_fusion.camera_bev import BEVGridConfig


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=_positive_int, default=250)
    parser.add_argument("--width", type=_positive_int, default=250)
    parser.add_argument("--channels", nargs=3, type=_positive_int, default=(32, 64, 96))
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument("--warmup", type=_positive_int, default=1)
    parser.add_argument("--iterations", type=_positive_int, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path("results/analysis/bev_fusion/backbone_cpu.json"))
    return parser.parse_args()


def probe(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    # Seed the CPU generator only; no CUDA initialization or global RNG change.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(13)
        model = SharedBEVBackbone(channels=tuple(args.channels))
    shape = (1, args.channels[0], args.height, args.width)
    features = torch.rand(shape, generator=torch.Generator().manual_seed(13), requires_grad=True)
    grid = BEVGridConfig((0., args.height * .2), (0., args.width * .2), (-3., 3.), .2)
    counts = torch.ones(1, 1, args.height, args.width, dtype=torch.int64)
    bev = FusedBEVFeatures(features, grid, counts, counts.float(), counts.clone())
    with torch.no_grad():
        output = model.eval()(bev)
        if output.features.shape != features.shape or not torch.isfinite(output.features).all():
            raise ValueError("backbone must return finite aligned features.")

    def forward() -> None:
        with torch.no_grad():
            model(bev)

    def forward_backward() -> None:
        model.zero_grad(set_to_none=True)
        features.grad = None
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
    if not torch.isfinite(features.grad).all() or any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise ValueError("backbone gradients must be finite.")
    summary = {
        "schema_version": "shared-bev-backbone-cpu-probe-v1",
        "device": "cpu", "dtype": "float32", "torch_version": torch.__version__,
        "shape": list(shape), "channels": list(args.channels), "threads": args.threads,
        "seed": 13, "warmup": args.warmup, "iterations": args.iterations,
        "spatial_shapes": [[args.height, args.width],
                           [(args.height + 1) // 2, (args.width + 1) // 2],
                           [(args.height + 3) // 4, (args.width + 3) // 4]],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_storage_mib": sum(parameter.numel() * parameter.element_size()
                                     for parameter in model.parameters()) / 1024**2,
        "receptive_field_cells_by_axis_phase": [43, 43, 39, 43],
        "scope": "Synthetic random weights; backbone only, including input validation/evidence copying "
                 "and a scalar loss in backward. No encoders, lift, fusion, detector, optimizer, "
                 "checkpoint, GPU, peak-memory measurement or end-to-end latency claim.",
        **timings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(probe(parse_args()), indent=2))
