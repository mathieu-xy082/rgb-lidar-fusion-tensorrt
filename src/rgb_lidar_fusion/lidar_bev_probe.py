"""Bounded LiDAR-only forward/backward diagnostics, without optimizer or checkpoint."""

from __future__ import annotations

from dataclasses import asdict
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .camera_bev import BEVGridConfig
from .camera_depth_analysis import require_gpu_ready, run_gpu_preflight
from .lidar_bev import LidarBEVContext, PillarFeatureNet, scatter_pillar_features
from .lidar_pillars import build_lidar_pillars


def run_lidar_bev_probe(
    point_cloud: np.ndarray,
    *,
    grid: BEVGridConfig,
    emit: Callable[[dict], None],
    device: str = "cuda",
    iterations: int = 3,
    warmup: int = 1,
    minimum_free_memory_mib: int = 2048,
) -> dict:
    """Measure batch-1 FP32 PFN/scatter/context with explicit synchronized phases.

    The 2048 MiB preflight threshold is specific to this LiDAR-only diagnostic;
    camera training retains its existing default. No unsafe automatic retry or
    CUDA memory/cache query occurs after a failure. CPU mode tests the tool, not
    GPU health. Timings exclude raw grouping and input/model transfer, reported
    separately; peak allocation includes resident models/inputs and autograd,
    but not other processes or all driver memory. No Adam state is allocated.
    """

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda.")
    if iterations < 1 or warmup < 1 or minimum_free_memory_mib < 1:
        raise ValueError("iterations, warmup and minimum_free_memory_mib must be positive.")
    phase = "preflight"

    def begin(name: str, **fields) -> None:
        nonlocal phase
        phase = name
        emit({"event": "phase_started", "phase": phase, **fields})

    try:
        begin("preflight")
        if device == "cuda":
            report = run_gpu_preflight(
                minimum_free_memory_mib=minimum_free_memory_mib, run_smoke_test=False,
            )
            emit({"event": "preflight", "report": asdict(report)})
            require_gpu_ready(report)
        selected = torch.device(device)

        def synchronize() -> None:
            if device == "cuda":
                torch.cuda.synchronize(selected)

        begin("raw_grouping")
        start = perf_counter()
        pillars = build_lidar_pillars([point_cloud], grid=grid)
        if not pillars.point_mask.any():
            raise ValueError("probe requires at least one retained point in the ROI.")
        grouping_s = perf_counter() - start
        begin("setup_and_transfer")
        start = perf_counter()
        # Seed CPU initialization only; do not seed or initialize CUDA before preflight.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(13)
            pfn, context = PillarFeatureNet().to(selected), LidarBEVContext().to(selected)
        features = torch.from_numpy(pillars.features).to(selected).requires_grad_()
        mask = torch.from_numpy(pillars.point_mask).to(selected)
        indices = torch.from_numpy(pillars.indices).to(selected)
        counts = torch.from_numpy(pillars.point_count).to(selected)
        synchronize()
        transfer_s = perf_counter() - start

        def forward():
            return context(scatter_pillar_features(
                pfn(features, mask), indices=indices, point_count=counts,
            ))

        measurements = {}
        # Disable algorithm benchmarking; preserve the caller's flags on exit.
        with torch.backends.cudnn.flags(benchmark=False):
            for mode in ("forward", "forward_backward"):
                training = mode == "forward_backward"
                pfn.train(training)
                context.train(training)
                timings, peaks = [], []
                for index in range(warmup + iterations):
                    pfn.zero_grad(set_to_none=True)
                    context.zero_grad(set_to_none=True)
                    features.grad = None
                    synchronize()
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats(selected)
                    begin("forward", mode=mode, iteration=index, warmup=index < warmup)
                    start = perf_counter()
                    with torch.set_grad_enabled(training):
                        output = forward()
                        loss = output.features.square().mean() if training else None
                    synchronize()
                    if not torch.isfinite(output.features).all():
                        raise ValueError("probe output must be finite.")
                    if training:
                        begin("backward", mode=mode, iteration=index)
                        loss.backward()
                        synchronize()
                    duration_ms = (perf_counter() - start) * 1000
                    begin("validation", mode=mode, iteration=index)
                    if training:
                        if features.grad is None or not torch.isfinite(features.grad).all():
                            raise ValueError("probe input gradients must be finite.")
                        for model in (pfn, context):
                            for parameter in model.parameters():
                                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                                    raise ValueError("probe parameter gradients must be finite.")
                    peak = torch.cuda.max_memory_allocated(selected) / 2**20 if device == "cuda" else None
                    if index >= warmup:
                        timings.append(duration_ms)
                        if peak is not None:
                            peaks.append(peak)
                    emit({"event": "iteration_completed", "mode": mode, "iteration": index,
                          "warmup": index < warmup, "duration_ms": duration_ms, "peak_allocated_mib": peak})
                    del output, loss
                measurements[mode] = {
                    "median_ms": median(timings),
                    "peak_allocated_mib": max(peaks) if peaks else None,
                }
        summary = {
            "device": device, "dtype": "float32", "batch_size": 1,
            "torch_version": str(torch.__version__), "iterations": iterations, "warmup": warmup,
            "minimum_free_memory_mib": minimum_free_memory_mib, "grid": asdict(grid),
            "grouping_s": grouping_s, "setup_and_transfer_s": transfer_s,
            "pillar_count": len(pillars.indices), "retained_points": int(pillars.retained_count.sum()),
            "input_shape": list(pillars.features.shape), "output_shape": [1, 32, *grid.shape],
            "measurements": measurements,
            "scope": "Random weights; no optimizer, camera branch, data loading, detector, or export.",
        }
        emit({"event": "completed", "summary": summary})
        return summary
    except Exception as exc:
        # Never synchronize, retry, empty the cache, or query CUDA counters here.
        emit({"event": "failed", "phase": phase, "error_type": type(exc).__name__, "error": str(exc)})
        raise
