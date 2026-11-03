"""Profile one raw KITTI cloud through the LiDAR-only encoder in a fresh process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velodyne", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--minimum-free-memory-mib", type=int, default=2048)
    parser.add_argument("--x-range", type=float, nargs=2, default=(0., 50.))
    parser.add_argument("--y-range", type=float, nargs=2, default=(-25., 25.))
    parser.add_argument("--z-range", type=float, nargs=2, default=(-3., 3.))
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--output", type=Path, help="New JSONL log; overwrite is refused.")
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.minimum_free_memory_mib) < 1:
        parser.error("iteration counts and memory threshold must be positive")
    import torch
    from rgb_lidar_fusion.camera_bev import BEVGridConfig
    from rgb_lidar_fusion.lidar_bev_probe import run_lidar_bev_probe
    from rgb_lidar_fusion.project_lidar import load_velodyne_bin

    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(f"results/diagnostics/lidar_bev_{stamp}_{os.getpid()}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostic log: {output}", flush=True)
    with output.open("x", encoding="utf-8") as log:
        def emit(event: dict) -> None:
            event["time_utc"] = datetime.now(timezone.utc).isoformat()
            log.write(json.dumps(event) + "\n")
            log.flush()
            print(json.dumps(event), flush=True)

        run_lidar_bev_probe(
            load_velodyne_bin(args.velodyne),
            grid=BEVGridConfig(tuple(args.x_range), tuple(args.y_range), tuple(args.z_range), args.resolution),
            device=args.device, iterations=args.iterations, warmup=args.warmup,
            minimum_free_memory_mib=args.minimum_free_memory_mib, emit=emit,
        )


if __name__ == "__main__":
    main()
