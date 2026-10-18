"""Run a bounded camera-depth diagnostic outside Jupyter; never save weights."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint", type=Path, help="Trusted local checkpoint; read only.")
    parser.add_argument("--disable-cudnn", action="store_true", help="Diagnostic comparison only.")
    parser.add_argument("--output", type=Path, help="New JSONL log path; existing files are refused.")
    args = parser.parse_args()
    for name in ("steps", "height", "width", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    # Set before importing torch, including any project module importing torch.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    from rgb_lidar_fusion.camera_depth_gpu_probe import run_camera_depth_probe

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(f"results/diagnostics/camera_depth_gpu_{stamp}_{os.getpid()}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Diagnostic log: {output}", flush=True)
    with output.open("x", encoding="utf-8") as log:
        def emit(event: dict) -> None:
            event["time_utc"] = datetime.now(timezone.utc).isoformat()
            event["pid"] = os.getpid()
            event["cuda_launch_blocking"] = os.environ["CUDA_LAUNCH_BLOCKING"]
            log.write(json.dumps(event) + "\n")
            log.flush()
            summary = {key: value for key, value in event.items() if key not in {"nvidia_smi", "report"}}
            print(json.dumps(summary), flush=True)

        run_camera_depth_probe(
            emit=emit, steps=args.steps, height=args.height, width=args.width,
            batch_size=args.batch_size, device=args.device,
            checkpoint=args.checkpoint, cudnn_enabled=not args.disable_cudnn,
        )


if __name__ == "__main__":
    main()
