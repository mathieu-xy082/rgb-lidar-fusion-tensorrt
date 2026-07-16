#!/usr/bin/env python3
"""Build a TensorRT FP16 engine from ONNX when a runtime is available.

This is intentionally a guarded stub: it fails before doing work on hosts without
TensorRT instead of trying to install platform-specific NVIDIA packages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rgb_lidar_fusion.tensorrt_runtime import (
    require_tensorrt_available,
    run_command,
    validate_engine_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="Input ONNX model path")
    parser.add_argument("--engine", required=True, type=Path, help="Output .engine/.plan path")
    parser.add_argument("--workspace-mb", type=int, default=2048, help="TensorRT workspace size in MiB")
    parser.add_argument("--dry-run", action="store_true", help="Print the trtexec command without executing it")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        engine = validate_engine_output_path(args.engine)
        env = require_tensorrt_available()
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.onnx.exists():
        print(f"ONNX model not found: {args.onnx}", file=sys.stderr)
        return 2

    if env.trtexec_path is None:
        print(
            "TensorRT Python bindings are present, but this stub currently requires `trtexec` "
            "for reproducible ONNX->FP16 engine builds.",
            file=sys.stderr,
        )
        return 2

    command = [
        env.trtexec_path,
        f"--onnx={args.onnx}",
        f"--saveEngine={engine}",
        "--fp16",
        f"--memPoolSize=workspace:{args.workspace_mb}",
    ]
    print(" ".join(command))
    if args.dry_run:
        return 0
    result = run_command(command)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
