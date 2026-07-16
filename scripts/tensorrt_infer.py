#!/usr/bin/env python3
"""Guarded TensorRT inference entrypoint for a future serialized engine.

The implementation is intentionally conservative until ONNX export and engine
format are available. It validates runtime/artifact assumptions and exits with a
clear message when TensorRT is not installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rgb_lidar_fusion.tensorrt_runtime import require_tensorrt_available, validate_engine_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path, help="Input TensorRT .engine/.plan file")
    parser.add_argument("--input", required=False, type=Path, help="Optional input tensor/sample path for future inference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_engine_output_path(args.engine)
        require_tensorrt_available()
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not args.engine.exists():
        print(f"TensorRT engine not found: {args.engine}", file=sys.stderr)
        return 2
    print(
        "TensorRT runtime detected. Inference execution is blocked until the ONNX "
        "milestone defines input/output tensor names, shapes, and preprocessing."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
