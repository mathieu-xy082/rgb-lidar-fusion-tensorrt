#!/usr/bin/env python3
"""Guarded TensorRT benchmark entrypoint and schema reporter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rgb_lidar_fusion.tensorrt_runtime import (
    benchmark_schema,
    print_json,
    require_tensorrt_available,
    runtime_diagnostic,
    validate_engine_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, help="Input TensorRT .engine/.plan file")
    parser.add_argument("--schema", action="store_true", help="Print benchmark JSON schema and exit")
    parser.add_argument("--detect", action="store_true", help="Print detected runtime capabilities and exit")
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--measured-runs", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.schema:
        print_json(benchmark_schema())
        return 0
    if args.detect:
        print_json(runtime_diagnostic())
        return 0
    if args.engine is None:
        print("--engine is required unless --schema or --detect is used", file=sys.stderr)
        return 2
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
        "TensorRT benchmark execution is blocked until ONNX/export integration "
        "defines real input tensors. Metrics to emit: p50/p95/mean/min/max latency and FPS."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
