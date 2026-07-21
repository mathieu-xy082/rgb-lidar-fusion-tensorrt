"""Run the CPU-safe/GPU-ready baseline training smoke loop."""

from __future__ import annotations

import argparse
import sys

from rgb_lidar_fusion.training import load_training_config, run_synthetic_smoke_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/training/synthetic_smoke.yaml",
        help="Flat YAML training config path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the configured output directory for checkpoints/metrics.",
    )
    parser.add_argument(
        "--resume-from",
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override configured device selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.resume_from:
        config["resume_from"] = args.resume_from
    if args.device:
        config["device"] = args.device

    try:
        result = run_synthetic_smoke_training(config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(result.device_diagnostic)
    print(f"start_epoch={result.start_epoch}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint={result.checkpoint_path}")
    for metric in result.metrics:
        print(
            "metric "
            f"epoch={metric.epoch} step={metric.step} "
            f"loss={metric.loss:.6f} grad_norm={metric.grad_norm:.6f}"
        )


if __name__ == "__main__":
    main()
