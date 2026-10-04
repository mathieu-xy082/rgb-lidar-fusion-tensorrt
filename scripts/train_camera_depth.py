"""Train dense camera depth on downloaded KITTI Object frames."""

from __future__ import annotations

import argparse
import sys

from rgb_lidar_fusion.training import (
    load_training_config,
    run_kitti_camera_depth_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/training/kitti_camera_depth.yaml",
        help="Flat YAML training config path.",
    )
    parser.add_argument("--output-dir", help="Override the configured output directory.")
    parser.add_argument("--resume-from", help="Optional checkpoint path to resume from.")
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
        result = run_kitti_camera_depth_training(config)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    print(result.device_diagnostic)
    print(f"start_epoch={result.start_epoch}")
    print(f"epochs_completed={result.epochs_completed}")
    print(f"checkpoint={result.checkpoint_path}")
    for metric in result.metrics:
        line = (
            "metric "
            f"epoch={metric.epoch} step={metric.step} "
            f"loss={metric.loss:.6f} grad_norm={metric.grad_norm:.6f}"
        )
        if metric.val_loss is not None:
            line += (
                f" val_loss={metric.val_loss:.6f}"
                f" val_mae_m={metric.val_mae_m:.6f}"
                f" val_rmse_m={metric.val_rmse_m:.6f}"
                f" val_splat_mae_m={metric.val_splat_mae_m:.6f}"
                f" val_splat_rmse_m={metric.val_splat_rmse_m:.6f}"
                f" val_splat_coverage={metric.val_splat_coverage:.6f}"
                f" val_pixel_count={metric.val_pixel_count}"
            )
        print(line)


if __name__ == "__main__":
    main()
