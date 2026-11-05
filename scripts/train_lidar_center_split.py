"""Train the LiDAR Car-center diagnostic on disjoint KITTI frame splits."""

import argparse
import json
from pathlib import Path
import sys

from rgb_lidar_fusion.lidar_center_training import LidarCenterTrainConfig
from rgb_lidar_fusion.lidar_center_trainval import run_lidar_center_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/kitti_lidar_center_512.json")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sample-count", type=int, help="Small CPU smoke subset; changes the saved split.")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    try:
        values = json.loads(Path(args.config).read_text())
        for name in ("device", "epochs", "sample_count", "output_dir"):
            if getattr(args, name) is not None:
                values[name] = getattr(args, name)
        run_lidar_center_training(
            LidarCenterTrainConfig.from_mapping(values), resume_from=args.resume_from,
            emit=lambda event: print(json.dumps(event, allow_nan=False), flush=True),
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
