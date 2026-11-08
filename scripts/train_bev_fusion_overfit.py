"""Run the explicit frozen-encoder BEV fusion memorization diagnostic."""

import argparse
import json
from pathlib import Path
import sys

from rgb_lidar_fusion.bev_training import BEVFusionOverfitConfig, run_bev_fusion_overfit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/kitti_bev_fusion_overfit.json")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    try:
        values = json.loads(Path(args.config).read_text())
        for name in ("device", "epochs", "output_dir"):
            if getattr(args, name) is not None:
                values[name] = getattr(args, name)
        run_bev_fusion_overfit(
            BEVFusionOverfitConfig.from_mapping(values), resume_from=args.resume_from,
            emit=lambda event: print(json.dumps(event, allow_nan=False), flush=True),
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
