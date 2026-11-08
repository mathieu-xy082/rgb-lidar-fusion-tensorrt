"""Train one explicitly selected matched BEV pilot variant."""

import argparse
import json
from pathlib import Path
import sys

from rgb_lidar_fusion.bev_pilot import BEVFusionPilotConfig, VARIANTS
from rgb_lidar_fusion.bev_trainval import run_bev_fusion_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/kitti_bev_fusion_pilot.json")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", help="Pilot root; variant name is appended automatically.")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    try:
        values = json.loads(Path(args.config).read_text())
        for name in ("device", "epochs", "output_dir"):
            if getattr(args, name) is not None:
                values[name] = getattr(args, name)
        run_bev_fusion_pilot(
            BEVFusionPilotConfig.from_mapping(values), variant=args.variant, resume_from=args.resume_from,
            emit=lambda event: print(json.dumps(event, allow_nan=False), flush=True),
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
