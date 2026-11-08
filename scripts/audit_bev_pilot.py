"""Audit both fresh BEV pilot starts on CPU, without training or CUDA."""

import argparse
import json
from pathlib import Path
import sys

import torch

from rgb_lidar_fusion.bev_pilot import BEVFusionPilotConfig, prepare_bev_fusion_pilot_pair
from rgb_lidar_fusion.lidar_center_training import _write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/kitti_bev_fusion_pilot.json")
    parser.add_argument("--output", help="Optional JSON audit report, not a training checkpoint.")
    args = parser.parse_args()
    previous_threads = torch.get_num_threads()
    try:
        config = BEVFusionPilotConfig.from_mapping(json.loads(Path(args.config).read_text()))
        torch.set_num_threads(config.cpu_threads)
        pair = prepare_bev_fusion_pilot_pair(config)
        first = pair["fusion"]
        report = {
            "task": "bev_fusion_pilot_preparation", "cpu_only": True, "training_started": False,
            "config": config.serialized(), "split": first.split, "pretraining": first.pretraining,
            "variants": {name: {"initialization": start.initialization,
                                "output_dir": str(start.output_dir)} for name, start in pair.items()},
        }
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(output, report)
        print(json.dumps({
            "cpu_only": True, "training_started": False,
            "train_count": len(first.split["train_ids"]), "validation_count": len(first.split["validation_ids"]),
            "camera_provenance": first.pretraining["camera"]["provenance"],
            "variants": report["variants"], "report": str(Path(args.output).resolve()) if args.output else None,
        }, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    main()
