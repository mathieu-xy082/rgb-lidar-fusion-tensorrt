"""Fetch labels for existing KITTI frames and audit the optional Car-center targets."""

import argparse
import json
from pathlib import Path

from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.kitti_label_data import audit_kitti_labels, install_kitti_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/kitti/training"))
    parser.add_argument("--download", action="store_true", help="Explicit network opt-in; existing modalities are unchanged.")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/lidar_bev/annotation_audit.json"))
    args = parser.parse_args()
    source = install_kitti_labels(args.root, args.archive) if args.download else None
    report = audit_kitti_labels(args.root, grid=BEVGridConfig((0., 50.), (-25., 25.), (-3., 3.), .2))
    report["download"] = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()
