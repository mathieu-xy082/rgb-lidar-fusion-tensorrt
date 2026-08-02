#!/usr/bin/env python3
"""Download a lightweight KITTI object mini-sequence for onboarding.

The files are written under data/kitti/, which is ignored by Git.

Examples:
  pdm run python docs/onboarding/scripts/download_kitti_samples.py --count 3
  pdm run python docs/onboarding/scripts/download_kitti_samples.py --ids 000000,000001
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/kuixu/kitti_object_vis/master/data/object/training"
KINDS = {
    "image_2": "png",
    "velodyne": "bin",
    "calib": "txt",
}


def parse_ids(ids: str | None, count: int) -> list[str]:
    if ids:
        return [item.strip() for item in ids.split(",") if item.strip()]
    return [f"{idx:06d}" for idx in range(count)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"COMMAND: wget -O {output} {url}", flush=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"SKIP existing {output} bytes={output.stat().st_size} sha256={sha256(output)}")
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except Exception as exc:  # pragma: no cover - network-dependent CLI path
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    output.write_bytes(payload)
    print(f"WROTE {output} bytes={len(payload)} sha256={sha256(output)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="Comma-separated KITTI frame IDs, e.g. 000000,000001")
    parser.add_argument("--count", type=int, default=1, help="Download IDs 000000..count-1 when --ids is absent")
    parser.add_argument("--output-root", type=Path, default=Path("data/kitti/training"))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--force", action="store_true", help="Re-download files even if present")
    args = parser.parse_args()

    sample_ids = parse_ids(args.ids, args.count)
    print(f"sample_ids={','.join(sample_ids)}", flush=True)
    print(f"output_root={args.output_root}", flush=True)

    for sample_id in sample_ids:
        print(f"\n## sample {sample_id}", flush=True)
        for kind, ext in KINDS.items():
            url = f"{args.base_url}/{kind}/{sample_id}.{ext}"
            output = args.output_root / kind / f"{sample_id}.{ext}"
            download(url, output, force=args.force)

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
