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
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/kuixu/kitti_object_vis/master/data/object/training"
AVAILABLE_IDS = ("000000", "000001", "000002")
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


def validate_default_source_ids(sample_ids: list[str], base_url: str) -> None:
    if base_url != BASE_URL:
        return
    missing = [sample_id for sample_id in sample_ids if sample_id not in AVAILABLE_IDS]
    if not missing:
        return
    available = ",".join(AVAILABLE_IDS)
    requested = ",".join(sample_ids)
    raise SystemExit(
        "ERROR: the lightweight onboarding KITTI mirror only contains "
        f"these frame IDs: {available}\n"
        f"requested: {requested}\n"
        "Use `--count 3` or `--ids 000000,000001,000002`. "
        "For more frames, use a different --base-url that exposes matching "
        "image_2/ velodyne/ calib/ files."
    )


def download(url: str, output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"COMMAND: wget -O {output} {url}", flush=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"SKIP existing {output} bytes={output.stat().st_size} sha256={sha256(output)}")
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network-dependent CLI path
        if exc.code == 404:
            raise RuntimeError(
                f"missing remote file: {url}\n"
                "The selected source does not provide this sample/kind. "
                "Try a smaller --count or explicit --ids."
            ) from exc
        raise RuntimeError(f"failed to download {url}: HTTP {exc.code} {exc.reason}") from exc
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
    parser.add_argument("--list-available", action="store_true", help="List IDs available in the default lightweight mirror")
    args = parser.parse_args()

    if args.list_available:
        print(",".join(AVAILABLE_IDS))
        return 0

    sample_ids = parse_ids(args.ids, args.count)
    validate_default_source_ids(sample_ids, args.base_url)

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
