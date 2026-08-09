#!/usr/bin/env python3
"""Download KITTI object samples for onboarding.

The files are written under data/kitti/, which is ignored by Git.

Examples:
  # Lightweight GitHub mirror: 3 frames only.
  pdm run python docs/onboarding/scripts/download_kitti_samples.py --count 3

  # Official KITTI S3 archives: many frames, fetched file-by-file via HTTP ranges.
  pdm run python docs/onboarding/scripts/download_kitti_samples.py --source official --count 10
  pdm run python docs/onboarding/scripts/download_kitti_samples.py --source official --ids 000000,000003,000010
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

MIRROR_BASE_URL = "https://raw.githubusercontent.com/kuixu/kitti_object_vis/master/data/object/training"
MIRROR_AVAILABLE_IDS = ("000000", "000001", "000002")
OFFICIAL_ZIPS = {
    "image_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
}
KINDS = {
    "image_2": "png",
    "velodyne": "bin",
    "calib": "txt",
}


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


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


def http_head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def http_range(url: str, start: int, end_inclusive: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def find_eocd(tail: bytes) -> int:
    signature = b"PK\x05\x06"
    index = tail.rfind(signature)
    if index < 0:
        raise ValueError("ZIP EOCD signature not found")
    return index


def zip64_value(raw: int, values: list[int]) -> int:
    if raw == 0xFFFF_FFFF:
        if not values:
            raise ValueError("ZIP64 value missing")
        return values.pop(0)
    return raw


def parse_zip64_extra(extra: bytes) -> list[int]:
    values: list[int] = []
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        payload = extra[pos : pos + size]
        pos += size
        if header_id == 0x0001:
            for offset in range(0, len(payload), 8):
                if offset + 8 <= len(payload):
                    values.append(struct.unpack_from("<Q", payload, offset)[0])
    return values


def central_directory_location(url: str) -> tuple[int, int]:
    size = http_head_size(url)
    tail_size = min(size, 1024 * 128)
    tail_start = size - tail_size
    tail = http_range(url, tail_start, size - 1)
    eocd_pos = find_eocd(tail)
    eocd = tail[eocd_pos : eocd_pos + 22]
    (_sig, _disk, _cd_disk, _entries_disk, _entries, cd_size_32, cd_offset_32, _comment_len) = struct.unpack(
        "<IHHHHIIH", eocd
    )
    if cd_size_32 != 0xFFFF_FFFF and cd_offset_32 != 0xFFFF_FFFF:
        return cd_offset_32, cd_size_32

    locator_sig = b"PK\x06\x07"
    locator_pos = tail.rfind(locator_sig, 0, eocd_pos)
    if locator_pos < 0:
        raise ValueError("ZIP64 locator not found")
    locator = tail[locator_pos : locator_pos + 20]
    (_sig, _disk, zip64_eocd_offset, _total_disks) = struct.unpack("<IIQI", locator)
    zip64_header = http_range(url, zip64_eocd_offset, zip64_eocd_offset + 56 - 1)
    if zip64_header[:4] != b"PK\x06\x06":
        raise ValueError("ZIP64 EOCD signature not found")
    cd_size = struct.unpack_from("<Q", zip64_header, 40)[0]
    cd_offset = struct.unpack_from("<Q", zip64_header, 48)[0]
    return cd_offset, cd_size


def load_central_directory(url: str) -> dict[str, ZipEntry]:
    cd_offset, cd_size = central_directory_location(url)
    print(f"COMMAND: HTTP Range central-directory {url} bytes={cd_offset}-{cd_offset + cd_size - 1}", flush=True)
    data = http_range(url, cd_offset, cd_offset + cd_size - 1)
    entries: dict[str, ZipEntry] = {}
    pos = 0
    while pos + 46 <= len(data):
        if data[pos : pos + 4] != b"PK\x01\x02":
            break
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", data, pos)
        method = fields[4]
        compressed_size_32 = fields[8]
        uncompressed_size_32 = fields[9]
        name_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        local_offset_32 = fields[16]
        name_start = pos + 46
        extra_start = name_start + name_len
        comment_start = extra_start + extra_len
        name = data[name_start:extra_start].decode("utf-8")
        extra = data[extra_start:comment_start]
        zip64_values = parse_zip64_extra(extra)
        uncompressed_size = zip64_value(uncompressed_size_32, zip64_values)
        compressed_size = zip64_value(compressed_size_32, zip64_values)
        local_header_offset = zip64_value(local_offset_32, zip64_values)
        entries[name] = ZipEntry(name, method, compressed_size, uncompressed_size, local_header_offset)
        pos = comment_start + comment_len
    return entries


def extract_zip_entry(url: str, entry: ZipEntry, output: Path, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"COMMAND: extract {entry.name} from {url} -> {output}", flush=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        print(f"SKIP existing {output} bytes={output.stat().st_size} sha256={sha256(output)}")
        return
    header = http_range(url, entry.local_header_offset, entry.local_header_offset + 30 - 1)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"local header not found for {entry.name}")
    fields = struct.unpack_from("<IHHHHHIIIHH", header, 0)
    name_len = fields[9]
    extra_len = fields[10]
    data_start = entry.local_header_offset + 30 + name_len + extra_len
    compressed = http_range(url, data_start, data_start + entry.compressed_size - 1)
    if entry.method == 0:
        payload = compressed
    elif entry.method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported ZIP compression method {entry.method} for {entry.name}")
    if len(payload) != entry.uncompressed_size:
        raise RuntimeError(f"size mismatch for {entry.name}: got {len(payload)}, expected {entry.uncompressed_size}")
    output.write_bytes(payload)
    print(f"WROTE {output} bytes={len(payload)} sha256={sha256(output)}")


def validate_mirror_ids(sample_ids: list[str]) -> None:
    missing = [sample_id for sample_id in sample_ids if sample_id not in MIRROR_AVAILABLE_IDS]
    if not missing:
        return
    available = ",".join(MIRROR_AVAILABLE_IDS)
    requested = ",".join(sample_ids)
    raise SystemExit(
        "ERROR: the lightweight onboarding KITTI mirror only contains "
        f"these frame IDs: {available}\n"
        f"requested: {requested}\n"
        "Use `--count 3`, `--ids 000000,000001,000002`, or add `--source official` "
        "to fetch arbitrary KITTI object frames from the official S3 archives."
    )


def download_mirror(sample_ids: list[str], output_root: Path, base_url: str, force: bool) -> None:
    validate_mirror_ids(sample_ids)
    for sample_id in sample_ids:
        print(f"\n## sample {sample_id}", flush=True)
        for kind, ext in KINDS.items():
            url = f"{base_url}/{kind}/{sample_id}.{ext}"
            output = output_root / kind / f"{sample_id}.{ext}"
            output.parent.mkdir(parents=True, exist_ok=True)
            print(f"COMMAND: wget -O {output} {url}", flush=True)
            if output.exists() and output.stat().st_size > 0 and not force:
                print(f"SKIP existing {output} bytes={output.stat().st_size} sha256={sha256(output)}")
                continue
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
            except urllib.error.HTTPError as exc:  # pragma: no cover - network-dependent CLI path
                raise RuntimeError(f"failed to download {url}: HTTP {exc.code} {exc.reason}") from exc
            output.write_bytes(payload)
            print(f"WROTE {output} bytes={len(payload)} sha256={sha256(output)}")


def download_official(sample_ids: list[str], output_root: Path, force: bool) -> None:
    directories: dict[str, dict[str, ZipEntry]] = {}
    for kind, url in OFFICIAL_ZIPS.items():
        directories[kind] = load_central_directory(url)
    for sample_id in sample_ids:
        print(f"\n## sample {sample_id}", flush=True)
        for kind, ext in KINDS.items():
            entry_name = f"training/{kind}/{sample_id}.{ext}"
            entries = directories[kind]
            if entry_name not in entries:
                raise SystemExit(f"ERROR: {entry_name} not found in official KITTI {kind} archive")
            output = output_root / kind / f"{sample_id}.{ext}"
            extract_zip_entry(OFFICIAL_ZIPS[kind], entries[entry_name], output, force=force)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="Comma-separated KITTI frame IDs, e.g. 000000,000001")
    parser.add_argument("--count", type=int, default=1, help="Download IDs 000000..count-1 when --ids is absent")
    parser.add_argument("--output-root", type=Path, default=Path("data/kitti/training"))
    parser.add_argument("--source", choices=("mirror", "official"), default="mirror")
    parser.add_argument("--base-url", default=MIRROR_BASE_URL, help="Mirror base URL; used only with --source mirror")
    parser.add_argument("--force", action="store_true", help="Re-download files even if present")
    parser.add_argument("--list-available", action="store_true", help="List IDs available in the default lightweight mirror")
    args = parser.parse_args()

    if args.list_available:
        if args.source == "mirror":
            print(",".join(MIRROR_AVAILABLE_IDS))
        else:
            print("official KITTI object training IDs: 000000..007480")
        return 0

    sample_ids = parse_ids(args.ids, args.count)
    print(f"source={args.source}", flush=True)
    print(f"sample_ids={','.join(sample_ids)}", flush=True)
    print(f"output_root={args.output_root}", flush=True)

    if args.source == "mirror":
        download_mirror(sample_ids, args.output_root, args.base_url, force=args.force)
    else:
        download_official(sample_ids, args.output_root, force=args.force)

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
