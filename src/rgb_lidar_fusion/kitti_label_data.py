"""Download only selected official KITTI labels, then audit local correspondence."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import urllib.request
from zipfile import ZipFile

import numpy as np

from .calibration import parse_kitti_calibration_file
from .kitti_labels import label_box_vehicle, load_kitti_labels, parse_kitti_labels


LABEL_ARCHIVE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"


def install_kitti_labels(root: str | Path, archive: str | Path | None = None) -> dict:
    """Install only IDs with local images. Existing different labels are refused.

    ZIP CRCs are checked by zipfile; member names are never used as output paths.
    Archive SHA-256 is provenance, not an independently known official checksum.
    No image, cloud or calibration download/write is performed.
    """

    root = Path(root)
    ids = sorted(path.stem for path in (root / "image_2").glob("*.png"))
    if not ids or any(len(value) != 6 or not value.isascii() or not value.isdigit() for value in ids):
        raise ValueError("root must contain six-digit KITTI image IDs.")
    archive = Path(archive) if archive is not None else root.parent / "downloads" / "data_object_label_2.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        temporary = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(LABEL_ARCHIVE_URL, timeout=60) as response:
            payload = response.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise ValueError("label archive exceeds the 10 MiB download bound.")
        with temporary.open("xb") as output:
            output.write(payload)
        with ZipFile(temporary) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("label archive CRC verification failed.")
        temporary.replace(archive)
    with ZipFile(archive) as zipped:
        # Validate every selected payload and conflict before writing any labels.
        selected = []
        for sample_id in ids:
            payload = zipped.read(f"training/label_2/{sample_id}.txt")
            try:
                parse_kitti_labels(payload.decode("utf-8"))
            except ValueError as exc:
                raise ValueError(f"KITTI frame {sample_id}: {exc}") from exc
            target = root / "label_2" / f"{sample_id}.txt"
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"refusing to overwrite different existing annotation: {target}")
            selected.append((target, payload))
    (root / "label_2").mkdir(parents=True, exist_ok=True)
    for target, payload in selected:
        if not target.exists():
            with target.open("xb") as output:
                output.write(payload)
    return {"source": LABEL_ARCHIVE_URL, "archive": str(archive),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "installed_frames": len(ids)}


def audit_kitti_labels(root: str | Path, *, grid=None) -> dict:
    """Check all local image/cloud/calibration/annotation counterparts.

    This validates format and geometry consistency, not the truth/completeness
    of every human annotation. Optional grid validation also builds all targets.
    """

    from PIL import Image

    root = Path(root)
    images = sorted((root / "image_2").glob("*.png"))
    errors, warnings, counts = [], [], Counter()
    target_count = supervised_frames = 0
    if not images:
        errors.append("no KITTI PNG images found")
    for image in images:
        sample_id = image.stem
        try:
            cloud = root / "velodyne" / f"{sample_id}.bin"
            if not cloud.is_file() or cloud.stat().st_size == 0 or cloud.stat().st_size % 16:
                raise ValueError("missing/invalid XYZI cloud file size")
            calibration = parse_kitti_calibration_file(root / "calib" / f"{sample_id}.txt")
            labels = load_kitti_labels(root / "label_2" / f"{sample_id}.txt")
            with Image.open(image) as rgb:
                shape = (rgb.height, rgb.width)
            for label in labels:
                counts[label.category] += 1
                if label.category != "DontCare":
                    center, corners = label_box_vehicle(label, calibration)
                    if not np.isfinite(corners).all() or not np.isfinite(center).all():
                        raise ValueError("non-finite transformed box")
                left, top, right, bottom = label.bbox_xyxy
                if left < 0 or top < 0 or right > shape[1] or bottom > shape[0]:
                    warnings.append(f"{sample_id}: {label.category} rectangle extends outside the image")
            if grid is not None:
                from .lidar_center_targets import build_lidar_center_targets

                target = build_lidar_center_targets(labels, calibration=calibration, image_shape=shape, grid=grid)
                target_count += len(target.centers_vehicle)
                supervised_frames += int(target.supervision_mask.any())
        except (OSError, ValueError, np.linalg.LinAlgError) as exc:
            errors.append(f"{sample_id}: {exc}")
    return {"ready": bool(images) and not errors, "data_root": str(root.resolve()),
            "frame_count": len(images), "class_counts": dict(counts),
            "target_car_count": target_count if grid is not None else None,
            "frames_with_supervision": supervised_frames if grid is not None else None,
            "errors": errors, "warnings": warnings,
            "scope": "Structural/calibration checks, not proof of perfect human annotations or sequence-disjoint splits."}
