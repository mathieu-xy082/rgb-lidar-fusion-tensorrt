"""Strict KITTI object-training labels and explicit camera-to-LiDAR box geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import CameraCalibration


KITTI_CLASSES = {"Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"}


@dataclass(frozen=True)
class KittiLabel:
    category: str
    bbox_xyxy: tuple[float, float, float, float]
    truncation: float
    occlusion: int
    dimensions_hwl: tuple[float, float, float] | None
    bottom_center_camera: tuple[float, float, float] | None
    rotation_y: float | None


def parse_kitti_labels(text: str) -> list[KittiLabel]:
    """Parse 15-column ground truth, not scored detection-result files.

    DontCare has a valid 2D rectangle but sentinel 3D fields: these are stored
    as None, never transformed into metric supervision. Empty files are valid.
    """

    labels = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 15 or fields[0] not in KITTI_CLASSES:
            raise ValueError(f"label line {number}: expected a known KITTI class and 15 columns.")
        try:
            values = np.asarray(fields[1:], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"label line {number}: non-numeric annotation.") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"label line {number}: annotation must be finite.")
        bbox = tuple(float(v) for v in values[3:7])
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"label line {number}: invalid 2D rectangle.")
        category = fields[0]
        if category == "DontCare":
            labels.append(KittiLabel(category, bbox, float(values[0]), int(values[1]), None, None, None))
            continue
        if not 0 <= values[0] <= 1 or values[1] not in (0, 1, 2, 3):
            raise ValueError(f"label line {number}: invalid truncation or occlusion.")
        # A partially visible long vehicle can have its center behind the camera.
        if np.any(values[7:10] <= 0):
            raise ValueError(f"label line {number}: box dimensions must be positive.")
        labels.append(KittiLabel(
            category, bbox, float(values[0]), int(values[1]),
            tuple(float(v) for v in values[7:10]),
            tuple(float(v) for v in values[10:13]), float(values[13]),
        ))
    return labels


def load_kitti_labels(path: str | Path) -> list[KittiLabel]:
    try:
        return parse_kitti_labels(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def label_box_vehicle(label: KittiLabel, calibration: CameraCalibration) -> tuple[np.ndarray, np.ndarray]:
    """Return geometric center [3] and eight corners [8,3] in Velodyne meters.

    KITTI location is bottom-center in rectified reference-camera coordinates.
    Move up by h/2 along camera -y BEFORE inverting T_vehicle_to_camera.
    P2's fourth column belongs to image projection, not this frame conversion.
    Corners 0..3 form the bottom rectangle; 4..7 form the top rectangle.
    """

    if label.dimensions_hwl is None or label.bottom_center_camera is None or label.rotation_y is None:
        raise ValueError("DontCare has no usable 3D box.")
    h, w, length = label.dimensions_hwl
    bottom = np.asarray(label.bottom_center_camera, dtype=np.float64)
    center = bottom + [0., -h / 2, 0.]
    local = np.array([
        [length / 2, 0, w / 2], [length / 2, 0, -w / 2],
        [-length / 2, 0, -w / 2], [-length / 2, 0, w / 2],
        [length / 2, -h, w / 2], [length / 2, -h, -w / 2],
        [-length / 2, -h, -w / 2], [-length / 2, -h, w / 2],
    ])
    c, s = np.cos(label.rotation_y), np.sin(label.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    camera = np.vstack([center, local @ rotation.T + bottom])
    transform = np.asarray(calibration.t_vehicle_to_camera, dtype=np.float64)
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("calibration transform must be finite and affine.")
    homogeneous = np.column_stack([camera, np.ones(len(camera))])
    vehicle = np.linalg.solve(transform, homogeneous.T).T[:, :3]
    return vehicle[0].astype(np.float32), vehicle[1:].astype(np.float32)
