"""Lightweight KITTI-style dataset utilities without a PyTorch dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import CameraCalibration
from .project_lidar import LIDAR_MAP_CHANNELS, build_sparse_lidar_maps, project_lidar_to_image


class KittiSparseLidarDataset:
    """Load RGB images and projected sparse LiDAR maps from a small manifest.

    The class intentionally implements only the Python sequence protocol so it is
    usable in tests and smoke checks without installing PyTorch. A future
    `torch.utils.data.Dataset` adapter can wrap this class when the ML dependency
    group is enabled.
    """

    def __init__(
        self,
        root: str | Path,
        manifest_name: str = "manifest.json",
        max_depth_m: float = 80.0,
    ) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / manifest_name
        self.max_depth_m = max_depth_m
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if "samples" not in manifest:
            raise ValueError(f"Dataset manifest {self.manifest_path} must define a samples list.")
        self.samples: list[dict[str, Any]] = list(manifest["samples"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        self._validate_sample(sample, index)
        image_path = self.root / sample["image"]
        lidar_path = self.root / sample["lidar"]
        self._ensure_exists(image_path)
        self._ensure_exists(lidar_path)

        image = self._load_image(image_path)
        points = np.load(lidar_path).astype(np.float32)
        calibration = self._load_calibration(sample["calibration"])
        image_shape = (int(image.shape[1]), int(image.shape[2]))
        projected = project_lidar_to_image(points, calibration, image_shape=image_shape)
        lidar_maps = build_sparse_lidar_maps(
            projected,
            image_shape=image_shape,
            max_depth_m=self.max_depth_m,
        )

        return {
            "image": image,
            "lidar_maps": lidar_maps,
            "target": sample.get("target", {}),
            "meta": {
                "sample_id": sample.get("id"),
                "image_shape": [image_shape[0], image_shape[1]],
                "image_path": sample["image"],
                "lidar_path": sample["lidar"],
                "lidar_map_channels": list(LIDAR_MAP_CHANNELS),
            },
        }

    @staticmethod
    def _validate_sample(sample: dict[str, Any], index: int) -> None:
        required = ("id", "image", "lidar", "calibration")
        missing = [field for field in required if field not in sample]
        if missing:
            sample_id = sample.get("id", f"at index {index}")
            raise ValueError(f"Manifest sample {sample_id} is missing required field(s): {', '.join(missing)}")

    @staticmethod
    def _ensure_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        image = np.load(path)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"RGB image array must have shape [H, W, 3], got {image.shape}.")
        return np.moveaxis(image.astype(np.float32) / 255.0, 2, 0)

    @staticmethod
    def _load_calibration(payload: dict[str, Any]) -> CameraCalibration:
        return CameraCalibration(
            k=np.asarray(payload["k"], dtype=np.float32),
            t_camera_from_vehicle=np.asarray(payload["t_camera_from_vehicle"], dtype=np.float32),
        )
