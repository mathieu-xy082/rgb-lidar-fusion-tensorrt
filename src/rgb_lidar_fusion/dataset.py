"""Lightweight KITTI-style dataset utilities without a PyTorch dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import CameraCalibration, parse_kitti_calibration_file
from .lidar_splatting import SPLATTED_LIDAR_MAP_CHANNELS, build_splatted_lidar_maps
from .project_lidar import (
    LIDAR_MAP_CHANNELS,
    build_sparse_lidar_maps,
    load_velodyne_bin,
    project_lidar_to_image,
)


class KittiObjectDepthDataset:
    """Load downloaded KITTI Object RGB/Velodyne/calibration frame triplets."""

    def __init__(
        self,
        root: str | Path,
        *,
        image_shape: tuple[int, int],
        max_depth_m: float = 80.0,
        sample_limit: int | None = None,
    ) -> None:
        height, width = image_shape
        if height <= 0 or width <= 0:
            raise ValueError("image_shape dimensions must be positive.")
        if max_depth_m <= 0.0:
            raise ValueError("max_depth_m must be positive.")
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit must be positive when provided.")

        self.root = Path(root)
        self.image_shape = (int(height), int(width))
        self.max_depth_m = float(max_depth_m)
        image_dir = self.root / "image_2"
        velodyne_dir = self.root / "velodyne"
        calibration_dir = self.root / "calib"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"KITTI image directory does not exist: {image_dir}")

        image_paths = sorted(image_dir.glob("*.png"))
        if not image_paths:
            raise ValueError(f"KITTI image directory contains no PNG frames: {image_dir}")
        if sample_limit is not None:
            if len(image_paths) < sample_limit:
                raise ValueError(
                    f"KITTI dataset provides {len(image_paths)} frames, "
                    f"but sample_limit={sample_limit} was requested."
                )
            image_paths = image_paths[:sample_limit]

        self.samples: list[tuple[str, Path, Path, Path]] = []
        for image_path in image_paths:
            sample_id = image_path.stem
            velodyne_path = velodyne_dir / f"{sample_id}.bin"
            calibration_path = calibration_dir / f"{sample_id}.txt"
            missing = [
                path
                for path in (velodyne_path, calibration_path)
                if not path.is_file()
            ]
            if missing:
                missing_paths = ", ".join(str(path) for path in missing)
                raise FileNotFoundError(
                    f"KITTI frame {sample_id} is incomplete; missing: {missing_paths}"
                )
            self.samples.append(
                (sample_id, image_path, velodyne_path, calibration_path)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index >= len(self.samples) or index < -len(self.samples):
            raise IndexError(
                f"Dataset index {index} out of range for {len(self.samples)} samples."
            )
        sample_id, image_path, velodyne_path, calibration_path = self.samples[index]
        image, source_shape = self._load_and_resize_rgb(image_path)
        calibration = parse_kitti_calibration_file(calibration_path)
        resized_calibration = self._resize_calibration(
            calibration,
            source_shape=source_shape,
            target_shape=self.image_shape,
        )
        projected = project_lidar_to_image(
            load_velodyne_bin(velodyne_path),
            resized_calibration,
            image_shape=self.image_shape,
        )
        sparse_lidar_maps = build_sparse_lidar_maps(
            projected,
            image_shape=self.image_shape,
            max_depth_m=self.max_depth_m,
        )
        return {
            "image": image,
            "lidar_maps": sparse_lidar_maps,
            "target": {},
            "meta": {
                "sample_id": sample_id,
                "image_shape": list(self.image_shape),
                "source_image_shape": list(source_shape),
                "image_path": str(image_path),
                "lidar_path": str(velodyne_path),
                "calibration_path": str(calibration_path),
                "lidar_representation": "sparse",
                "lidar_map_channels": list(LIDAR_MAP_CHANNELS),
            },
        }

    def _load_and_resize_rgb(
        self,
        path: Path,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on optional group.
            raise ImportError(
                "KITTI PNG loading requires the ml dependency group; "
                "install with `pdm install -G ml`."
            ) from exc

        with Image.open(path) as image_file:
            rgb_image = image_file.convert("RGB")
            source_shape = (rgb_image.height, rgb_image.width)
            target_height, target_width = self.image_shape
            resized = rgb_image.resize(
                (target_width, target_height),
                resample=Image.Resampling.BILINEAR,
            )
            image = np.asarray(resized, dtype=np.float32) / 255.0
        return np.moveaxis(image, 2, 0), source_shape

    @staticmethod
    def _resize_calibration(
        calibration: CameraCalibration,
        *,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> CameraCalibration:
        source_height, source_width = source_shape
        target_height, target_width = target_shape
        scale_x = target_width / source_width
        scale_y = target_height / source_height

        intrinsic = calibration.k.copy()
        intrinsic[0] *= scale_x
        intrinsic[1] *= scale_y
        projection = None
        if calibration.projection_matrix is not None:
            projection = calibration.projection_matrix.copy()
            projection[0] *= scale_x
            projection[1] *= scale_y
        return CameraCalibration(
            k=intrinsic,
            t_camera_from_vehicle=calibration.t_camera_from_vehicle,
            projection_matrix=projection,
        )


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
        lidar_representation: str = "sparse",
    ) -> None:
        if lidar_representation not in {"sparse", "splatted"}:
            raise ValueError("lidar_representation must be 'sparse' or 'splatted'.")
        self.root = Path(root)
        self.manifest_path = self.root / manifest_name
        self.max_depth_m = max_depth_m
        self.lidar_representation = lidar_representation
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if "samples" not in manifest:
            raise ValueError(f"Dataset manifest {self.manifest_path} must define a samples list.")
        self.samples: list[dict[str, Any]] = list(manifest["samples"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index >= len(self.samples) or index < -len(self.samples):
            label = "sample" if len(self.samples) == 1 else "samples"
            raise IndexError(f"Dataset index {index} out of range for {len(self.samples)} {label}.")
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
        sparse_lidar_maps = build_sparse_lidar_maps(
            projected,
            image_shape=image_shape,
            max_depth_m=self.max_depth_m,
        )
        if self.lidar_representation == "splatted":
            lidar_maps = build_splatted_lidar_maps(sparse_lidar_maps)
            lidar_map_channels = list(SPLATTED_LIDAR_MAP_CHANNELS)
        else:
            lidar_maps = sparse_lidar_maps
            lidar_map_channels = list(LIDAR_MAP_CHANNELS)

        return {
            "image": image,
            "lidar_maps": lidar_maps,
            "target": sample.get("target", {}),
            "meta": {
                "sample_id": sample.get("id"),
                "image_shape": [image_shape[0], image_shape[1]],
                "image_path": sample["image"],
                "lidar_path": sample["lidar"],
                "lidar_representation": self.lidar_representation,
                "lidar_map_channels": lidar_map_channels,
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
