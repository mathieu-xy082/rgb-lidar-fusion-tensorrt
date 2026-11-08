"""Streaming, content-verified camera/LiDAR inputs and native KITTI targets."""

import hashlib
from pathlib import Path

import torch

from .bev_batch import prepare_bev_fusion_batch
from .bev_checkpoint import validate_bev_fusion_split
from .calibration import parse_kitti_calibration_file
from .dataset import KittiObjectDepthDataset
from .kitti_labels import load_kitti_labels
from .lidar_center_targets import build_lidar_center_targets
from .lidar_center_training import _Frame
from .project_lidar import load_velodyne_bin


class BEVFusionFrameSource:
    """Keep only paths/manifest, not 512 decoded frames or padded pillar tensors.

    Inputs never depend on labels or target masks. Full scans feed both branches,
    without holdout. Missing labels are errors; no-Car frames are negatives and
    empty clouds remain valid inputs. Annotation geometry uses native calibration
    and image dimensions, not the resized camera input's calibration.
    """

    def __init__(self, data_root, *, model_config, split):
        self.root = Path(data_root)
        self.config = model_config
        self.split = validate_bev_fusion_split(split)
        self.dataset = KittiObjectDepthDataset(
            self.root, image_shape=model_config.image_shape, max_depth_m=model_config.max_depth_m,
            sample_limit=len(self.split["source_ids"]),
        )
        if [sample[0] for sample in self.dataset.samples] != self.split["source_ids"]:
            raise ValueError("camera dataset population/order differs from the saved split.")
        self.positions = {sample[0]: index for index, sample in enumerate(self.dataset.samples)}

    def _fingerprint(self, sample_id):
        result = {}
        for name, suffix in (("image_2", "png"), ("velodyne", "bin"), ("calib", "txt"), ("label_2", "txt")):
            with (self.root / name / f"{sample_id}.{suffix}").open("rb") as stream:
                result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result

    def read(self, sample_id):
        if sample_id not in self.positions:
            raise ValueError("sample_id is not in the recorded population.")
        expected = self.split["fingerprints"][sample_id]
        if self._fingerprint(sample_id) != expected:
            raise ValueError(f"dataset files changed before reading {sample_id}.")
        item = self.dataset[self.positions[sample_id]]
        raw = load_velodyne_bin(self.root / "velodyne" / f"{sample_id}.bin")
        inputs = prepare_bev_fusion_batch([item], {sample_id: raw}, config=self.config).inputs
        target = build_lidar_center_targets(
            load_kitti_labels(self.root / "label_2" / f"{sample_id}.txt"),
            calibration=parse_kitti_calibration_file(self.root / "calib" / f"{sample_id}.txt"),
            image_shape=tuple(item["meta"]["source_image_shape"]), grid=self.config.grid,
        )
        if self._fingerprint(sample_id) != expected:
            raise ValueError(f"dataset files changed after reading {sample_id}.")
        return _Frame(
            sample_id, inputs, {name: torch.from_numpy(getattr(target, name)[None]) for name in (
                "heatmap", "offset_xy", "center_mask", "supervision_mask",
            )}, target.centers_vehicle[:, :2].copy(), expected.copy(),
        )

    def frames(self, sample_ids):
        for sample_id in sample_ids:
            yield self.read(sample_id)
