"""Matched BEV pilot configuration and fresh, audited CPU initialization.

This preparation increment does not train, select a device, create an optimizer
or write experiment directories. The runner will consume these same starts.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import torch

from .bev_checkpoint import initialize_bev_fusion_encoders, validate_bev_fusion_split
from .bev_model import BEVFusionConfig, BEVFusionDiagnostic
from .dataset import KittiObjectDepthDataset
from .lidar_center_training import LidarCenterConfig


VARIANTS = ("lidar_only", "fusion")
_NEW_BLOCKS = ("fusion.", "backbone.", "head.")


@dataclass(frozen=True)
class BEVFusionPilotConfig:
    """One protocol for both variants; membership comes from a saved split."""

    data_root: str = "data/kitti/training"
    output_dir: str = "results/training/kitti_bev_fusion_pilot_512"
    split_path: str = "results/training/kitti_lidar_center_512/split.json"
    camera_checkpoint: str = "results/training/kitti_camera_depth_block_dropout_512_provenance/checkpoints/latest.pt"
    lidar_checkpoint: str = "results/training/kitti_lidar_center_512/checkpoints/best.pt"
    model_config: BEVFusionConfig = BEVFusionConfig()
    sample_count: int = 512
    epochs: int = 10
    learning_rate: float = .001
    seed: int = 13
    cpu_threads: int = 2
    device: str = "cpu"
    minimum_free_memory_mib: int = 2048
    score_threshold: float = .3
    top_k: int = 50
    match_radius_m: float = 1.
    max_grad_norm: float = 10.
    log_every_steps: int = 50

    def __post_init__(self):
        if (not isinstance(self.model_config, BEVFusionConfig)
                or not self.model_config.freeze_encoders or not self.model_config.camera_enabled):
            raise ValueError("the pilot requires a camera-enabled base config and frozen encoders.")
        for name in ("split_path", "camera_checkpoint", "lidar_checkpoint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path string.")
        for name, minimum in (("sample_count", 2), ("log_every_steps", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        # Reuse the existing schedule/device/matching validation, not a new policy.
        self.center_config()

    def center_config(self):
        names = ("data_root", "output_dir", "epochs", "learning_rate", "seed", "cpu_threads",
                 "device", "minimum_free_memory_mib", "score_threshold", "top_k",
                 "match_radius_m", "max_grad_norm")
        return LidarCenterConfig(
            **{name: getattr(self, name) for name in names}, eval_every=1,
            grid=self.model_config.grid, channels=self.model_config.lidar_channels,
            max_points_per_pillar=self.model_config.max_points_per_pillar,
        )

    def model_for_variant(self, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}.")
        return replace(self.model_config, camera_enabled=variant == "fusion")

    def output_for_variant(self, variant: str):
        self.model_for_variant(variant)
        return Path(self.output_dir).resolve() / variant

    def serialized(self):
        values = asdict(self)
        for name in ("data_root", "output_dir", "split_path", "camera_checkpoint", "lidar_checkpoint"):
            values[name] = str(Path(getattr(self, name)).resolve())
        return json.loads(json.dumps(values, allow_nan=False))

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        try:
            if "model_config" in values:
                values["model_config"] = BEVFusionConfig.from_mapping(values["model_config"])
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid BEV pilot configuration: {exc}") from exc


@dataclass(frozen=True)
class BEVFusionPilotStart:
    variant: str
    model: BEVFusionDiagnostic
    split: dict
    pretraining: dict
    initialization: dict
    output_dir: Path


def read_bev_pilot_split(config: BEVFusionPilotConfig) -> dict:
    """Verify saved IDs/order/content without rerolling a random split."""

    split = json.loads(Path(config.split_path).read_text())
    split["data_root"] = str(Path(config.data_root).resolve())
    split = validate_bev_fusion_split(split, verify_files=True)
    if len(split["source_ids"]) != config.sample_count:
        raise ValueError("saved split population differs from sample_count.")
    dataset = KittiObjectDepthDataset(
        config.data_root, image_shape=config.model_config.image_shape,
        max_depth_m=config.model_config.max_depth_m, sample_limit=config.sample_count,
    )
    if [sample[0] for sample in dataset.samples] != split["source_ids"]:
        raise ValueError("camera dataset population/order differs from the saved split.")
    return split


def bev_pilot_initial_state_sha256(model: BEVFusionDiagnostic) -> str:
    """Fingerprint the actual fresh fusion/shared/head state, including structure."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(_NEW_BLOCKS):
            value = value.detach().cpu().contiguous()
            header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
            digest.update(header.encode("ascii") + b"\n")
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prepare_bev_fusion_pilot(config: BEVFusionPilotConfig, *, variant: str) -> BEVFusionPilotStart:
    """Construct a new CPU model and import only normally audited source encoders.

    Even the LiDAR-only control audits the camera source for a matched lineage,
    but its forward skips the camera encoder and produces zero camera features,
    count/mass and support. Raw LiDAR/context and all new blocks stay present.
    No four-frame fused checkpoint or source optimizer/head is accepted here.
    """

    model_config = config.model_for_variant(variant)
    split = read_bev_pilot_split(config)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(config.seed)
        model = BEVFusionDiagnostic(model_config).eval()
    pretraining = initialize_bev_fusion_encoders(
        model, camera_checkpoint=config.camera_checkpoint,
        lidar_checkpoint=config.lidar_checkpoint, split=split,
    )
    if pretraining["camera"]["provenance"] != "recorded_at_training":
        raise ValueError("pilot requires recorded camera provenance, without legacy exceptions.")
    for name in ("camera", "lidar"):
        for partition in ("train", "validation"):
            if pretraining[name][partition + "_ids"] != split[partition + "_ids"]:
                raise ValueError(f"{name} source {partition} membership differs from the canonical split.")
    initialization = {
        "policy": "fresh fusion/backbone/head; audited frozen source encoders; no overfit warm start",
        "seed": config.seed,
        "new_blocks_sha256": bev_pilot_initial_state_sha256(model),
        "new_blocks": ["fusion", "backbone", "head"],
        "camera_enabled": model.config.camera_enabled,
        "torch_version": str(torch.__version__),
    }
    return BEVFusionPilotStart(variant, model, split, pretraining, initialization,
                               config.output_for_variant(variant))


def prepare_bev_fusion_pilot_pair(config: BEVFusionPilotConfig) -> dict[str, BEVFusionPilotStart]:
    """Read-only CPU control: verify identical initialization and source lineage."""

    starts = {name: prepare_bev_fusion_pilot(config, variant=name) for name in VARIANTS}
    first, second = (starts[name] for name in VARIANTS)
    if (first.split != second.split or first.pretraining != second.pretraining
            or first.initialization["new_blocks_sha256"] != second.initialization["new_blocks_sha256"]):
        raise ValueError("paired pilot initialization or source lineage changed between variants.")
    return starts
