"""Assemble existing camera/LiDAR encoders and shared BEV center diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import torch
from torch import nn

from .bev_backbone import SharedBEVBackbone
from .bev_fusion import BEVFeatureFusion, FusedBEVFeatures
from .camera_bev import BEVGridConfig, CameraBEVFeatures, lift_camera_features_to_bev
from .camera_depth_model import CameraDepthModel
from .camera_depth_support import prepare_camera_depth_support
from .lidar_bev import LidarBEVContext, LidarBEVFeatures, PillarFeatureNet, scatter_pillar_features
from .lidar_center_model import CenterPrediction, LidarCenterHead


@dataclass(frozen=True)
class BEVFusionConfig:
    """Architecture and preprocessing only; runner/schedule settings stay separate."""

    grid: BEVGridConfig = BEVGridConfig((0., 50.), (-25., 25.), (-3., 3.), .2)
    image_shape: tuple[int, int] = (320, 1024)
    max_depth_m: float = 80.
    splat_radius_px: int = 2
    splat_sigma_px: float = 1.
    min_splat_confidence: float = 0.
    max_points_per_pillar: int = 64
    pillar_seed: int = 13
    lidar_channels: int = 32
    backbone_channels: tuple[int, int, int] = (32, 64, 96)
    freeze_encoders: bool = True
    camera_enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.grid, BEVGridConfig):
            raise ValueError("grid must be BEVGridConfig.")
        object.__setattr__(self, "grid", BEVGridConfig(
            tuple(self.grid.x_range_m), tuple(self.grid.y_range_m), tuple(self.grid.z_range_m),
            self.grid.resolution_m,
        ))
        for name, size in (("image_shape", 2), ("backbone_channels", 3)):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or len(values) != size or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values
            ):
                raise ValueError(f"{name} must contain {size} positive integers.")
            object.__setattr__(self, name, tuple(values))
        for name, minimum in (("splat_radius_px", 0), ("max_points_per_pillar", 1),
                              ("pillar_seed", 0), ("lidar_channels", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("max_depth_m", "splat_sigma_px", "min_splat_confidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.max_depth_m <= 0 or self.splat_sigma_px <= 0 or not 0 <= self.min_splat_confidence <= 1:
            raise ValueError("depth/sigma must be positive and min_splat_confidence must lie in [0,1].")
        if type(self.freeze_encoders) is not bool or type(self.camera_enabled) is not bool:
            raise ValueError("freeze_encoders and camera_enabled must be boolean.")

    def serialized(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_mapping(cls, values: dict) -> BEVFusionConfig:
        values = dict(values)
        try:
            if "grid" in values:
                values["grid"] = BEVGridConfig(**values["grid"])
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid BEV fusion configuration: {exc}") from exc


@dataclass(frozen=True)
class BEVFusionRepresentations:
    camera_bev: CameraBEVFeatures
    lidar_bev: LidarBEVFeatures
    fused_bev: FusedBEVFeatures
    shared_bev: FusedBEVFeatures


class BEVFusionDiagnostic(nn.Module):
    """Multimodal features plus an optional-by-use Car-center diagnostic head.

    Raw XYZI grouping and image projection/splatting remain CPU preprocessing.
    Forward consumes explicit same-device tensors, never labels, targets or
    supervision masks. encode() exposes representations without using the head.
    The camera depth head is retained for checkpoint compatibility, but is not
    evaluated or optimized: the lift uses splatted depth and heuristic weights.
    """

    def __init__(self, config: BEVFusionConfig | None = None) -> None:
        super().__init__()
        if config is not None and not isinstance(config, BEVFusionConfig):
            raise ValueError("config must be BEVFusionConfig.")
        self.config = config or BEVFusionConfig()
        self.camera = CameraDepthModel(lidar_mode="enriched")
        self.pfn = PillarFeatureNet(out_channels=self.config.lidar_channels)
        self.context = LidarBEVContext(channels=self.config.lidar_channels)
        self.fusion = BEVFeatureFusion(lidar_channels=self.config.lidar_channels,
                                       out_channels=self.config.backbone_channels[0])
        self.backbone = SharedBEVBackbone(channels=self.config.backbone_channels)
        self.head = LidarCenterHead(channels=self.config.backbone_channels[0])
        self.camera.requires_grad_(not self.config.freeze_encoders and self.config.camera_enabled)
        self.pfn.requires_grad_(not self.config.freeze_encoders)
        self.context.requires_grad_(not self.config.freeze_encoders)
        self.camera.depth_head.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_encoders:
            for module in (self.camera, self.pfn, self.context):
                module.eval()
        elif not self.config.camera_enabled:
            self.camera.eval()
        return self

    def encode(
        self, rgb: torch.Tensor, lidar_maps: torch.Tensor, *,
        camera_projection: torch.Tensor, t_vehicle_to_camera: torch.Tensor,
        pillar_features: torch.Tensor, point_mask: torch.Tensor,
        indices: torch.Tensor, point_count: torch.Tensor,
    ) -> BEVFusionRepresentations:
        """Expose camera, contextual LiDAR, fused and shared features in one pass."""

        parameter = next(self.parameters())
        if rgb.ndim != 4 or rgb.shape[0] < 1 or rgb.shape[1:] != (3, *self.config.image_shape):
            raise ValueError("rgb must have non-empty shape [B, 3, *config.image_shape].")
        if lidar_maps.shape != (rgb.shape[0], 8, *self.config.image_shape):
            raise ValueError("lidar_maps must have shape [B, 8, *config.image_shape].")
        for name, tensor, shape in (
            ("rgb", rgb, rgb.shape), ("lidar_maps", lidar_maps, lidar_maps.shape),
            ("camera_projection", camera_projection, (rgb.shape[0], 3, 4)),
            ("t_vehicle_to_camera", t_vehicle_to_camera, (rgb.shape[0], 4, 4)),
        ):
            if tensor.shape != shape or tensor.dtype != parameter.dtype or tensor.device != parameter.device:
                raise ValueError(f"{name} must match its shape and the model dtype/device.")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite.")
        if point_count.shape != (rgb.shape[0], 1, *self.config.grid.shape):
            raise ValueError("point_count must match the RGB batch and configured BEV grid.")
        if pillar_features.ndim != 3 or pillar_features.shape[1:] != (self.config.max_points_per_pillar, 9):
            raise ValueError("pillar_features must match the configured point cap and nine features.")

        if self.config.camera_enabled:
            image_features = self.camera.forward_features(rgb, lidar_maps)
            support = prepare_camera_depth_support(
                lidar_maps, source="splatted", max_depth_m=self.config.max_depth_m,
                max_valid_depth_m=self.config.max_depth_m,
                min_splat_confidence=self.config.min_splat_confidence,
            )
            camera_bev = lift_camera_features_to_bev(
                image_features, depth_support=support, camera_projection=camera_projection,
                t_vehicle_to_camera=t_vehicle_to_camera, grid=self.config.grid,
                weight_by_splat_confidence=True,
            )
        else:
            shape = (rgb.shape[0], 1, *self.config.grid.shape)
            camera_bev = CameraBEVFeatures(
                rgb.new_zeros(rgb.shape[0], 32, *self.config.grid.shape),
                torch.zeros(shape, dtype=torch.int64, device=rgb.device), rgb.new_zeros(shape),
                torch.zeros_like(lidar_maps[:, :1], dtype=torch.bool),
            )
        lidar_bev = self.context(scatter_pillar_features(
            self.pfn(pillar_features, point_mask), indices=indices, point_count=point_count,
        ))
        fused = self.fusion(camera_bev, lidar_bev, camera_grid=self.config.grid,
                             lidar_grid=self.config.grid)
        shared = self.backbone(fused)
        return BEVFusionRepresentations(camera_bev, lidar_bev, fused, shared)

    def forward(self, *args, **kwargs) -> CenterPrediction:
        return self.head(self.encode(*args, **kwargs).shared_bev.features)
