"""Full-resolution camera features lifted into a metric Velodyne-frame grid."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .camera_depth_support import CameraDepthSupport
from .camera_geometry import backproject_depth_to_vehicle


@dataclass(frozen=True)
class BEVGridConfig:
    """Half-open metric bounds [min, max) in the current vehicle/LiDAR frame.

    KITTI uses x forward, y left, z up, with no gravity/world alignment.
    Rows increase with x; columns increase with y. Cell [0, 0] starts at
    (x_min, y_min); its center is offset by half a resolution on both axes.
    z_range_m filters height relative to the LiDAR origin, not the ground.
    XY extents must be integer multiples of the isotropic resolution.
    """

    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    z_range_m: tuple[float, float]
    resolution_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("resolution_m must be finite and positive.")
        for name in ("x_range_m", "y_range_m", "z_range_m"):
            bounds = getattr(self, name)
            if (
                len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} must contain finite min < max.")
        for bounds in (self.x_range_m, self.y_range_m):
            cells = (bounds[1] - bounds[0]) / self.resolution_m
            if (
                not math.isfinite(cells)
                or not math.isclose(cells, round(cells), rel_tol=0, abs_tol=1e-7)
                or round(cells) < 1
            ):
                raise ValueError("XY extents must be positive integer multiples of resolution_m.")

    @property
    def shape(self) -> tuple[int, int]:
        """Return (Hbev, Wbev), corresponding to (x cells, y cells)."""

        return tuple(
            round((bounds[1] - bounds[0]) / self.resolution_m)
            for bounds in (self.x_range_m, self.y_range_m)
        )


@dataclass(frozen=True)
class CameraBEVFeatures:
    """Mean features plus evidence maps; empty cells contain zeros.

    features is [B, C, Hbev, Wbev]; pixel_count and weight_sum are
    [B, 1, Hbev, Wbev]. pixel_count > 0 distinguishes occupied cells from
    genuine zero feature vectors. pixel_mask [B, 1, H, W] identifies the
    pixels actually accumulated, after validity, finite-feature, XYZ and
    height/bounds filtering. Counts are contributions, not independent rays
    or confidence: neighboring splats may come from the same LiDAR return.
    """

    features: torch.Tensor
    pixel_count: torch.Tensor
    weight_sum: torch.Tensor
    pixel_mask: torch.Tensor


def lift_camera_features_to_bev(
    image_features: torch.Tensor,
    *,
    depth_support: CameraDepthSupport,
    camera_projection: torch.Tensor,
    t_vehicle_to_camera: torch.Tensor,
    grid: BEVGridConfig,
    weight_by_splat_confidence: bool = False,
) -> CameraBEVFeatures:
    """Backproject valid pixels, then aggregate image features by cell mean.

    No new learned parameters or implicit feature/depth resize is introduced.
    All tensors must share the full-resolution image grid, dtype and device.
    Geometry uses float32/float64. By default each accepted pixel weighs 1;
    the optional weighted mean uses the splatted source's heuristic score,
    never a synthetic confidence assigned to predicted depth.

    Floor/discrete cell indices do not provide depth/calibration gradients
    through coordinates. Gradients do reach accepted image features and the
    backbone; a feature-only BEV loss does not train the depth head. This is
    a geometric feature representation, not a detection or occupancy model.
    """

    if (
        image_features.ndim != 4 or any(size == 0 for size in image_features.shape)
        or image_features.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("image_features must be non-empty float32/float64 [B, C, H, W].")
    batch, channels, height, width = image_features.shape
    shape = (batch, 1, height, width)
    depth = depth_support.depth_m
    valid = depth_support.valid_mask
    if (
        depth.shape != shape or depth.dtype != image_features.dtype
        or depth.device != image_features.device
    ):
        raise ValueError("depth_m must share image_features' grid, dtype, and device.")
    if valid.shape != shape or valid.dtype != torch.bool or valid.device != depth.device:
        raise ValueError("valid_mask must be bool with depth_m's shape and device.")
    valid = valid & torch.isfinite(depth) & (depth > 0)
    safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
    xyz = backproject_depth_to_vehicle(
        safe_depth, camera_projection=camera_projection,
        t_vehicle_to_camera=t_vehicle_to_camera,
    )
    x, y, z = xyz.unbind(dim=1)
    pixel_mask = valid[:, 0] & torch.isfinite(xyz).all(dim=1)
    pixel_mask = pixel_mask & torch.isfinite(image_features).all(dim=1)
    for values, bounds in ((x, grid.x_range_m), (y, grid.y_range_m), (z, grid.z_range_m)):
        pixel_mask = pixel_mask & (values >= bounds[0]) & (values < bounds[1])

    weights = torch.ones_like(depth)
    if weight_by_splat_confidence:
        confidence = depth_support.splat_confidence
        if depth_support.source != "splatted" or confidence is None:
            raise ValueError("confidence weighting requires the splatted depth source.")
        if (
            confidence.shape != shape or confidence.dtype != depth.dtype
            or confidence.device != depth.device
        ):
            raise ValueError("splat_confidence must share depth_m's shape, dtype, and device.")
        weights = confidence
        pixel_mask = pixel_mask & (
            torch.isfinite(confidence[:, 0]) & (confidence[:, 0] > 0) & (confidence[:, 0] <= 1)
        )

    # Select before flooring or multiplying: NaN coordinates/features/scores
    # in rejected pixels must not become integer indices or poison a sum.
    b, v, u = torch.where(pixel_mask)
    rows = torch.floor((x[b, v, u] - grid.x_range_m[0]) / grid.resolution_m).long()
    cols = torch.floor((y[b, v, u] - grid.y_range_m[0]) / grid.resolution_m).long()
    bev_height, bev_width = grid.shape
    # Floating-point division can round a near-upper-bound index upward.
    # Exclude it explicitly; never clamp out-of-grid points into border cells.
    inside = (rows >= 0) & (rows < bev_height) & (cols >= 0) & (cols < bev_width)
    b, v, u, rows, cols = (value[inside] for value in (b, v, u, rows, cols))
    accepted = torch.zeros_like(valid)
    accepted[b, 0, v, u] = True
    cell_count = batch * bev_height * bev_width
    indices = b * (bev_height * bev_width) + rows * bev_width + cols
    values = image_features.permute(0, 2, 3, 1)[b, v, u]
    selected_weights = weights[b, 0, v, u, None]
    sums = image_features.new_zeros((cell_count, channels)).index_add(
        0, indices, values * selected_weights,
    )
    mass = image_features.new_zeros((cell_count, 1)).index_add(0, indices, selected_weights)
    counts = torch.zeros(cell_count, 1, dtype=torch.long, device=depth.device).index_add(
        0, indices, torch.ones_like(indices[:, None]),
    )
    # Empty cells divide by 1 and remain zero. Do not clamp positive small
    # confidence masses to epsilon, which would bias their weighted mean.
    means = sums / torch.where(mass > 0, mass, torch.ones_like(mass))
    return CameraBEVFeatures(
        features=means.reshape(batch, bev_height, bev_width, channels).permute(0, 3, 1, 2),
        pixel_count=counts.reshape(batch, 1, bev_height, bev_width),
        weight_sum=mass.reshape(batch, 1, bev_height, bev_width),
        pixel_mask=accepted,
    )
