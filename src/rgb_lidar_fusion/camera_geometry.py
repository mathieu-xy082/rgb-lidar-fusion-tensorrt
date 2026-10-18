"""Differentiable camera-z geometry, before support filtering or BEV binning.

KITTI's camera frame here is the rectified reference camera frame obtained
with R0_rect @ Tr_velo_to_cam, not a new frame centered on camera 2.
"""

from __future__ import annotations

import torch


def normalized_depth_to_meters(
    depth: torch.Tensor,
    *,
    max_depth_m: float | torch.Tensor,
) -> torch.Tensor:
    """Scale [B, 1, H, W] camera-z by a scalar or per-sample [B] scale.

    Use the checkpoint's scale (not an implicit 80 m default). No clamping,
    masking, or range-to-z conversion is performed; invalid depths stay invalid.
    """

    _validate_depth_shape(depth)
    scale = torch.as_tensor(max_depth_m, dtype=depth.dtype, device=depth.device)
    if scale.ndim == 1 and scale.shape == (depth.shape[0],):
        scale = scale[:, None, None, None]
    elif scale.ndim != 0:
        raise ValueError("max_depth_m must be a scalar or have shape [B].")
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("max_depth_m must be finite and positive.")
    return depth * scale


def backproject_depth_to_vehicle(
    depth_m: torch.Tensor,
    *,
    camera_projection: torch.Tensor,
    t_camera_from_vehicle: torch.Tensor,
) -> torch.Tensor:
    """Map camera-z [B, 1, H, W] to vehicle XYZ [B, 3, H, W], in meters.

    Pixel centers are integer (u, v), starting at (0, 0). P is [B, 3, 4]
    with P[:, 2, :3] = [0, 0, 1], including KITTI's non-zero fourth column.
    For P = [A | t], solve A*X_camera = (z + t_z)*[u, v, 1] - t.
    Then invert T_camera_from_vehicle [B, 4, 4]. Never apply P's translation
    twice or treat camera-z as the Euclidean distance along the ray.

    Inputs must share float32/float64 dtype and device. This algebraic operation
    does not select reliable support or filter invalid depth; callers must
    mask it before BEV aggregation. No coordinates are rounded here.
    """

    _validate_depth_shape(depth_m)
    if depth_m.dtype not in (torch.float32, torch.float64):
        raise ValueError("backprojection requires float32 or float64 depth and matrices.")
    batch, _, height, width = depth_m.shape
    for name, matrix, shape in (
        ("camera_projection", camera_projection, (batch, 3, 4)),
        ("t_camera_from_vehicle", t_camera_from_vehicle, (batch, 4, 4)),
    ):
        if matrix.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if matrix.device != depth_m.device or matrix.dtype != depth_m.dtype:
            raise ValueError(f"{name} must share depth_m's dtype and device.")
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} must be finite.")
    expected_row = depth_m.new_tensor([0.0, 0.0, 1.0]).expand(batch, -1)
    if not torch.allclose(camera_projection[:, 2, :3], expected_row, atol=1e-7, rtol=0):
        raise ValueError("camera_projection must be rectified with row 2[:3] = [0, 0, 1].")
    affine_row = depth_m.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(batch, -1)
    if not torch.allclose(t_camera_from_vehicle[:, 3], affine_row, atol=1e-7, rtol=0):
        raise ValueError("t_camera_from_vehicle must be an affine homogeneous transform.")

    v, u = torch.meshgrid(
        torch.arange(height, dtype=depth_m.dtype, device=depth_m.device),
        torch.arange(width, dtype=depth_m.dtype, device=depth_m.device),
        indexing="ij",
    )
    pixel_h = torch.stack((u, v, torch.ones_like(u))).reshape(1, 3, -1)
    translation = camera_projection[:, :, 3:4]
    projected_z = depth_m.reshape(batch, 1, -1) + translation[:, 2:3]
    camera_xyz = torch.linalg.solve(
        camera_projection[:, :, :3], projected_z * pixel_h - translation
    )
    camera_h = torch.cat((camera_xyz, torch.ones_like(camera_xyz[:, :1])), dim=1)
    vehicle_h = torch.linalg.solve(t_camera_from_vehicle, camera_h)
    return vehicle_h[:, :3].reshape(batch, 3, height, width)


def _validate_depth_shape(depth: torch.Tensor) -> None:
    if depth.ndim != 4 or depth.shape[1] != 1 or any(size == 0 for size in depth.shape):
        raise ValueError("depth must have non-empty shape [B, 1, H, W].")
    if not depth.is_floating_point():
        raise ValueError("depth must have a floating dtype.")
