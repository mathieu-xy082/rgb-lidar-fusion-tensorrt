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
    t_vehicle_to_camera: torch.Tensor,
) -> torch.Tensor:
    """Map camera-z [B, 1, H, W] to vehicle XYZ [B, 3, H, W], in meters.

    Pixel centers are integer (u, v), starting at (0, 0). P is [B, 3, 4]
    with P[:, 2, :3] = [0, 0, 1], including KITTI's non-zero fourth column.
    For P = [A | t], solve A*X_camera = (z + t_z)*[u, v, 1] - t.
    T_vehicle_to_camera [B, 4, 4] maps vehicle -> camera; backprojection
    solves the opposite direction. See the derivation next to the computation.

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
        ("t_vehicle_to_camera", t_vehicle_to_camera, (batch, 4, 4)),
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
    if not torch.allclose(t_vehicle_to_camera[:, 3], affine_row, atol=1e-7, rtol=0):
        raise ValueError("t_vehicle_to_camera must be an affine homogeneous transform.")

    # u indexes columns and v indexes rows: depth[b, 0, v, u]. Integer
    # centers start at (0, 0); flattening preserves the image's row order.
    # pixel_h is [1, 3, H*W], broadcast across the B calibrations.
    v, u = torch.meshgrid(
        torch.arange(height, dtype=depth_m.dtype, device=depth_m.device),
        torch.arange(width, dtype=depth_m.dtype, device=depth_m.device),
        indexing="ij",
    )
    pixel_h = torch.stack((u, v, torch.ones_like(u))).reshape(1, 3, -1)

    # Projection gives pixel coordinates only after homogeneous division:
    #   s * p = P * [X_camera, 1] = A * X_camera + t,
    # where p = [u, v, 1], P = [A | t], and X_camera = [X, Y, z] in meters.
    # A's third row is [0, 0, 1], so its third equation is s = z + t_z.
    # Substitution gives A * X_camera = (z + t_z) * p - t.
    # Subtract the full fourth column once, including t_z; do not replace
    # s by z unless t_z is zero. This column is a projection offset, not
    # the vehicle-to-camera extrinsic translation (it already includes A).
    #
    # Example: P = [[2, 0, 1, 4], [0, 2, 1, -2], [0, 0, 1, 0.5]],
    #          (u, v) = (2, 1), z = 10 m -> s = 10.5.
    # A * X_camera = [21, 10.5, 10.5] - [4, -2, 0.5] = [17, 12.5, 10].
    # Thus 2X + z = 17, 2Y + z = 12.5 -> X_camera = [3.5, 1.25, 10].
    # Its Euclidean norm is about 10.67 m, not the camera-z depth of 10 m.
    translation = camera_projection[:, :, 3:4]
    projected_z = depth_m.reshape(batch, 1, -1) + translation[:, 2:3]
    camera_xyz = torch.linalg.solve(
        camera_projection[:, :, :3], projected_z * pixel_h - translation
    )

    # Homogeneous coordinates append 1 so the extrinsic can translate too:
    #   X_camera_h = T_vehicle_to_camera * X_vehicle_h.
    # We know X_camera_h, so solve this system for X_vehicle_h, equivalent
    # to inverse(T_vehicle_to_camera) * X_camera_h without forming an inverse.
    # For a rigid T = [R | a; 0 | 1], this is X_vehicle = R.T*(X_camera-a),
    # where R.T is the transpose (inverse) of the rotation matrix R.
    # Here a is the extrinsic translation; P's offset was already handled
    # above. Both solves retain gradients, unlike rounding into BEV cells.
    camera_h = torch.cat((camera_xyz, torch.ones_like(camera_xyz[:, :1])), dim=1)
    vehicle_h = torch.linalg.solve(t_vehicle_to_camera, camera_h)
    return vehicle_h[:, :3].reshape(batch, 3, height, width)


def _validate_depth_shape(depth: torch.Tensor) -> None:
    if depth.ndim != 4 or depth.shape[1] != 1 or any(size == 0 for size in depth.shape):
        raise ValueError("depth must have non-empty shape [B, 1, H, W].")
    if not depth.is_floating_point():
        raise ValueError("depth must have a floating dtype.")
