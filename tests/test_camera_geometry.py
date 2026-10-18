import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.calibration import CameraCalibration
from rgb_lidar_fusion.camera_geometry import (
    backproject_depth_to_vehicle,
    normalized_depth_to_meters,
)
from rgb_lidar_fusion.dataset import KittiObjectDepthDataset
from rgb_lidar_fusion.project_lidar import project_lidar_to_image


def _geometry(batch=1):
    projection = torch.tensor([
        [7.0, 0.2, 3.0, 1.5],
        [0.0, 8.0, 2.0, -0.7],
        [0.0, 0.0, 1.0, 0.25],
    ], dtype=torch.float64)
    # LiDAR x forward, y left, z up -> camera x right, y down, z forward.
    transform = torch.tensor([
        [0.0, -1.0, 0.0, 0.3],
        [0.0, 0.0, -1.0, -0.2],
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    return {
        "camera_projection": projection.repeat(batch, 1, 1),
        "t_camera_from_vehicle": transform.repeat(batch, 1, 1),
    }


def test_metric_depth_uses_explicit_per_sample_scale_and_keeps_gradients():
    depth = torch.tensor([0.25, 0.5], dtype=torch.float64).reshape(2, 1, 1, 1)
    depth.requires_grad_()
    metric = normalized_depth_to_meters(depth, max_depth_m=torch.tensor([80.0, 60.0]))
    torch.testing.assert_close(metric.flatten(), depth.new_tensor([20, 30]))
    metric.sum().backward()
    torch.testing.assert_close(depth.grad.flatten(), depth.new_tensor([80, 60]))
    torch.testing.assert_close(
        normalized_depth_to_meters(depth, max_depth_m=100), depth * 100
    )


def test_metric_conversion_does_not_clamp_or_make_invalid_depths_valid():
    depth = torch.tensor([[[[-1.0, 0.0, 2.0, float("nan")]]]])
    metric = normalized_depth_to_meters(depth, max_depth_m=80)
    torch.testing.assert_close(metric[..., :3], torch.tensor([[[[-80.0, 0, 160]]]]))
    assert torch.isnan(metric[..., 3]).all()


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), [80, 60]])
def test_metric_depth_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="max_depth_m"):
        normalized_depth_to_meters(torch.ones(1, 1, 2, 3), max_depth_m=scale)


@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (4, 6)])
def test_backprojection_roundtrip_preserves_camera_z_and_full_projection(shape):
    height, width = shape
    geometry = _geometry(batch=2)
    geometry["camera_projection"][1, 0, 0] = 11
    geometry["t_camera_from_vehicle"][1, 0, 3] = -1
    depth = torch.linspace(5, 20, 2 * height * width, dtype=torch.float64).reshape(
        2, 1, height, width
    )
    xyz = backproject_depth_to_vehicle(depth, **geometry)
    assert xyz.shape == (2, 3, height, width)
    for b in range(2):
        calibration = CameraCalibration(
            k=geometry["camera_projection"][b, :, :3].numpy(),
            projection_matrix=geometry["camera_projection"][b].numpy(),
            t_camera_from_vehicle=geometry["t_camera_from_vehicle"][b].numpy(),
        )
        # Reproject independently through the existing NumPy production path.
        # The larger bounds retain tiny round-off excursions at right/bottom.
        projected = project_lidar_to_image(
            xyz[b].reshape(3, -1).T.numpy(), calibration, (height + 1, width + 1)
        )
        # Float32 may place a boundary pixel infinitesimally below zero;
        # compare the exact homogeneous projection before its bounds filter.
        vehicle_h = torch.cat((xyz[b].reshape(3, -1), torch.ones(1, height * width)))
        camera_h = geometry["t_camera_from_vehicle"][b] @ vehicle_h
        q = geometry["camera_projection"][b] @ camera_h
        v, u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        torch.testing.assert_close(q[:2] / q[2:], torch.stack((u, v)).reshape(2, -1).double())
        torch.testing.assert_close(camera_h[2], depth[b].flatten())
        for pixel, camera_point in zip(projected.pixels, projected.camera_points):
            np.testing.assert_allclose(pixel, np.rint(pixel), atol=2e-6)
            v_idx, u_idx = np.rint(pixel[::-1]).astype(int)
            assert camera_point[2] == pytest.approx(depth[b, 0, v_idx, u_idx].item())


def test_backprojection_matches_known_point_not_euclidean_ray_length():
    projection = torch.tensor([[[2., 0, 1, 4], [0, 2, 1, -2], [0, 0, 1, 0]]])
    transform = torch.eye(4)[None]
    xyz = backproject_depth_to_vehicle(
        torch.full((1, 1, 2, 3), 10.), camera_projection=projection,
        t_camera_from_vehicle=transform,
    )
    torch.testing.assert_close(xyz[0, :, 1, 2], torch.tensor([3., 1., 10.]))
    assert xyz[0, :, 1, 2].norm() > 10


def test_resize_then_backproject_uses_scaled_translation_column():
    geometry = _geometry()
    original = CameraCalibration(
        k=geometry["camera_projection"][0, :, :3].numpy(),
        projection_matrix=geometry["camera_projection"][0].numpy(),
        t_camera_from_vehicle=geometry["t_camera_from_vehicle"][0].numpy(),
    )
    resized = KittiObjectDepthDataset._resize_calibration(
        original, source_shape=(6, 20), target_shape=(3, 5)
    )
    before = backproject_depth_to_vehicle(torch.full((1, 1, 6, 20), 10.).double(), **geometry)
    after = backproject_depth_to_vehicle(
        torch.full((1, 1, 3, 5), 10.),
        camera_projection=torch.from_numpy(resized.projection_matrix)[None],
        t_camera_from_vehicle=torch.from_numpy(resized.t_camera_from_vehicle)[None],
    )
    torch.testing.assert_close(after.double(), before[:, :, ::2, ::4], atol=2e-6, rtol=1e-6)


def test_backprojection_depth_backward_matches_numerical_derivative():
    depth = torch.full((1, 1, 2, 3), 10., dtype=torch.float64, requires_grad=True)
    geometry = _geometry()
    assert torch.autograd.gradcheck(
        lambda values: backproject_depth_to_vehicle(values, **geometry), (depth,)
    )
    backproject_depth_to_vehicle(depth, **geometry).square().sum().backward()
    assert torch.isfinite(depth.grad).all()
    assert torch.all(depth.grad != 0)


@pytest.mark.parametrize("bad_key,bad_value,match", [
    ("camera_projection", torch.eye(3)[None], "shape"),
    ("camera_projection", torch.zeros(1, 3, 4), "rectified"),
    ("camera_projection", torch.full((1, 3, 4), float("nan")), "finite"),
    ("t_camera_from_vehicle", torch.zeros(1, 4, 4), "homogeneous"),
])
def test_backprojection_rejects_invalid_matrices(bad_key, bad_value, match):
    geometry = {key: value.float() for key, value in _geometry().items()}
    geometry[bad_key] = bad_value
    with pytest.raises(ValueError, match=match):
        backproject_depth_to_vehicle(torch.ones(1, 1, 2, 3), **geometry)


@pytest.mark.parametrize("depth", [torch.ones(1, 2, 3), torch.ones(1, 2, 3, 4),
                                   torch.empty(1, 1, 0, 3), torch.ones(1, 1, 2, 3).int()])
def test_geometry_rejects_invalid_depth_layout_or_dtype(depth):
    with pytest.raises(ValueError, match="depth"):
        backproject_depth_to_vehicle(depth, **_geometry())
    with pytest.raises(ValueError, match="depth"):
        normalized_depth_to_meters(depth, max_depth_m=80)


def test_backprojection_rejects_implicit_matrix_dtype_conversion():
    with pytest.raises(ValueError, match="dtype and device"):
        backproject_depth_to_vehicle(torch.ones(1, 1, 2, 3), **_geometry())


def test_backprojection_requires_explicit_full_precision_geometry():
    geometry = {key: value.half() for key, value in _geometry().items()}
    with pytest.raises(ValueError, match="float32 or float64"):
        backproject_depth_to_vehicle(torch.ones(1, 1, 2, 3).half(), **geometry)
