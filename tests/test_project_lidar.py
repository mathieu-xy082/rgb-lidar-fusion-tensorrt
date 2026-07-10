import numpy as np
import pytest

from rgb_lidar_fusion.calibration import CameraCalibration, make_identity_calibration
from rgb_lidar_fusion.project_lidar import build_sparse_lidar_maps, project_lidar_to_image


def test_identity_projection_keeps_points_in_front_and_inside_image():
    calibration = make_identity_calibration(fx=100, fy=100, cx=320, cy=180)
    points = np.array(
        [
            [0.0, 0.0, 10.0, 0.9],      # center
            [1.0, 0.5, 20.0, 0.6],      # inside
            [0.0, 0.0, -5.0, 0.1],      # behind camera
            [100.0, 0.0, 1.0, 0.2],     # outside image
        ],
        dtype=np.float32,
    )

    projected = project_lidar_to_image(points, calibration, image_shape=(360, 640))

    assert projected.pixels.shape == (2, 2)
    np.testing.assert_allclose(projected.pixels[0], [320.0, 180.0], atol=1e-4)
    np.testing.assert_allclose(projected.pixels[1], [325.0, 182.5], atol=1e-4)
    np.testing.assert_allclose(projected.intensity, [0.9, 0.6], atol=1e-6)


def test_projection_rejects_invalid_lidar_shape():
    calibration = make_identity_calibration()
    with pytest.raises(ValueError, match="shape"):
        project_lidar_to_image(np.zeros((3, 2), dtype=np.float32), calibration, image_shape=(10, 10))


def test_calibration_validates_matrix_shapes():
    with pytest.raises(ValueError, match="shape"):
        CameraCalibration(k=np.eye(4), t_camera_from_vehicle=np.eye(4))
    with pytest.raises(ValueError, match="shape"):
        CameraCalibration(k=np.eye(3), t_camera_from_vehicle=np.eye(3))


def test_sparse_maps_keep_nearest_point_when_pixels_collide():
    calibration = make_identity_calibration(fx=1, fy=1, cx=5, cy=5)
    points = np.array(
        [
            [0.0, 0.0, 20.0, 0.2],
            [0.0, 0.0, 10.0, 0.8],
        ],
        dtype=np.float32,
    )
    projected = project_lidar_to_image(points, calibration, image_shape=(10, 10))
    maps = build_sparse_lidar_maps(projected, image_shape=(10, 10), max_depth_m=100.0)

    assert maps.shape == (6, 10, 10)
    assert maps[5].sum() == 1.0
    assert maps[0, 5, 5] == pytest.approx(0.1)  # 10m / 100m, nearest point wins
    assert maps[4, 5, 5] == pytest.approx(0.8)
