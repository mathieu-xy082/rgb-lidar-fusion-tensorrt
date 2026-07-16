import subprocess
import sys

import numpy as np
import pytest

from rgb_lidar_fusion import calibration as calibration_module
from rgb_lidar_fusion import project_lidar as project_lidar_module
from rgb_lidar_fusion.calibration import CameraCalibration, make_identity_calibration
from rgb_lidar_fusion.project_lidar import (
    build_enriched_lidar_maps,
    build_sparse_lidar_maps,
    project_lidar_to_image,
    read_ppm_image,
    render_lidar_overlay,
    write_ppm_image,
)


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


def test_enriched_lidar_maps_preserve_sparse_channels_and_append_splatted_maps():
    sparse_maps = np.zeros((6, 5, 5), dtype=np.float32)
    sparse_maps[0, 2, 2] = 0.25
    sparse_maps[1, 2, 2] = 0.10
    sparse_maps[2, 2, 2] = -0.05
    sparse_maps[3, 2, 2] = 0.02
    sparse_maps[4, 2, 2] = 0.90
    sparse_maps[5, 2, 2] = 1.0

    enriched = build_enriched_lidar_maps(sparse_maps)

    assert enriched.shape == (8, 5, 5)
    np.testing.assert_array_equal(enriched[:6], sparse_maps)
    assert enriched[6, 2, 2] == pytest.approx(0.25)
    assert enriched[6, 2, 3] == pytest.approx(0.25)
    assert enriched[7, 2, 2] == pytest.approx(1.0)
    assert 0.0 < enriched[7, 2, 3] < enriched[7, 2, 2]


def test_enriched_lidar_maps_reject_sparse_map_shape_that_would_violate_baseline_contract():
    with pytest.raises(ValueError, match="sparse_lidar_maps must have shape \\[6, H, W\\]"):
        build_enriched_lidar_maps(np.zeros((5, 4, 4), dtype=np.float32))


def test_parse_kitti_calibration_file_builds_rectified_velodyne_to_camera_transform(tmp_path):
    calib_file = tmp_path / "000000.txt"
    calib_file.write_text(
        "\n".join(
            [
                "P2: 7.0 0.0 6.0 0.0 0.0 8.0 5.0 0.0 0.0 0.0 1.0 0.0",
                "R0_rect: 0.0 -1.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0",
                "Tr_velo_to_cam: 1.0 0.0 0.0 1.0 0.0 1.0 0.0 2.0 0.0 0.0 1.0 3.0",
            ]
        ),
        encoding="utf-8",
    )

    calibration = calibration_module.parse_kitti_calibration_file(calib_file)

    np.testing.assert_allclose(
        calibration.k,
        [[7.0, 0.0, 6.0], [0.0, 8.0, 5.0], [0.0, 0.0, 1.0]],
    )
    expected_transform = np.array(
        [
            [0.0, -1.0, 0.0, -2.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(calibration.t_camera_from_vehicle, expected_transform)


def test_load_velodyne_bin_reads_kitti_float32_xyzi_points(tmp_path):
    velodyne_file = tmp_path / "000000.bin"
    expected_points = np.array(
        [[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25]],
        dtype=np.float32,
    )
    expected_points.tofile(velodyne_file)

    points = project_lidar_module.load_velodyne_bin(velodyne_file)

    assert points.dtype == np.float32
    np.testing.assert_allclose(points, expected_points)


def test_kitti_projection_uses_full_p2_projection_matrix_translation(tmp_path):
    calib_file = tmp_path / "000001.txt"
    calib_file.write_text(
        "\n".join(
            [
                "P2: 1.0 0.0 0.0 10.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0",
                "R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0",
                "Tr_velo_to_cam: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    calibration = calibration_module.parse_kitti_calibration_file(calib_file)
    points = np.array([[0.0, 0.0, 10.0, 1.0]], dtype=np.float32)

    projected = project_lidar_to_image(points, calibration, image_shape=(10, 10))

    np.testing.assert_allclose(projected.pixels, [[1.0, 0.0]], atol=1e-6)


def test_render_lidar_overlay_draws_depth_colored_points_without_mutating_image():
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    calibration = make_identity_calibration(fx=1, fy=1, cx=2, cy=2)
    points = np.array(
        [
            [0.0, 0.0, 5.0, 1.0],
            [20.0, 0.0, 10.0, 1.0],
        ],
        dtype=np.float32,
    )
    projected = project_lidar_to_image(points, calibration, image_shape=(5, 6))

    overlay = render_lidar_overlay(image, projected, max_depth_m=10.0, point_radius=0)

    assert overlay.dtype == np.uint8
    assert image.sum() == 0
    np.testing.assert_array_equal(overlay[2, 2], [127, 0, 128])
    np.testing.assert_array_equal(overlay[2, 2 + 20 // 10], [0, 0, 255])


def test_write_ppm_image_saves_ascii_visualization(tmp_path):
    output_file = tmp_path / "overlay.ppm"
    image = np.array([[[255, 0, 0], [0, 128, 255]]], dtype=np.uint8)

    write_ppm_image(output_file, image)

    assert output_file.read_text(encoding="ascii") == "P3\n2 1\n255\n255 0 0 0 128 255\n"


def test_read_ppm_image_loads_ascii_rgb_canvas(tmp_path):
    image_file = tmp_path / "canvas.ppm"
    image_file.write_text(
        "P3\n# tiny KITTI-like crop converted locally\n2 1\n255\n255 0 0 0 128 255\n",
        encoding="ascii",
    )

    image = read_ppm_image(image_file)

    assert image.dtype == np.uint8
    assert image.shape == (1, 2, 3)
    np.testing.assert_array_equal(image, [[[255, 0, 0], [0, 128, 255]]])


def test_smoke_script_can_generate_synthetic_overlay_ppm(tmp_path):
    output_file = tmp_path / "synthetic_overlay.ppm"

    completed = subprocess.run(
        [sys.executable, "scripts/smoke_project_lidar.py", "--overlay-output", str(output_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "overlay_output=" in completed.stdout
    assert output_file.read_text(encoding="ascii").startswith("P3\n640 360\n255\n")


def test_smoke_script_can_project_local_kitti_calib_and_velodyne_sample(tmp_path):
    calib_file = tmp_path / "000000.txt"
    calib_file.write_text(
        "\n".join(
            [
                "P2: 10.0 0.0 15.0 0.0 0.0 10.0 10.0 0.0 0.0 0.0 1.0 0.0",
                "R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0",
                "Tr_velo_to_cam: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    velodyne_file = tmp_path / "000000.bin"
    np.array(
        [
            [0.0, 0.0, 10.0, 0.9],
            [100.0, 0.0, 10.0, 0.1],
        ],
        dtype=np.float32,
    ).tofile(velodyne_file)
    output_file = tmp_path / "kitti_overlay.ppm"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_project_lidar.py",
            "--calib-file",
            str(calib_file),
            "--velodyne-file",
            str(velodyne_file),
            "--image-size",
            "20x30",
            "--overlay-output",
            str(output_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mode=kitti" in completed.stdout
    assert "loaded_points=2" in completed.stdout
    assert "projected_points=1" in completed.stdout
    assert output_file.read_text(encoding="ascii").startswith("P3\n30 20\n255\n")


def test_smoke_script_can_render_overlay_on_local_ppm_rgb_canvas(tmp_path):
    image_file = tmp_path / "canvas.ppm"
    image_file.write_text("P3\n4 3\n255\n" + " ".join(["10 20 30"] * 12) + "\n", encoding="ascii")
    output_file = tmp_path / "overlay.ppm"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_project_lidar.py",
            "--image-file",
            str(image_file),
            "--image-size",
            "3x4",
            "--overlay-output",
            str(output_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "image_source=ppm" in completed.stdout
    overlay_text = output_file.read_text(encoding="ascii")
    assert overlay_text.startswith("P3\n4 3\n255\n")
    assert "10 20 30" in overlay_text


def test_smoke_script_infers_image_size_from_local_ppm_canvas(tmp_path):
    image_file = tmp_path / "canvas.ppm"
    image_file.write_text("P3\n5 4\n255\n" + " ".join(["10 20 30"] * 20) + "\n", encoding="ascii")
    output_file = tmp_path / "overlay.ppm"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_project_lidar.py",
            "--image-file",
            str(image_file),
            "--overlay-output",
            str(output_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "image_source=ppm" in completed.stdout
    assert "image_shape=4x5" in completed.stdout
    assert output_file.read_text(encoding="ascii").startswith("P3\n5 4\n255\n")


def test_smoke_script_can_save_sparse_lidar_maps_npz(tmp_path):
    sparse_file = tmp_path / "lidar_maps.npz"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_project_lidar.py",
            "--sparse-output",
            str(sparse_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "sparse_output=" in completed.stdout
    with np.load(sparse_file) as saved:
        assert set(saved.files) == {"lidar_maps"}
        assert saved["lidar_maps"].shape == (6, 360, 640)
        assert saved["lidar_maps"].dtype == np.float32


def test_smoke_script_can_write_synthetic_kitti_format_sample(tmp_path):
    sample_dir = tmp_path / "synthetic_kitti"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_project_lidar.py",
            "--write-synthetic-sample",
            str(sample_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "synthetic_sample=" in completed.stdout
    calib_file = sample_dir / "training" / "calib" / "000000.txt"
    velodyne_file = sample_dir / "training" / "velodyne" / "000000.bin"
    assert calib_file.read_text(encoding="utf-8").startswith("P2:")
    np.testing.assert_allclose(
        project_lidar_module.load_velodyne_bin(velodyne_file),
        [[0.0, 0.0, 10.0, 0.9], [1.0, 0.5, 20.0, 0.6], [-1.0, 0.2, 15.0, 0.7]],
    )
