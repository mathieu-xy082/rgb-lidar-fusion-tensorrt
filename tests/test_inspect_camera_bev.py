import hashlib
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion.camera_depth_model import CameraDepthModel
from rgb_lidar_fusion.camera_bev import BEVGridConfig


@pytest.mark.parametrize("source,weighted", [("sparse", False), ("splatted", True)])
def test_inspection_cli_writes_finite_bev_and_keeps_checkpoint_unchanged(tmp_path, source, weighted):
    root = tmp_path / "training"
    for name in ("image_2", "velodyne", "calib"):
        (root / name).mkdir(parents=True)
    Image.fromarray(np.full((6, 8, 3), 100, dtype=np.uint8)).save(root / "image_2" / "000000.png")
    np.array([[0., 0., 5., 0.8], [1., 0., 5., 0.5]], dtype=np.float32).tofile(root / "velodyne" / "000000.bin")
    (root / "calib" / "000000.txt").write_text(
        "P2: 4 0 4 0 0 4 3 0 0 0 1 0\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    checkpoint = tmp_path / "weights.pt"
    torch.save({
        "model_state_dict": CameraDepthModel().state_dict(),
        "config": {"height": 6, "width": 8, "max_depth_m": 80., "lidar_mode": "enriched"},
        "epoch": 5, "step": 2050,
    }, checkpoint)
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / "inspection"
    command = [sys.executable, "scripts/inspect_camera_bev.py", "--checkpoint", str(checkpoint),
               "--data-root", str(root), "--source", source, "--z-range", "0", "10",
               "--output-dir", str(output)]
    if weighted:
        command.append("--weight-by-splat-confidence")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert json.loads(next(output.glob("*.json")).read_text()) == summary
    assert summary["feature_shape"] == [1, 32, 100, 100]
    assert summary["accepted_pixels"] > 0
    assert summary["occupied_cells"] > 0
    assert summary["source"] == source
    assert summary["checkpoint_epoch"] == 5 and summary["checkpoint_step"] == 2050
    assert summary["device"] == "cpu" and summary["holdout"] is False
    assert summary["reference_points_in_roi"] == 2
    with np.load(summary["npz"]) as saved:
        assert np.isfinite(saved["camera_bev"]).all()
        assert saved["camera_projection"].shape == (1, 3, 4)
        assert saved["t_vehicle_to_camera"].shape == (1, 4, 4)
        assert int(saved["pixel_mask"].sum()) == summary["accepted_pixels"]
        assert int(saved["pixel_count"].sum()) == summary["accepted_pixels"]
    with Image.open(summary["png"]) as png:
        assert png.width > 100 and png.height > 100
        assert np.asarray(png).std() > 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize("source", ["sparse", "splatted"])
def test_joint_inspection_preserves_raw_lidar_outside_camera_fov_and_shared_grid(tmp_path, source):
    root = tmp_path / "training"
    for name in ("image_2", "velodyne", "calib"):
        (root / name).mkdir(parents=True)
    Image.fromarray(np.full((6, 8, 3), 100, dtype=np.uint8)).save(root / "image_2" / "000000.png")
    # The third point is inside the metric ROI, but outside the camera image.
    raw = np.array([[0., 0., 5., .8], [1., 0., 5., .5], [10., 10., 5., .2]], dtype=np.float32)
    path = root / "velodyne" / "000000.bin"
    raw.tofile(path)
    (root / "calib" / "000000.txt").write_text(
        "P2: 5 0 4 0 0 5 3 0 0 0 1 0\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    checkpoint = tmp_path / "weights.pt"
    torch.save({
        "model_state_dict": CameraDepthModel().state_dict(),
        "config": {"height": 6, "width": 8, "max_depth_m": 80., "lidar_mode": "enriched"},
        "epoch": 5, "step": 2050,
    }, checkpoint)
    before = checkpoint.read_bytes(), path.read_bytes()
    command = [sys.executable, "scripts/inspect_camera_bev.py", "--checkpoint", str(checkpoint),
               "--data-root", str(root), "--source", source, "--with-lidar", "--resolution", ".2",
               "--z-range", "0", "10", "--output-dir", str(tmp_path / "joint")]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["feature_shape"] == report["lidar"]["feature_shape"] == [1, 32, 250, 250]
    assert report["grid"]["resolution_m"] == .2
    assert report["reference_points_in_roi"] == 2
    assert report["lidar"]["pre_cap_points"] == report["lidar"]["retained_points"] == 3
    assert report["lidar"]["weights"] == "random" and report["lidar"]["seed"] == 13
    assert "_lidar_res0p2m" in report["png"]
    assert sum(report["coverage_overlap_cells"].values()) == 250 * 250
    with np.load(report["npz"]) as saved:
        assert saved["lidar_observed_mask"][0, 0, 50, 175]
        assert saved["pixel_count"][0, 0, 50, 175] == 0
        assert saved["coverage_overlap"][0, 0, 50, 175] == 2
        assert saved["lidar_point_count"].sum() == 3
        assert np.isfinite(saved["lidar_context_bev"]).all()
        assert saved["lidar_context_bev"].shape == saved["camera_bev"].shape
    with Image.open(report["png"]) as png:
        assert png.width > 1000 and np.asarray(png).std() > 0
    assert before == (checkpoint.read_bytes(), path.read_bytes())


def test_lidar_inspection_initialization_is_reproducible_without_cuda_or_global_rng_changes(monkeypatch):
    spec = importlib.util.spec_from_file_location("inspect_bev", "scripts/inspect_camera_bev.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: pytest.fail("inspection must not seed CUDA"))
    cloud = np.array([[.05, -.35, 1., .2]], dtype=np.float32)
    grid = BEVGridConfig((0., 1.), (-.4, .2), (-3., 3.), .2)
    before = torch.random.get_rng_state()
    first, _ = module._lidar_features(cloud, grid, 13)
    second, _ = module._lidar_features(cloud, grid, 13)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    np.testing.assert_array_equal(first["lidar_context_bev"], second["lidar_context_bev"])
