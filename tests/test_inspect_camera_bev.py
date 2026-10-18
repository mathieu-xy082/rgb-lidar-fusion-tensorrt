import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion.camera_depth_model import CameraDepthModel


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
