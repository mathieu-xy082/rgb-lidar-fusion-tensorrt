from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rgb_lidar_fusion.demo_artifacts import generate_synthetic_demo_artifacts


def test_generate_synthetic_demo_artifacts_writes_all_expected_files(tmp_path: Path) -> None:
    manifest = generate_synthetic_demo_artifacts(tmp_path)

    assert manifest["mode"] == "synthetic"
    assert manifest["image_shape"] == [360, 640]
    assert manifest["projected_points"] == 3
    assert manifest["occupied_pixels"] == 3
    assert manifest["splatted_pixels"] > manifest["occupied_pixels"]

    overlay_path = tmp_path / "projected_lidar_examples" / "synthetic_overlay.ppm"
    sparse_path = tmp_path / "sparse_maps" / "synthetic_sparse_maps.npz"
    splat_path = tmp_path / "splatted_maps" / "synthetic_splatted_maps.npz"
    manifest_path = tmp_path / "manifest.json"

    assert overlay_path.exists()
    assert overlay_path.read_text(encoding="ascii").startswith("P3\n640 360\n255\n")
    assert sparse_path.exists()
    assert splat_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    sparse = np.load(sparse_path)
    splatted = np.load(splat_path)
    assert sparse["lidar_maps"].shape == (6, 360, 640)
    assert splatted["depth_expanded"].shape == (360, 640)
    assert splatted["confidence"].shape == (360, 640)
    assert int((splatted["confidence"] > 0).sum()) == manifest["splatted_pixels"]
