from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rgb_lidar_fusion.demo_artifacts import generate_synthetic_demo_artifacts


def test_generate_synthetic_demo_artifacts_removes_stale_files(tmp_path: Path) -> None:
    stale_path = tmp_path / "projected_lidar_examples" / "stale_overlay.ppm"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("old", encoding="ascii")

    generate_synthetic_demo_artifacts(tmp_path)

    assert not stale_path.exists()


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
    model_inputs_path = tmp_path / "model_inputs" / "synthetic_baseline_inputs.npz"
    manifest_path = tmp_path / "manifest.json"

    assert overlay_path.exists()
    assert overlay_path.read_text(encoding="ascii").startswith("P3\n640 360\n255\n")
    assert sparse_path.exists()
    assert splat_path.exists()
    assert model_inputs_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["outputs"] == {
        "overlay": "projected_lidar_examples/synthetic_overlay.ppm",
        "sparse_maps": "sparse_maps/synthetic_sparse_maps.npz",
        "splatted_maps": "splatted_maps/synthetic_splatted_maps.npz",
        "model_inputs": "model_inputs/synthetic_baseline_inputs.npz",
    }
    assert manifest["model_input_shapes"] == {
        "rgb": [1, 3, 360, 640],
        "lidar_maps": [1, 8, 360, 640],
    }
    assert manifest["artifact_integrity"] == {
        "overlay": {
            "bytes": overlay_path.stat().st_size,
            "sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        },
        "sparse_maps": {
            "bytes": sparse_path.stat().st_size,
            "sha256": hashlib.sha256(sparse_path.read_bytes()).hexdigest(),
        },
        "splatted_maps": {
            "bytes": splat_path.stat().st_size,
            "sha256": hashlib.sha256(splat_path.read_bytes()).hexdigest(),
        },
        "model_inputs": {
            "bytes": model_inputs_path.stat().st_size,
            "sha256": hashlib.sha256(model_inputs_path.read_bytes()).hexdigest(),
        },
    }

    sparse = np.load(sparse_path)
    splatted = np.load(splat_path)
    model_inputs = np.load(model_inputs_path)
    assert sparse["lidar_maps"].shape == (6, 360, 640)
    assert splatted["depth_expanded"].shape == (360, 640)
    assert splatted["confidence"].shape == (360, 640)
    assert int((splatted["confidence"] > 0).sum()) == manifest["splatted_pixels"]
    assert model_inputs["rgb"].shape == (1, 3, 360, 640)
    assert model_inputs["lidar_maps"].shape == (1, 8, 360, 640)
