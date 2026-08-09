import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path("docs/onboarding/scripts/render_splatting_ppm.py")
spec = importlib.util.spec_from_file_location("render_splatting_ppm", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
render_splatting_ppm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_splatting_ppm)


def test_load_splat_from_splatted_npz_uses_precomputed_depth_and_confidence(tmp_path):
    sparse = np.zeros((6, 2, 3), dtype=np.float32)
    sparse[0, 0, 1] = 0.25
    sparse[5, 0, 1] = 1.0
    splatted = np.zeros((8, 2, 3), dtype=np.float32)
    splatted[:6] = sparse
    splatted[6, 0, 2] = 0.25
    splatted[7, 0, 2] = 0.5
    npz = tmp_path / "splatted_maps_000000.npz"
    np.savez_compressed(
        npz,
        schema_version=np.array("splatted-lidar-maps-v1"),
        source_sparse_maps=np.array("sparse_maps_000000.npz"),
        splat_radius_px=np.array(2, dtype=np.int32),
        splat_sigma_px=np.array(1.0, dtype=np.float32),
        splat_conflict_strategy=np.array("nearest_depth_then_confidence"),
        sparse_lidar_maps=sparse,
        splatted_lidar_maps=splatted,
        splatted_channel_names=np.array([str(i) for i in range(8)]),
    )

    loaded_sparse, loaded_splatted, out = render_splatting_ppm.load_splat_from_splatted_npz(
        npz,
        tmp_path / "viz",
    )

    np.testing.assert_array_equal(loaded_sparse, sparse)
    np.testing.assert_array_equal(loaded_splatted, splatted[6:8])
    assert out == tmp_path / "viz"
