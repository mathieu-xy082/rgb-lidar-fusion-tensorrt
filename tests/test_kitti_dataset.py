import numpy as np
import pytest

from rgb_lidar_fusion.dataset import KittiObjectDepthDataset, KittiSparseLidarDataset


FIXTURE_ROOT = "tests/fixtures/synthetic_kitti"


def test_dataset_returns_rgb_lidar_maps_target_and_meta():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT)

    sample = dataset[0]

    assert len(dataset) == 1
    assert set(sample) == {"image", "lidar_maps", "target", "meta"}
    assert sample["image"].shape == (3, 4, 5)
    assert sample["image"].dtype == np.float32
    np.testing.assert_allclose(sample["image"][:, 0, 0], [1.0, 0.0, 0.0])
    assert sample["lidar_maps"].shape == (6, 4, 5)
    assert sample["lidar_maps"].dtype == np.float32
    assert sample["lidar_maps"][5].sum() == pytest.approx(2.0)
    assert sample["lidar_maps"][0, 2, 2] == pytest.approx(10.0 / 80.0)
    assert sample["lidar_maps"][4, 2, 2] == pytest.approx(0.7)
    assert sample["target"] == {"class_name": "synthetic_car", "bbox_2d": [1, 1, 3, 3]}
    assert sample["meta"] == {
        "sample_id": "000001",
        "image_shape": [4, 5],
        "image_path": "image-000001.npy",
        "lidar_path": "lidar-000001.npy",
        "lidar_representation": "sparse",
        "lidar_map_channels": [
            "normalized_camera_depth",
            "normalized_vehicle_x",
            "normalized_vehicle_y",
            "normalized_vehicle_z",
            "intensity",
            "point_mask",
        ],
    }


def test_dataset_can_return_splatted_lidar_representation_with_metadata():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT, lidar_representation="splatted")

    sample = dataset[0]

    assert sample["lidar_maps"].shape == (8, 4, 5)
    assert sample["lidar_maps"].dtype == np.float32
    assert sample["lidar_maps"][5].sum() == pytest.approx(2.0)
    assert sample["lidar_maps"][6, 2, 2] == pytest.approx(10.0 / 80.0)
    assert sample["lidar_maps"][7, 2, 2] == pytest.approx(1.0)
    assert sample["meta"]["lidar_representation"] == "splatted"
    assert sample["meta"]["lidar_map_channels"] == [
        "normalized_camera_depth_sparse",
        "normalized_vehicle_x_sparse",
        "normalized_vehicle_y_sparse",
        "normalized_vehicle_z_sparse",
        "intensity_sparse",
        "point_mask_sparse",
        "normalized_camera_depth_splatted",
        "splat_confidence",
    ]


def test_dataset_rejects_unknown_lidar_representation():
    with pytest.raises(ValueError, match="lidar_representation must be 'sparse' or 'splatted'"):
        KittiSparseLidarDataset(FIXTURE_ROOT, lidar_representation="dense")


def test_dataset_rejects_missing_sample_files():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT, manifest_name="missing-file-manifest.json")

    with pytest.raises(FileNotFoundError, match="missing-image.npy"):
        _ = dataset[0]


def test_dataset_reports_missing_required_manifest_fields():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT, manifest_name="missing-field-manifest.json")

    with pytest.raises(ValueError, match="sample 000002.*lidar"):
        _ = dataset[0]


def test_dataset_rejects_manifest_without_samples(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"dataset": "synthetic"}', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.json.*samples"):
        KittiSparseLidarDataset(tmp_path)


def test_dataset_reports_out_of_range_index_with_dataset_size():
    dataset = KittiSparseLidarDataset(FIXTURE_ROOT)

    with pytest.raises(IndexError, match="Dataset index 1 out of range for 1 sample"):
        _ = dataset[1]


def _write_kitti_object_frame(root, sample_id="000000"):
    Image = pytest.importorskip("PIL.Image")
    image_dir = root / "image_2"
    velodyne_dir = root / "velodyne"
    calibration_dir = root / "calib"
    image_dir.mkdir(parents=True)
    velodyne_dir.mkdir()
    calibration_dir.mkdir()

    pixels = np.zeros((6, 8, 3), dtype=np.uint8)
    pixels[..., 0] = 255
    Image.fromarray(pixels, mode="RGB").save(image_dir / f"{sample_id}.png")
    points = np.array(
        [
            [0.0, 0.0, 5.0, 0.8],
            [2.0, 0.0, 5.0, 0.6],
            [-2.0, 0.0, 5.0, 0.4],
        ],
        dtype=np.float32,
    )
    points.tofile(velodyne_dir / f"{sample_id}.bin")
    (calibration_dir / f"{sample_id}.txt").write_text(
        "\n".join(
            (
                "P2: 4 0 4 0 0 4 3 0 0 0 1 0",
                "R0_rect: 1 0 0 0 1 0 0 0 1",
                "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0",
            )
        )
        + "\n"
    )


def test_kitti_object_depth_dataset_loads_resizes_and_projects_downloaded_frame(tmp_path):
    _write_kitti_object_frame(tmp_path)
    dataset = KittiObjectDepthDataset(tmp_path, image_shape=(3, 4))

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["image"].shape == (3, 3, 4)
    np.testing.assert_allclose(sample["image"][0], 1.0)
    np.testing.assert_allclose(sample["image"][1:], 0.0)
    assert sample["lidar_maps"].shape == (6, 3, 4)
    assert sample["lidar_maps"][5].sum() == pytest.approx(3.0)
    assert sample["lidar_maps"][0, 2, 2] == pytest.approx(5.0 / 80.0)
    assert sample["meta"]["sample_id"] == "000000"
    assert sample["meta"]["image_shape"] == [3, 4]
    assert sample["meta"]["source_image_shape"] == [6, 8]


def test_kitti_object_depth_dataset_rejects_incomplete_download(tmp_path):
    _write_kitti_object_frame(tmp_path)
    (tmp_path / "velodyne" / "000000.bin").unlink()

    with pytest.raises(FileNotFoundError, match="frame 000000 is incomplete"):
        KittiObjectDepthDataset(tmp_path, image_shape=(3, 4))


def test_kitti_object_depth_dataset_requires_requested_sample_count(tmp_path):
    _write_kitti_object_frame(tmp_path)

    with pytest.raises(ValueError, match="provides 1 frames.*sample_limit=2"):
        KittiObjectDepthDataset(tmp_path, image_shape=(3, 4), sample_limit=2)


@pytest.mark.parametrize("scale", [0, -80, np.nan, np.inf])
def test_kitti_depth_dataset_rejects_invalid_geometry_scale(tmp_path, scale):
    with pytest.raises(ValueError, match="max_depth_m must be finite and positive"):
        KittiObjectDepthDataset(tmp_path, image_shape=(3, 4), max_depth_m=scale)


def test_kitti_depth_geometry_is_resized_without_changing_projection_inputs(tmp_path):
    from rgb_lidar_fusion.calibration import parse_kitti_calibration_file
    from rgb_lidar_fusion.project_lidar import build_sparse_lidar_maps, project_lidar_to_image

    _write_kitti_object_frame(tmp_path)
    path = tmp_path / "calib" / "000000.txt"
    path.write_text(
        "P2: 4 0 4 2 0 4 3 -1 0 0 1 0.25\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    dataset = KittiObjectDepthDataset(tmp_path, image_shape=(3, 6), max_depth_m=60)
    sample = dataset[0]
    original = parse_kitti_calibration_file(path)
    expected_projection = original.projection_matrix.copy()
    expected_projection[0] *= 0.75
    expected_projection[1] *= 0.5

    np.testing.assert_array_equal(sample["camera_projection"], expected_projection)
    np.testing.assert_array_equal(sample["t_camera_from_vehicle"], np.eye(4))
    assert sample["max_depth_m"] == np.float32(60)
    points = np.fromfile(tmp_path / "velodyne" / "000000.bin", dtype=np.float32).reshape(-1, 4)
    resized = dataset._resize_calibration(
        original, source_shape=(6, 8), target_shape=(3, 6)
    )
    before = project_lidar_to_image(points, original, image_shape=(6, 8))
    after = project_lidar_to_image(points, resized, image_shape=(3, 6))
    np.testing.assert_allclose(after.pixels, before.pixels * [0.75, 0.5], atol=1e-6)
    np.testing.assert_array_equal(original.projection_matrix[0], [4, 0, 4, 2])
    np.testing.assert_array_equal(
        sample["lidar_maps"], build_sparse_lidar_maps(after, (3, 6), max_depth_m=60)
    )


def test_kitti_geometry_survives_batching_holdout_and_fixed_evaluation(tmp_path):
    from rgb_lidar_fusion.lidar_splatting import SplattingConfig
    from rgb_lidar_fusion.model_batch import (
        CAMERA_GEOMETRY_SHAPES,
        dataset_items_to_model_batch,
        model_batch_to_camera_depth_training_batch,
    )

    _write_kitti_object_frame(tmp_path)
    dataset = KittiObjectDepthDataset(tmp_path, image_shape=(6, 8))
    item = dataset[0]
    second = {**item, "max_depth_m": np.float32(100)}
    second["camera_projection"] = item["camera_projection"].copy()
    second["camera_projection"][0, 3] = 5
    second["t_camera_from_vehicle"] = item["t_camera_from_vehicle"].copy()
    second["t_camera_from_vehicle"][0, 3] = 2
    batch = dataset_items_to_model_batch([item, second], include_splatted_depth=False)
    prepared = model_batch_to_camera_depth_training_batch(batch, seed=13)
    for key, shape in CAMERA_GEOMETRY_SHAPES.items():
        assert batch[key].shape == (2, *shape)
        assert batch[key].dtype == np.float32
        np.testing.assert_array_equal(batch[key], np.stack([item[key], second[key]]))
        np.testing.assert_array_equal(prepared[key], batch[key])
        assert not np.shares_memory(prepared[key], batch[key])
    assert batch["inputs"].shape[1] == 9  # Geometry is not a CNN input channel.

    pytest.importorskip("torch")
    from rgb_lidar_fusion.training import prepare_camera_depth_evaluation_batch

    options = dict(
        dataset=[item, second], holdout_fraction=0.2, holdout_seed=1013,
        splatting_config=SplattingConfig(),
    )
    combined = prepare_camera_depth_evaluation_batch(indices=[1, 0], **options)
    single = prepare_camera_depth_evaluation_batch(indices=[1], **options)
    for key in combined:
        np.testing.assert_array_equal(combined[key][:1], single[key])
    for key in CAMERA_GEOMETRY_SHAPES:
        np.testing.assert_array_equal(combined[key], batch[key][::-1])
