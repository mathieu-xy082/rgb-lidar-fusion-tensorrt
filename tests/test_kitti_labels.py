from zipfile import ZipFile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion.calibration import CameraCalibration
from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.kitti_labels import label_box_vehicle, parse_kitti_labels
from rgb_lidar_fusion.kitti_label_data import audit_kitti_labels, install_kitti_labels
from rgb_lidar_fusion.lidar_center_targets import build_lidar_center_targets


CAR = "Car 0 0 0 6 6 14 14 2 2 4 0 1 5 0"
IGNORE = "DontCare -1 -1 -10 7 7 13 13 -1 -1 -1 -1000 -1000 -1000 -10"


@pytest.fixture
def calibration():
    return CameraCalibration(
        np.array([[10., 0, 10], [0, 10, 10], [0, 0, 1]]),
        np.array([[0, -1., 0, 0], [0, 0, -1., 0], [1., 0, 0, 0], [0, 0, 0, 1.]]),
        np.array([[10., 0, 10, 0], [0, 10, 10, 0], [0, 0, 1., 0]]),
    )


@pytest.fixture
def grid():
    return BEVGridConfig((0., 10.), (-4., 4.), (-3., 3.), .2)


def test_training_label_parser_keeps_dontcare_as_2d_only_and_accepts_empty_files():
    labels = parse_kitti_labels(CAR + "\n\n" + IGNORE)
    assert len(labels) == 2
    assert labels[0].dimensions_hwl == (2., 2., 4.)
    assert labels[0].bottom_center_camera == (0., 1., 5.)
    assert labels[1].bottom_center_camera is None and labels[1].rotation_y is None
    assert parse_kitti_labels("\n") == []


def test_partially_visible_long_vehicle_may_have_center_behind_camera():
    label = parse_kitti_labels(CAR.replace("Car", "Truck").replace(" 0 1 5 ", " 0 1 -0.16 "))[0]
    assert label.bottom_center_camera[2] == -.16


@pytest.mark.parametrize("text", [CAR + " .9", CAR.replace("Car", "Alien"), CAR.replace(" 2 2 4 ", " -1 2 4 "),
                                      CAR.replace(" 0 1 5 ", " 0 1 nan "), CAR.replace("6 6 14 14", "14 6 6 14")])
def test_invalid_or_scored_training_labels_are_rejected(text):
    with pytest.raises(ValueError, match="label line 1"):
        parse_kitti_labels(text)


def test_box_conversion_uses_geometric_center_before_transform_not_p2_translation(calibration):
    calibration.t_vehicle_to_camera[:3, 3] = [1., 2., 3.]
    calibration.projection_matrix[:, 3] = [20., 30., .5]
    center, corners = label_box_vehicle(parse_kitti_labels(CAR)[0], calibration)
    np.testing.assert_allclose(center, [2., 1., 2.])
    np.testing.assert_allclose(corners.mean(axis=0), center)
    np.testing.assert_allclose(np.ptp(corners, axis=0), [2., 4., 2.])
    with pytest.raises(ValueError, match="DontCare"):
        label_box_vehicle(parse_kitti_labels(IGNORE)[0], calibration)


def test_targets_share_row_x_column_y_and_fractional_offset_convention(calibration, grid):
    target = build_lidar_center_targets(parse_kitti_labels(CAR), calibration=calibration, image_shape=(20, 20), grid=grid)
    assert target.heatmap.shape == target.supervision_mask.shape == (1, 50, 40)
    assert target.center_mask.sum() == 1 and target.heatmap[0, 25, 20] == 1
    np.testing.assert_array_equal(target.offset_xy[:, 25, 20], [0., 0.])
    np.testing.assert_array_equal(target.centers_vehicle, [[5., 0., 0.]])
    assert 0 < target.heatmap[0, 26, 20] < 1


def test_unknown_space_and_dontcare_are_not_background_but_known_positive_wins(calibration, grid):
    ignored = build_lidar_center_targets(parse_kitti_labels(IGNORE), calibration=calibration, image_shape=(20, 20), grid=grid)
    assert not ignored.supervision_mask[0, 25, 20]
    assert not ignored.supervision_mask[0, 0, 0]
    positive = build_lidar_center_targets(parse_kitti_labels(IGNORE + "\n" + CAR), calibration=calibration, image_shape=(20, 20), grid=grid)
    assert positive.supervision_mask[0, 25, 20]
    assert not positive.supervision_mask[0, 26, 20]  # Gaussian tails do not override ignores.
    assert positive.center_mask.sum() == 1


def test_van_and_out_of_roi_cars_are_ignored_instead_of_mislabeled_background(calibration, grid):
    for text in (CAR.replace("Car", "Van"), CAR.replace(" 0 1 5 ", " 0 1 15 ")):
        target = build_lidar_center_targets(parse_kitti_labels(text), calibration=calibration, image_shape=(20, 20), grid=grid)
        assert target.center_mask.sum() == 0 and target.ignored_label_count == 1
        assert not target.supervision_mask[0, 25, 20]


def test_colliding_centers_raise_instead_of_overwriting_offsets(calibration, grid):
    with pytest.raises(ValueError, match="same BEV cell"):
        build_lidar_center_targets(parse_kitti_labels(CAR + "\n" + CAR), calibration=calibration, image_shape=(20, 20), grid=grid)


def test_p2_translation_does_not_make_behind_camera_columns_supervised():
    calibration = CameraCalibration(np.eye(3), np.eye(4), np.array([
        [10., 0, 10., 1000.], [0, 10., 10., 1000.], [0, 0, 1., 100.],
    ]))
    grid = BEVGridConfig((-2., 0.), (-1., 1.), (-3., -1.), 1.)
    target = build_lidar_center_targets([], calibration=calibration, image_shape=(20, 20), grid=grid)
    assert not target.supervision_mask.any()


def test_invalid_projection_cannot_silently_produce_training_targets(calibration, grid):
    calibration.projection_matrix[:] = 0
    with pytest.raises(ValueError, match="projection"):
        build_lidar_center_targets([], calibration=calibration, image_shape=(20, 20), grid=grid)


def _dataset(root):
    for name in ("image_2", "calib", "velodyne"):
        (root / name).mkdir(parents=True)
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(root / "image_2/000000.png")
    np.array([[5., 0., 0., .5]], dtype=np.float32).tofile(root / "velodyne/000000.bin")
    (root / "calib/000000.txt").write_text(
        "P2: 10 0 10 0 0 10 10 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
    )


def test_label_install_selects_only_local_ids_preserves_existing_data_and_audits(tmp_path, grid):
    root = tmp_path / "training"
    _dataset(root)
    archive = tmp_path / "labels.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("training/label_2/000000.txt", CAR + "\n" + IGNORE)
        zipped.writestr("training/label_2/000001.txt", CAR)
        zipped.writestr("../../unsafe.txt", "unused entry")
    before = (root / "velodyne/000000.bin").read_bytes()
    report = install_kitti_labels(root, archive)
    assert report["installed_frames"] == 1
    assert len(list((root / "label_2").iterdir())) == 1
    assert not (tmp_path / "unsafe.txt").exists()
    assert (root / "velodyne/000000.bin").read_bytes() == before
    result = audit_kitti_labels(root, grid=grid)
    assert result["ready"] and result["target_car_count"] == 1 and not result["errors"]
    assert result["class_counts"] == {"Car": 1, "DontCare": 1}
    assert install_kitti_labels(root, archive) == report
    (root / "label_2/000000.txt").write_text("different annotation")
    with pytest.raises(ValueError, match="overwrite"):
        install_kitti_labels(root, archive)


def test_missing_or_corrupt_labels_fail_audit(tmp_path, grid):
    root = tmp_path / "training"
    _dataset(root)
    assert not audit_kitti_labels(root, grid=grid)["ready"]
    (root / "label_2").mkdir()
    (root / "label_2/000000.txt").write_text("broken")
    assert not audit_kitti_labels(root, grid=grid)["ready"]
