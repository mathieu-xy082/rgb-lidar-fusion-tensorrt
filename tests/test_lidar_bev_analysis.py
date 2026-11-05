import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.lidar_center_model import LidarCenterDiagnostic
from rgb_lidar_fusion.lidar_bev_analysis import analyze_lidar_bev_sample, plot_lidar_bev_sample


@pytest.fixture
def root(tmp_path):
    for name in ("image_2", "velodyne", "calib", "label_2"):
        (tmp_path / name).mkdir()
    Image.fromarray(np.full((20, 20, 3), 100, dtype=np.uint8)).save(tmp_path / "image_2/000000.png")
    np.array([[5., 0, 0, .8], [5.1, .1, .2, .5]], dtype=np.float32).tofile(tmp_path / "velodyne/000000.bin")
    (tmp_path / "calib/000000.txt").write_text(
        "P2: 10 0 10 0 0 10 10 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
    )
    return tmp_path


def test_frame_analysis_is_read_only_and_labels_do_not_enter_model_inputs(root):
    label_path = root / "label_2/000000.txt"
    label_path.write_text("Car 0 0 0 6 6 14 14 2 2 4 0 1 5 0\n")
    model = LidarCenterDiagnostic().train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    grid = BEVGridConfig((0., 10.), (-4., 4.), (-3., 3.), .2)
    first = analyze_lidar_bev_sample(root, "000000", grid=grid, model=model)
    assert model.training and first.targets.center_mask.sum() == 1
    assert first.loss["total"] > 0
    label_path.write_text(label_path.read_text().replace("Car", "Van"))
    second = analyze_lidar_bev_sample(root, "000000", grid=grid, model=model)
    assert second.targets.center_mask.sum() == 0
    for name in ("point_count", "pillar_rms", "context_rms", "center_scores"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_missing_labels_mean_unknown_not_a_negative_training_frame(root):
    grid = BEVGridConfig((0., 10.), (-4., 4.), (-3., 3.), .2)
    sample = analyze_lidar_bev_sample(root, "000000", grid=grid, model=LidarCenterDiagnostic())
    assert sample.labels is None and sample.targets is None and sample.loss is None
    figure = plot_lidar_bev_sample(sample)
    figure.canvas.draw()
    assert np.asarray(figure.canvas.buffer_rgba()).std() > 0
    import matplotlib.pyplot as plt
    plt.close(figure)
