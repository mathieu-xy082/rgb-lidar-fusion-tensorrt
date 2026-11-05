import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.lidar_pillars import build_lidar_pillars
from rgb_lidar_fusion.lidar_center_model import (
    CenterPrediction, LidarCenterDiagnostic, LidarCenterHead, decode_lidar_centers, lidar_center_loss,
)


def _targets():
    heatmap = torch.zeros(1, 1, 5, 7)
    heatmap[0, 0, 2, 3] = 1
    return dict(heatmap=heatmap, offset_xy=torch.zeros(1, 2, 5, 7),
                center_mask=heatmap.bool(), supervision_mask=torch.ones_like(heatmap, dtype=torch.bool))


def test_head_outputs_and_loss_backward_through_whole_lidar_branch():
    torch.manual_seed(13)
    grid = BEVGridConfig((0., 1.), (-.6, .8), (-3., 3.), .2)
    cloud = np.array([[.05, -.35, -1., .2], [.08, -.32, 1., .8], [.45, .1, 0., .5]], dtype=np.float32)
    pillars = build_lidar_pillars([cloud], grid=grid)
    features = torch.from_numpy(pillars.features).requires_grad_()
    model = LidarCenterDiagnostic()
    prediction = model(features, torch.from_numpy(pillars.point_mask), torch.from_numpy(pillars.indices), torch.from_numpy(pillars.point_count))
    assert prediction.heatmap_logits.shape == (1, 1, 5, 7)
    assert prediction.offset_xy.shape == (1, 2, 5, 7)
    loss = lidar_center_loss(prediction, **_targets())
    assert torch.isfinite(loss.total) and loss.total > 0
    loss.total.backward()
    for block in (model.pfn, model.context, model.head):
        for parameter in block.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert torch.isfinite(features.grad).all()
    assert torch.all(features.grad[~torch.from_numpy(pillars.point_mask)] == 0)


def test_ignored_cells_have_no_loss_or_gradient_and_offsets_only_use_centers():
    targets = _targets()
    targets["supervision_mask"][:, :, 0] = False
    logits = torch.zeros(1, 1, 5, 7, requires_grad=True)
    offsets = torch.full((1, 2, 5, 7), .5, requires_grad=True)
    original = lidar_center_loss(CenterPrediction(logits, offsets), **targets)
    changed = logits.detach().clone()
    changed[:, :, 0] = 100
    altered = lidar_center_loss(CenterPrediction(changed, offsets), **targets)
    torch.testing.assert_close(original.total, altered.total)
    original.total.backward()
    assert torch.all(logits.grad[:, :, 0] == 0)
    assert torch.count_nonzero(offsets.grad) == 2


def test_entirely_unknown_batch_has_zero_loss_not_fake_success_metric():
    targets = _targets()
    targets["heatmap"].zero_()
    targets["center_mask"].zero_()
    targets["supervision_mask"].zero_()
    logits = torch.zeros(1, 1, 5, 7, requires_grad=True)
    offsets = torch.zeros(1, 2, 5, 7, requires_grad=True)
    loss = lidar_center_loss(CenterPrediction(logits, offsets), **targets)
    loss.total.backward()
    assert loss.total == 0 and torch.all(logits.grad == 0) and torch.all(offsets.grad == 0)


def test_negative_frame_without_car_still_has_heatmap_training_signal():
    targets = _targets()
    targets["heatmap"].zero_()
    targets["center_mask"].zero_()
    prediction = LidarCenterHead()(torch.ones(1, 32, 5, 7))
    loss = lidar_center_loss(prediction, **targets)
    assert loss.heatmap > 0 and loss.offset == 0


def test_loss_matches_simple_focal_and_offset_reference():
    targets = _targets()
    prediction = CenterPrediction(torch.zeros(1, 1, 5, 7), torch.full((1, 2, 5, 7), .5))
    loss = lidar_center_loss(prediction, **targets)
    assert loss.heatmap.item() == pytest.approx(35 * np.log(2) * .25)
    assert loss.offset.item() == .5


def test_decode_uses_fractional_row_x_column_y_without_extra_half_cell():
    grid = BEVGridConfig((10., 11.), (-.6, .8), (-3., 3.), .2)
    logits = torch.full((1, 1, 5, 7), -20.)
    logits[0, 0, 2, 3] = 10.
    offsets = torch.zeros(1, 2, 5, 7)
    offsets[0, :, 2, 3] = torch.tensor([.25, .75])
    result = decode_lidar_centers(CenterPrediction(logits, offsets), grid=grid)[0]
    assert result.shape == (1, 3)
    torch.testing.assert_close(result[0, :2], torch.tensor([10.45, .15]))
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[0, 0, 2, 3] = False
    assert len(decode_lidar_centers(CenterPrediction(logits, offsets), grid=grid, valid_mask=mask)[0]) == 0


def test_decode_masks_unknown_peaks_before_local_maximum_suppression():
    grid = BEVGridConfig((0., 1.), (-.6, .8), (-3., 3.), .2)
    logits = torch.full((1, 1, 5, 7), -20.)
    logits[0, 0, 2, 3], logits[0, 0, 2, 4] = 10., 5.
    offsets = torch.full((1, 2, 5, 7), .5)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[0, 0, 2, 3] = False
    result = decode_lidar_centers(CenterPrediction(logits, offsets), grid=grid, valid_mask=mask)[0]
    assert len(result) == 1
    torch.testing.assert_close(result[0, :2], torch.tensor([.5, .3]))


@pytest.mark.parametrize("mutation", ["mask_dtype", "positive_ignored", "bad_offset", "nan_target"])
def test_invalid_training_contracts_are_rejected(mutation):
    targets = _targets()
    if mutation == "mask_dtype":
        targets["center_mask"] = targets["center_mask"].float()
    elif mutation == "positive_ignored":
        targets["supervision_mask"].zero_()
    elif mutation == "bad_offset":
        targets["offset_xy"][0, 0, 2, 3] = 2.
    else:
        targets["heatmap"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        lidar_center_loss(CenterPrediction(torch.zeros(1, 1, 5, 7), torch.zeros(1, 2, 5, 7)), **targets)
