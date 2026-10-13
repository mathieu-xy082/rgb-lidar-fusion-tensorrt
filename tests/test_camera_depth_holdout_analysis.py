from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import rgb_lidar_fusion.camera_depth_analysis as analysis_tools
import rgb_lidar_fusion.training as training
from rgb_lidar_fusion.camera_depth_analysis import (
    analyze_camera_depth_sample,
    compare_camera_depth_holdout_checkpoints,
    plot_camera_depth_holdout_comparison,
    plot_camera_depth_holdout_input,
)
from rgb_lidar_fusion.camera_depth_model import CameraDepthModel
from rgb_lidar_fusion.lidar_splatting import SplattingConfig


@pytest.fixture
def comparison_case(tmp_path, monkeypatch):
    config = {
        "dataset": "kitti_camera_depth", "data_root": str(tmp_path),
        "sample_count": 4, "height": 12, "width": 16,
        "max_depth_m": 80.0, "validation_fraction": 0.5,
        "holdout_fraction": 0.4, "seed": 13, "split_seed": 19,
        "validation_holdout_seed": 119, "lidar_mode": "enriched",
        "learning_rate": 1e-3, "smooth_l1_beta": 0.1, "batch_size": 2,
        "splat_radius_px": 1, "splat_sigma_px": 1.0,
        "block_height_min_px": 3, "block_height_max_px": 4,
        "block_width_min_px": 4, "block_width_max_px": 5, "block_max_count": 3,
    }
    dataset = []
    for index in range(4):
        sparse = np.zeros((6, 12, 16), dtype=np.float32)
        sparse[0, ::2, ::2] = 0.1 + index * 0.01
        sparse[5, ::2, ::2] = 1
        dataset.append({
            "image": np.full((3, 12, 16), 0.25, dtype=np.float32),
            "lidar_maps": sparse, "meta": {"sample_id": str(index)},
        })
    monkeypatch.setattr(analysis_tools, "KittiObjectDepthDataset", lambda *args, **kwargs: dataset)
    model = CameraDepthModel()
    paths = {}
    for mode in ("random_points", "spatial_blocks"):
        path = tmp_path / f"{mode}.pt"
        checkpoint_config = dict(config)
        # Exercise compatibility with checkpoints predating holdout_mode.
        if mode == "spatial_blocks":
            checkpoint_config["holdout_mode"] = mode
        torch.save({
            "config": checkpoint_config, "model_state_dict": model.state_dict(),
            "epoch": 2, "step": 6,
        }, path)
        paths[mode] = path
    return config, dataset, paths


def test_cross_evaluation_uses_identical_inputs_per_mode_without_gradients(
    comparison_case, monkeypatch,
):
    config, _, paths = comparison_case
    prepared_batches = []
    prepare = training.prepare_camera_depth_evaluation_batch

    def record_batch(**kwargs):
        batch = prepare(**kwargs)
        prepared_batches.append(batch)
        return batch

    monkeypatch.setattr(training, "prepare_camera_depth_evaluation_batch", record_batch)
    rows = compare_camera_depth_holdout_checkpoints(
        paths, evaluation_config={**config, "batch_size": 1}, device="cpu"
    )

    assert [(row["training_holdout_mode"], row["evaluation_holdout_mode"]) for row in rows] == [
        ("random_points", "random_points"), ("random_points", "spatial_blocks"),
        ("spatial_blocks", "random_points"), ("spatial_blocks", "spatial_blocks"),
    ]
    assert len(prepared_batches) == 8
    for index in range(4):
        for key in prepared_batches[index]:
            np.testing.assert_array_equal(
                prepared_batches[index][key], prepared_batches[index + 4][key]
            )
    # Identical weights and identical inputs must give identical metrics.
    for first, second in ((rows[0], rows[2]), (rows[1], rows[3])):
        for key in ("mae_m", "rmse_m", "splat_mae_m", "splat_coverage", "pixel_count", "holdout_fraction"):
            assert first[key] == second[key]
        assert 0 < first["holdout_fraction"] < 1
    assert rows[1]["evaluation_options"]["block_width_px"] == (4, 5)
    assert all(np.isfinite(row["mae_m"]) for row in rows)


@pytest.mark.parametrize("field, value", [
    ("sample_count", 512), ("split_seed", 20), ("validation_holdout_seed", 120),
    ("max_depth_m", 100.0), ("splat_radius_px", 2),
])
def test_comparison_rejects_incompatible_checkpoint_protocol(comparison_case, field, value):
    config, _, paths = comparison_case
    payload = torch.load(paths["spatial_blocks"], weights_only=False)
    payload["config"][field] = value
    torch.save(payload, paths["spatial_blocks"])

    with pytest.raises(ValueError, match=f"protocol mismatch: {field}"):
        compare_camera_depth_holdout_checkpoints(paths, evaluation_config=config, device="cpu")


@pytest.mark.parametrize("field, value", [
    ("epoch", 3), ("step", 7), ("learning_rate", 0.02), ("batch_size", 1),
])
def test_comparison_rejects_different_training_budgets_and_settings(comparison_case, field, value):
    config, _, paths = comparison_case
    payload = torch.load(paths["spatial_blocks"], weights_only=False)
    if field in ("epoch", "step"):
        payload[field] = value
    else:
        payload["config"][field] = value
    torch.save(payload, paths["spatial_blocks"])

    with pytest.raises(ValueError, match="same epoch and step|different training"):
        compare_camera_depth_holdout_checkpoints(paths, evaluation_config=config, device="cpu")


def test_comparison_rejects_a_mislabeled_checkpoint(comparison_case):
    config, _, paths = comparison_case
    with pytest.raises(ValueError, match="different training holdout mode"):
        compare_camera_depth_holdout_checkpoints(
            {"random_points": paths["spatial_blocks"], "spatial_blocks": paths["random_points"]},
            evaluation_config=config, device="cpu",
        )


def test_comparison_resolves_saved_relative_dataset_paths_from_project_root(
    comparison_case, tmp_path,
):
    config, _, paths = comparison_case
    for path in paths.values():
        payload = torch.load(path, weights_only=False)
        payload["config"]["data_root"] = "."
        torch.save(payload, path)

    rows = compare_camera_depth_holdout_checkpoints(
        paths, evaluation_config=config, device="cpu", project_root=tmp_path
    )

    assert len(rows) == 4


def test_comparison_requires_both_checkpoints():
    with pytest.raises(ValueError, match="requires random_points and spatial_blocks"):
        compare_camera_depth_holdout_checkpoints({}, evaluation_config={}, device="cpu")


def test_sample_analysis_reproduces_spatial_validation_regions(comparison_case):
    config, dataset, _ = comparison_case
    options = training.camera_depth_holdout_options({**config, "holdout_mode": "spatial_blocks"})
    expected = training.prepare_camera_depth_evaluation_batch(
        dataset=dataset, indices=[1], holdout_fraction=0.4, holdout_seed=119,
        splatting_config=SplattingConfig(radius_px=1), **options,
    )
    model = CameraDepthModel()

    analysis = analyze_camera_depth_sample(
        model=model, dataset=dataset, dataset_index=1,
        holdout_fraction=0.4, holdout_seed=119,
        splatting_config=SplattingConfig(radius_px=1), device="cpu", max_depth_m=80,
        **options,
    )

    assert analysis.holdout_mode == "spatial_blocks"
    np.testing.assert_array_equal(analysis.holdout_mask, expected["loss_mask"][0, 0])
    np.testing.assert_array_equal(analysis.holdout_region_mask, expected["holdout_region_mask"][0, 0])
    assert np.isnan(analysis.holdout_target_m[~analysis.holdout_mask]).all()
    assert np.isfinite(analysis.holdout_absolute_error_m[analysis.holdout_mask]).all()
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def _pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def test_holdout_plots_show_regions_and_correct_matrix_positions(comparison_case):
    plt = _pyplot()
    config, dataset, _ = comparison_case
    batch = training.prepare_camera_depth_evaluation_batch(
        dataset=dataset, indices=[0], holdout_fraction=0.4, holdout_seed=119,
        splatting_config=SplattingConfig(radius_px=1),
        **training.camera_depth_holdout_options({**config, "holdout_mode": "spatial_blocks"}),
    )
    preview = plot_camera_depth_holdout_input(batch, max_depth_m=80)
    preview.canvas.draw()
    np.testing.assert_allclose(
        preview.axes[1].images[1].get_array()[..., 3],
        batch["holdout_region_mask"][0, 0] * 0.3,
    )
    rows = [
        {"training_holdout_mode": train, "evaluation_holdout_mode": evaluate,
         "mae_m": value, "rmse_m": value + 1}
        for train, evaluate, value in (
            ("spatial_blocks", "spatial_blocks", 4), ("random_points", "spatial_blocks", 2),
            ("spatial_blocks", "random_points", 3), ("random_points", "random_points", 1),
        )
    ]
    comparison = plot_camera_depth_holdout_comparison(rows)
    comparison.canvas.draw()
    np.testing.assert_array_equal(comparison.axes[0].images[0].get_array(), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(comparison.axes[1].images[0].get_array(), [[2, 3], [4, 5]])
    plt.close(preview)
    plt.close(comparison)
    with pytest.raises(ValueError, match="all four"):
        plot_camera_depth_holdout_comparison(rows[:3])


def _notebook_cell(cell_id):
    path = Path(__file__).resolve().parents[1] / "notebooks/03_camera_depth_training_analysis.ipynb"
    notebook = json.loads(path.read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    return compile("".join(cell["source"]), f"{path.name}:{cell_id}", "exec")


def test_notebook_preview_and_comparison_cells_execute_on_cpu(comparison_case, tmp_path):
    plt = _pyplot()
    config, dataset, paths = comparison_case
    figures = []
    namespace = {
        "Path": Path, "json": json, "plt": plt, "display": figures.append,
        "ROOT": tmp_path, "config": config, "PREVIEW_SAMPLE_INDEX": 0,
        "KittiObjectDepthDataset": lambda *args, **kwargs: dataset,
        "prepare_camera_depth_evaluation_batch": training.prepare_camera_depth_evaluation_batch,
        "camera_depth_holdout_options": training.camera_depth_holdout_options,
        "SplattingConfig": SplattingConfig,
        "plot_camera_depth_holdout_input": plot_camera_depth_holdout_input,
    }
    exec(_notebook_cell("holdout-preview"), namespace)
    assert len(figures) == 2

    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text("\n".join(f"{key}: {value}" for key, value in config.items()))
    output = tmp_path / "analysis/comparison.json"
    namespace.update({
        "RUN_HOLDOUT_COMPARISON": True, "REQUIRE_CUDA": False, "DEVICE": "cpu",
        "HOLDOUT_COMPARISON_CONFIG_PATH": config_path, "HOLDOUT_CHECKPOINTS": paths,
        "HOLDOUT_COMPARISON_OUTPUT": output, "BATCH_SIZE": 1,
        "load_training_config": training.load_training_config,
        "compare_camera_depth_holdout_checkpoints": compare_camera_depth_holdout_checkpoints,
        "plot_camera_depth_holdout_comparison": plot_camera_depth_holdout_comparison,
    })
    exec(_notebook_cell("holdout-comparison"), namespace)
    saved = json.loads(output.read_text())
    assert len(saved["rows"]) == 4
    assert saved["evaluation_config"]["batch_size"] == 1
    assert saved["rows"][1]["evaluation_options"]["block_width_px"] == [4, 5]
    assert len(figures) == 3


@pytest.mark.parametrize("saved_mode", ["random_points", "spatial_blocks"])
def test_notebook_sample_uses_saved_checkpoint_holdout_settings(comparison_case, tmp_path, saved_mode):
    plt = _pyplot()
    config, dataset, paths = comparison_case
    namespace = {
        "ROOT": tmp_path, "config": {**config, "holdout_mode": "spatial_blocks", "width": 99},
        "checkpoint_path": paths[saved_mode], "DEVICE": "cpu",
        "VALIDATION_SAMPLE_POSITION": 0, "plt": plt, "display": lambda figure: None,
        "KittiObjectDepthDataset": lambda *args, **kwargs: dataset,
        "load_camera_depth_checkpoint": analysis_tools.load_camera_depth_checkpoint,
        "split_dataset_indices": training.split_dataset_indices,
        "analyze_camera_depth_sample": analyze_camera_depth_sample,
        "camera_depth_holdout_options": training.camera_depth_holdout_options,
        "SplattingConfig": SplattingConfig,
        "plot_camera_depth_sample": analysis_tools.plot_camera_depth_sample,
    }

    exec(_notebook_cell("sample-analysis"), namespace)

    assert namespace["analysis"].holdout_mode == saved_mode
    assert namespace["sample_config"]["width"] == 16
    if saved_mode == "spatial_blocks":
        assert namespace["analysis"].holdout_region_mask.sum() > namespace["analysis"].holdout_mask.sum()
    else:
        np.testing.assert_array_equal(
            namespace["analysis"].holdout_region_mask, namespace["analysis"].holdout_mask
        )


def test_pilot_configurations_only_differ_in_holdout_and_output():
    config_root = Path(__file__).resolve().parents[1] / "configs/training"
    spatial = training.load_training_config(config_root / "kitti_camera_depth_block_dropout.yaml")
    random = training.load_training_config(config_root / "kitti_camera_depth_random_holdout_512.yaml")

    def shared(config):
        return {key: value for key, value in config.items()
                if key not in ("holdout_mode", "output_dir") and not key.startswith("block_")}

    assert shared(spatial) == shared(random)
