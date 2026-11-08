from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_bev_pilot import config as config
from test_bev_training import config as overfit_config
from rgb_lidar_fusion.bev_analysis import plot_bev_fusion_sample
from rgb_lidar_fusion.bev_pilot import VARIANTS, prepare_bev_fusion_pilot_pair, read_bev_pilot_split
from rgb_lidar_fusion.bev_pilot_analysis import (
    analyze_bev_pilot_validation, compare_bev_pilot_runs, load_bev_pilot_run, plot_bev_pilot_curves,
)
from rgb_lidar_fusion.bev_trainval import run_bev_fusion_pilot, validate_bev_pilot_resume
from rgb_lidar_fusion.lidar_center_training import _write_json


@pytest.fixture
def pair(config):
    config = replace(config, epochs=1)
    results = {name: run_bev_fusion_pilot(config, variant=name) for name in VARIANTS}
    return config, {name: Path(result["checkpoint"]) for name, result in results.items()}


def _pyplot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def test_complete_matched_report_and_curves_are_cpu_only(pair, monkeypatch):
    config, paths = pair
    def unexpected(*args, **kwargs):
        pytest.fail("analysis must not touch CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", unexpected)
    report = compare_bev_pilot_runs(paths["lidar_only"], paths["fusion"])
    assert report["complete"] and report["live_dataset_files_verified"]
    assert report["train_count"] == 4 and report["validation_count"] == 2
    runs = {}
    for name, path in paths.items():
        model, saved, payload = load_bev_pilot_run(path)
        assert next(model.parameters()).device.type == "cpu" and not model.training
        assert saved.serialized() == config.serialized()
        best = min(payload["metrics"], key=lambda r: r["val_loss"])
        assert report["variants"][name]["best"]["epoch"] == best["epoch"]
        assert report["variants"][name]["best"]["loss"] == best["val_loss"]
        assert report["variants"][name]["final"]["epoch"] == config.epochs
        runs[name] = payload
    plt = _pyplot()
    figure = plot_bev_pilot_curves(runs)
    figure.canvas.draw()
    assert len(figure.axes) == 4
    plt.close(figure)


@pytest.mark.parametrize("change,error", [
    ("learning_rate", "protocol"), ("initialization", "initialization"),
    ("source_sha", "sources"), ("matching", "matching"),
    ("precision", "policy"), ("domain", "domain"), ("counts", "counts"),
])
def test_incompatible_or_corrupted_runs_are_not_compared(pair, change, error):
    _, paths = pair
    payload = torch.load(paths["fusion"], map_location="cpu", weights_only=True)
    metadata = payload["training_metadata"]
    if change == "learning_rate":
        metadata["runner_config"]["learning_rate"] *= 2
    elif change == "initialization":
        metadata["initialization"]["new_blocks_sha256"] = "0" * 64
    elif change == "source_sha":
        payload["pretraining"]["lidar"]["sha256"] = "0" * 64
    elif change == "matching":
        metadata["matching"] = "different matching policy"
    elif change == "precision":
        metadata["precision"] = "float16"
    elif change == "domain":
        payload["metrics"][-1]["val_per_frame"].pop()
    else:
        payload["metrics"][-1]["val_tp"] += 1
    torch.save(payload, paths["fusion"])
    with pytest.raises(ValueError, match=error):
        compare_bev_pilot_runs(paths["lidar_only"], paths["fusion"])


def test_partial_runs_allow_curves_but_not_a_final_report(pair):
    _, paths = pair
    runs = {}
    for name, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["training_metadata"]["runner_config"]["epochs"] = 2
        torch.save(payload, path)
        runs[name] = load_bev_pilot_run(path)[2]
    plt = _pyplot()
    figure = plot_bev_pilot_curves(runs)
    figure.canvas.draw()
    plt.close(figure)
    with pytest.raises(ValueError, match="complete"):
        compare_bev_pilot_runs(paths["lidar_only"], paths["fusion"])


def test_overlaid_curves_refuse_different_domains_or_variant_labels(pair):
    _, paths = pair
    runs = {name: load_bev_pilot_run(path)[2] for name, path in paths.items()}
    with pytest.raises(ValueError, match="labels"):
        plot_bev_pilot_curves({"fusion": runs["lidar_only"]})
    runs["fusion"]["metrics"][0]["val_per_frame"].pop()
    with pytest.raises(ValueError, match="domains"):
        plot_bev_pilot_curves(runs)


def test_validation_selection_only_uses_saved_validation_and_saved_best_weights(pair, monkeypatch):
    config, paths = pair
    Path(config.camera_checkpoint).unlink()
    Path(config.lidar_checkpoint).unlink()
    _, _, payload = load_bev_pilot_run(paths["fusion"])
    saved_val_id = payload["split"]["validation_ids"][0]
    import rgb_lidar_fusion.bev_pilot_analysis as analysis
    original = analysis.analyze_bev_fusion_frame
    observed = []
    def checked(model, frame, **kwargs):
        observed.append(frame.sample_id)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, payload["best"]["model_state_dict"][name], rtol=0, atol=0)
        return original(model, frame, **kwargs)
    monkeypatch.setattr(analysis, "analyze_bev_fusion_frame", checked)
    result = analyze_bev_pilot_validation(paths["fusion"], selection="best")
    assert result.sample.sample_id == saved_val_id and observed == [saved_val_id]
    assert saved_val_id not in payload["split"]["train_ids"]
    assert result.checkpoint_epoch == payload["best"]["epoch"]
    assert result.selection == "best" and result.variant == "fusion"


@pytest.mark.parametrize("position", [-1, True, 2])
def test_validation_position_is_checked(pair, position):
    _, paths = pair
    with pytest.raises(ValueError, match="validation_position"):
        analyze_bev_pilot_validation(paths["fusion"], validation_position=position)


def test_changed_data_is_detected_and_best_file_is_not_a_complete_run(pair):
    config, paths = pair
    best_path = paths["fusion"].with_name("best.pt")
    with pytest.raises(ValueError, match="purpose"):
        load_bev_pilot_run(best_path)
    path = Path(config.data_root) / "calib/000000.txt"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="file changed"):
        compare_bev_pilot_runs(paths["lidar_only"], paths["fusion"])
    report = compare_bev_pilot_runs(paths["lidar_only"], paths["fusion"], verify_files=False)
    assert not report["live_dataset_files_verified"]


def _cells():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/08_bev_fusion_training_analysis.ipynb").read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def test_notebook_cpu_audit_curves_comparison_and_png_inspection(pair, tmp_path):
    config, paths = pair
    plt = _pyplot()
    displayed = []
    namespace = {
        "config": config, "VARIANT": "fusion", "RESUME_EXISTING": False,
        "RUN_TRAINING": False, "RUN_COMPARISON": False,
        "read_bev_pilot_split": read_bev_pilot_split,
        "prepare_bev_fusion_pilot_pair": prepare_bev_fusion_pilot_pair,
        "load_bev_pilot_run": load_bev_pilot_run, "validate_bev_pilot_resume": validate_bev_pilot_resume,
        "plot_bev_pilot_curves": plot_bev_pilot_curves, "compare_bev_pilot_runs": compare_bev_pilot_runs,
        "analyze_bev_pilot_validation": analyze_bev_pilot_validation,
        "plot_bev_fusion_sample": plot_bev_fusion_sample, "_write_json": _write_json,
        "VARIANTS": VARIANTS, "display": displayed.append, "plt": plt,
        "ANALYSIS_DIR": tmp_path / "analysis", "CHECKPOINT_PATH": paths["fusion"],
        "VALIDATION_SAMPLE_POSITION": 0, "INSPECT_SELECTION": "best",
    }
    cells = _cells()
    exec(compile(cells["audit"], "audit", "exec"), namespace)
    exec(compile(cells["train-pilot"], "train-pilot", "exec"), namespace)
    exec(compile(cells["compare-runs"], "compare-runs", "exec"), namespace)
    assert not namespace["ANALYSIS_DIR"].exists()
    namespace["RESUME_EXISTING"] = True
    exec(compile(cells["audit"], "audit", "exec"), namespace)
    exec(compile(cells["curves"], "curves", "exec"), namespace)
    namespace["RUN_COMPARISON"] = True
    exec(compile(cells["compare-runs"], "compare-runs", "exec"), namespace)
    exec(compile(cells["inspect-validation"], "inspect-validation", "exec"), namespace)
    assert len(displayed) == 3
    assert json.loads((namespace["ANALYSIS_DIR"] / "comparison.json").read_text())["complete"]
    images = list(namespace["ANALYSIS_DIR"].glob("*.png"))
    assert len(images) == 2
    from PIL import Image
    for path in images:
        with Image.open(path) as image:
            assert image.format == "PNG" and image.width > 1000


@pytest.mark.parametrize("outcome", ["success", "failure", "interrupt"])
def test_notebook_launches_effective_config_and_cleans_up_child(config, tmp_path, outcome):
    calls = []
    root = Path(__file__).resolve().parents[1]
    launch_path = tmp_path / "analysis/launch_fusion.json"
    class Child:
        returncode = None
        def __enter__(self):
            def lines():
                yield '{"event": "step_completed", "step": 1}\n'
                if outcome == "interrupt":
                    raise KeyboardInterrupt
            self.stdout = lines()
            return self
        def __exit__(self, *args):
            return False
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if outcome == "interrupt" and self.returncode is None:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = 0 if outcome == "success" else 2
            return self.returncode
        def terminate(self):
            calls.append(("terminate",))
        def kill(self):
            calls.append(("kill",))
            self.returncode = -9
    def popen(command, **kwargs):
        assert command[-2:] == ["--variant", "fusion"]
        assert str(launch_path) in command
        assert json.loads(launch_path.read_text()) == config.serialized()
        assert kwargs["env"]["CUDA_LAUNCH_BLOCKING"] == "1"
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(root / "src")
        return Child()
    namespace = {
        "RUN_TRAINING": True, "RESUME_EXISTING": False,
        "RUN_DIR": tmp_path / "new_run/fusion", "config": config,
        "ROOT": root, "VARIANT": "fusion", "LAUNCH_CONFIG_PATH": launch_path,
        "LOG_PATH": tmp_path / "pilot_fusion.jsonl", "os": os, "sys": sys,
        "_write_json": _write_json,
        "subprocess": SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                      TimeoutExpired=subprocess.TimeoutExpired),
    }
    code = compile(_cells()["train-pilot"], "train-pilot", "exec")
    if outcome == "success":
        exec(code, namespace)
    else:
        with pytest.raises(KeyboardInterrupt if outcome == "interrupt" else RuntimeError):
            exec(code, namespace)
    assert "step_completed" in namespace["LOG_PATH"].read_text()
    assert not namespace["RUN_DIR"].exists()
    if outcome == "interrupt":
        assert calls == [("terminate",), ("wait", 10), ("kill",), ("wait", None)]


def test_notebook_real_cpu_child_honors_resolved_overrides(config, tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = replace(config, epochs=1, learning_rate=.002, output_dir=str(tmp_path / "actual_pilot"))
    run_dir = config.output_for_variant("lidar_only")
    namespace = {
        "RUN_TRAINING": True, "RESUME_EXISTING": False, "config": config,
        "RUN_DIR": run_dir, "ROOT": root, "VARIANT": "lidar_only",
        "LAUNCH_CONFIG_PATH": tmp_path / "analysis/launch_lidar_only.json",
        "LOG_PATH": tmp_path / "actual_pilot_lidar_only.jsonl", "os": os, "sys": sys,
        "_write_json": _write_json, "subprocess": subprocess,
    }
    exec(compile(_cells()["train-pilot"], "train-pilot", "exec"), namespace)
    _, saved, payload = load_bev_pilot_run(run_dir / "checkpoints/latest.pt")
    assert payload["epoch"] == 1 and saved.learning_rate == .002
    assert saved.serialized() == config.serialized()
    events = [json.loads(line) for line in namespace["LOG_PATH"].read_text().splitlines()]
    assert events[-1]["event"] == "completed"
