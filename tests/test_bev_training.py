from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from rgb_lidar_fusion import bev_training
from rgb_lidar_fusion.bev_analysis import analyze_bev_fusion_frame, plot_bev_fusion_sample
from rgb_lidar_fusion.bev_checkpoint import load_bev_fusion_checkpoint
from rgb_lidar_fusion.bev_model import BEVFusionConfig
from rgb_lidar_fusion.bev_training import (
    BEVFusionOverfitConfig, prepare_bev_overfit_frames, read_bev_overfit_split, run_bev_fusion_overfit,
)
from rgb_lidar_fusion.camera_bev import BEVGridConfig
from rgb_lidar_fusion.camera_depth_manifest import CAMERA_DEPTH_CHECKPOINT_KIND, build_camera_depth_manifest
from rgb_lidar_fusion.camera_depth_model import CameraDepthModel
from rgb_lidar_fusion.dataset import KittiObjectDepthDataset
from rgb_lidar_fusion.lidar_center_model import LidarCenterDiagnostic
from rgb_lidar_fusion.lidar_center_training import LidarCenterTrainConfig
from rgb_lidar_fusion.lidar_center_trainval import build_lidar_center_split


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "data"
    for name in ("image_2", "calib", "velodyne", "label_2"):
        (root / name).mkdir(parents=True)
    for index in range(6):
        sample_id = f"{index:06d}"
        Image.fromarray(np.full((20, 24, 3), 120 + index, dtype=np.uint8)).save(root / "image_2" / f"{sample_id}.png")
        (root / "calib" / f"{sample_id}.txt").write_text(
            "P2: 10 0 12 0 0 10 10 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\n"
            "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
        )
        (root / "label_2" / f"{sample_id}.txt").write_text("Car 0 0 0 6 6 14 14 2 2 4 0 1 5 0\n")
        np.array([[5., 0., -.2, .8], [5.1, .1, .2, .5], [6., .5, .1, .9]], dtype=np.float32).tofile(
            root / "velodyne" / f"{sample_id}.bin",
        )
    architecture = BEVFusionConfig(
        grid=BEVGridConfig((0., 10.), (-4., 4.), (-3., 3.), 1.), image_shape=(16, 24),
        max_points_per_pillar=4, lidar_channels=2, backbone_channels=(2, 3, 4),
    )
    lidar_config = LidarCenterTrainConfig(data_root=str(root), device="cpu", sample_count=6,
                                         validation_fraction=1/3, channels=2, grid=architecture.grid,
                                         max_points_per_pillar=4)
    split = build_lidar_center_split(lidar_config)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    dataset = KittiObjectDepthDataset(root, image_shape=architecture.image_shape)
    protocol = {"dataset": "kitti_camera_depth", "lidar_mode": "enriched", "height": 16, "width": 24,
                "max_depth_m": 80., "splat_radius_px": 2, "splat_sigma_px": 1.}
    manifest = build_camera_depth_manifest(
        dataset, train_indices=[split["source_ids"].index(s) for s in split["train_ids"]],
        validation_indices=[split["source_ids"].index(s) for s in split["validation_ids"]], protocol=protocol,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        camera = CameraDepthModel(lidar_mode="enriched")
        lidar = LidarCenterDiagnostic(channels=2)
    camera_path, lidar_path = tmp_path / "camera.pt", tmp_path / "lidar.pt"
    torch.save({"kind": CAMERA_DEPTH_CHECKPOINT_KIND, "model_state_dict": camera.state_dict(),
                "data_manifest": manifest, "config": protocol, "epoch": 5, "step": 20}, camera_path)
    torch.save({"kind": "lidar_center_trainval_v1", "model_state_dict": lidar.state_dict(),
                "config": lidar_config.serialized(), "split": split, "epoch": 9, "step": 36}, lidar_path)
    return BEVFusionOverfitConfig(
        data_root=str(root), output_dir=str(tmp_path / "run"), split_path=str(split_path),
        camera_checkpoint=str(camera_path), lidar_checkpoint=str(lidar_path), model_config=architecture,
        sample_ids=tuple(split["train_ids"][:2]), epochs=4, eval_every=1, cpu_threads=1,
    )


def test_configuration_roundtrip_and_defaults():
    config = BEVFusionOverfitConfig.from_mapping(json.loads(Path("configs/training/kitti_bev_fusion_overfit.json").read_text()))
    assert config.device == "cpu" and config.model_config.freeze_encoders
    assert config.minimum_free_memory_mib == 2048
    assert BEVFusionOverfitConfig.from_mapping(config.serialized()).serialized() == config.serialized()


@pytest.mark.parametrize("values", [
    {"sample_ids": []}, {"sample_ids": ["000000"] * 5}, {"sample_ids": ["bad"]},
    {"epochs": 0}, {"device": "auto"}, {"learning_rate": float("nan")}, {"split_path": ""},
    {"model_config": BEVFusionConfig(freeze_encoders=False)},
    {"model_config": BEVFusionConfig(camera_enabled=False)},
])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        BEVFusionOverfitConfig(**values)


def test_cpu_training_freezes_encoders_and_writes_audited_memorization(config):
    events = []
    before_rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    result = run_bev_fusion_overfit(config, emit=events.append)
    assert torch.get_num_threads() == threads
    torch.testing.assert_close(torch.random.get_rng_state(), before_rng, atol=0, rtol=0)
    model, payload = load_bev_fusion_checkpoint(result["checkpoint"])
    assert result["epochs_completed"] == 4 and payload["step"] == 8
    assert payload["training_metadata"]["task"] == "bev_fusion_overfit"
    assert payload["training_metadata"]["train_ids"] == list(config.sample_ids)
    assert payload["training_metadata"]["evaluation_ids"] == list(config.sample_ids)
    assert "no held-out" in payload["training_metadata"]["evaluation_scope"]
    assert payload["pretraining"]["camera"]["provenance"] == "recorded_at_training"
    assert payload["pretraining"]["source_head_and_optimizer_loaded"] is False
    assert len([e for e in events if e["event"] == "step_completed"]) == 8
    assert all(e["event"] != "preflight" for e in events)
    assert result["resources"]["peak_allocated_mib"] is None
    assert result["metrics"][-1]["train_eval_loss"] < result["metrics"][0]["train_eval_loss"]
    camera = torch.load(config.camera_checkpoint, weights_only=True)
    lidar = torch.load(config.lidar_checkpoint, weights_only=True)
    model.train()
    for name, module in (("camera", model.camera), ("pfn", model.pfn), ("context", model.context)):
        assert not module.training and all(not p.requires_grad for p in module.parameters())
        expected = camera["model_state_dict"] if name == "camera" else {
            k.removeprefix(name + "."): v for k, v in lidar["model_state_dict"].items() if k.startswith(name + ".")
        }
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    saved = json.loads((Path(config.output_dir) / "pretraining.json").read_text())
    assert saved == payload["pretraining"]
    assert json.loads((Path(config.output_dir) / "metrics.json").read_text()) == result["metrics"]


def test_resume_exact_adam_order_without_source_weights_and_repairs_sidecars(config, tmp_path):
    whole = run_bev_fusion_overfit(config)
    partial_config = replace(config, epochs=2, output_dir=str(tmp_path / "resumed"))
    partial = run_bev_fusion_overfit(partial_config)
    Path(config.camera_checkpoint).unlink()
    Path(config.lidar_checkpoint).unlink()
    (Path(partial_config.output_dir) / "metrics.json").write_text("[]")
    restored = run_bev_fusion_overfit(replace(partial_config, epochs=4), resume_from=partial["checkpoint"])
    first, _ = load_bev_fusion_checkpoint(whole["checkpoint"])
    second, _ = load_bev_fusion_checkpoint(restored["checkpoint"])
    assert whole["metrics"] == restored["metrics"]
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert json.loads((Path(partial_config.output_dir) / "metrics.json").read_text()) == restored["metrics"]
    with pytest.raises(ValueError, match="configuration"):
        run_bev_fusion_overfit(replace(partial_config, epochs=5, learning_rate=.02), resume_from=partial["checkpoint"])


def test_validation_ids_and_changed_files_fail_before_device_or_output(config, monkeypatch):
    from rgb_lidar_fusion import camera_depth_analysis

    def unexpected(**kwargs):
        pytest.fail("preflight must not run before membership/content checks")

    monkeypatch.setattr(camera_depth_analysis, "run_gpu_preflight", unexpected)
    split = json.loads(Path(config.split_path).read_text())
    with pytest.raises(ValueError, match="training split"):
        run_bev_fusion_overfit(replace(config, device="cuda", sample_ids=(split["validation_ids"][0],)))
    label = Path(config.data_root) / "label_2" / f"{config.sample_ids[0]}.txt"
    label.write_text(label.read_text() + "\n")
    with pytest.raises(ValueError, match="file changed"):
        run_bev_fusion_overfit(replace(config, device="cuda"))
    assert not Path(config.output_dir).exists()


def test_legacy_camera_weights_are_rejected_without_exception(config):
    payload = torch.load(config.camera_checkpoint, weights_only=True)
    payload.pop("kind")
    payload.pop("data_manifest")
    torch.save(payload, config.camera_checkpoint)
    with pytest.raises(ValueError, match="legacy camera"):
        run_bev_fusion_overfit(config)


def test_preflight_failure_has_no_cuda_cleanup_or_retry(config, monkeypatch):
    from rgb_lidar_fusion import camera_depth_analysis

    @dataclass
    class Blocked:
        ready: bool = False

        def format(self):
            return "GPU deliberately unavailable"

    monkeypatch.setattr(camera_depth_analysis, "run_gpu_preflight", lambda **kwargs: Blocked())
    def unexpected(*args, **kwargs):
        pytest.fail("CUDA must not be touched after failed preflight")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", unexpected)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", unexpected)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    events = []
    with pytest.raises(RuntimeError, match="preflight failed"):
        run_bev_fusion_overfit(replace(config, device="cuda"), emit=events.append)
    assert events[-1]["event"] == "failed"
    assert not Path(config.output_dir).exists()


def test_cpu_inspection_no_mutation_and_real_plot(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = run_bev_fusion_overfit(replace(config, epochs=1))
    model, _ = load_bev_fusion_checkpoint(result["checkpoint"])
    model.train()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    frame = prepare_bev_overfit_frames(config, read_bev_overfit_split(config))[0]
    sample = analyze_bev_fusion_frame(model, frame)
    assert model.training and not model.pfn.training
    assert sample.image.shape == (*config.model_config.image_shape, 3)
    assert all(values.shape == config.grid.shape for values in sample.maps.values())
    np.testing.assert_array_equal(sample.maps["lidar_count"], frame.inputs["point_count"][0, 0])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    figure = plot_bev_fusion_sample(sample, grid=config.grid)
    figure.canvas.draw()
    assert len(figure.axes) == 23  # 12 panels plus 11 color bars.
    plt.close(figure)


def test_notebook_cells_execute_on_cpu_and_training_is_opt_in(config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/07_bev_fusion_overfit.ipynb").read_text())
    cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"]}
    from rgb_lidar_fusion.bev_checkpoint import initialize_bev_fusion_encoders
    from rgb_lidar_fusion.bev_model import BEVFusionDiagnostic
    figures = []
    namespace = {
        "config": config, "torch": torch, "json": json, "plt": plt, "display": figures.append,
        "BEVFusionDiagnostic": BEVFusionDiagnostic, "BEVFusionOverfitConfig": BEVFusionOverfitConfig,
        "read_bev_overfit_split": read_bev_overfit_split,
        "prepare_bev_overfit_frames": prepare_bev_overfit_frames,
        "initialize_bev_fusion_encoders": initialize_bev_fusion_encoders,
        "analyze_bev_fusion_frame": analyze_bev_fusion_frame, "plot_bev_fusion_sample": plot_bev_fusion_sample,
        "load_bev_fusion_checkpoint": load_bev_fusion_checkpoint,
        "RUN_OVERFIT": False, "INSPECT_IDS": list(config.sample_ids[:1]),
        "OUTPUT_DIR": Path(config.output_dir),
        "ANALYSIS_DIR": Path(config.output_dir).parent / "analysis",
        "CHECKPOINT_PATH": Path(config.output_dir) / "checkpoints/latest.pt",
    }
    exec(compile(cells["audit-inspect"], "audit-inspect", "exec"), namespace)
    exec(compile(cells["train-overfit"], "train-overfit", "exec"), namespace)
    assert not Path(config.output_dir).exists() and len(figures) == 1
    initial_path = namespace["ANALYSIS_DIR"] / f"initial_{config.sample_ids[0]}.png"
    with Image.open(initial_path) as image:
        assert image.format == "PNG" and image.width > 1000 and image.height > 1000
    run_bev_fusion_overfit(replace(config, epochs=1))
    exec(compile(cells["curves"], "curves", "exec"), namespace)
    exec(compile(cells["inspect-checkpoint"], "inspect-checkpoint", "exec"), namespace)
    assert len(figures) == 4  # Initial plot, metrics dict, curves, fitted plot.
    trained_path = namespace["ANALYSIS_DIR"] / f"epoch_0001_{config.sample_ids[0]}.png"
    with Image.open(trained_path) as image:
        assert image.format == "PNG" and image.width > 1000 and image.height > 1000
    assert initial_path.is_file()


def test_resume_and_output_safeguards(config):
    result = run_bev_fusion_overfit(replace(config, epochs=1))
    with pytest.raises(ValueError, match="not empty"):
        run_bev_fusion_overfit(config)
    with pytest.raises(ValueError, match="latest.pt"):
        run_bev_fusion_overfit(config, resume_from="other.pt")
    payload = torch.load(result["checkpoint"], weights_only=True)
    payload["training_metadata"]["task"] = "some_other_task"
    torch.save(payload, result["checkpoint"])
    with pytest.raises(ValueError, match="task"):
        run_bev_fusion_overfit(config, resume_from=result["checkpoint"])


@pytest.mark.parametrize("outcome", ["success", "failure", "interrupt"])
def test_notebook_child_process_logging_and_interrupt_cleanup(tmp_path, outcome):
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/07_bev_fusion_overfit.ipynb").read_text())
    code = "".join(next(c for c in notebook["cells"] if c["id"] == "train-overfit")["source"])
    calls = []

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

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if outcome == "interrupt" and self.returncode is None:
                raise subprocess.TimeoutExpired("fake-child", timeout)
            self.returncode = 0 if outcome == "success" else 2
            return self.returncode

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append(("terminate",))

        def kill(self):
            calls.append(("kill",))
            self.returncode = -9

    def popen(command, **kwargs):
        assert kwargs["env"]["CUDA_LAUNCH_BLOCKING"] == "1"
        assert "train_bev_fusion_overfit.py" in command[1]
        return Child()

    namespace = {
        "RUN_OVERFIT": True, "RESUME_EXISTING": False, "sys": sys, "ROOT": root,
        "CONFIG_PATH": root / "configs/training/kitti_bev_fusion_overfit.json",
        "DEVICE": "cuda", "EPOCHS": 200, "OUTPUT_DIR": tmp_path / "run", "os": os,
        "LOG_PATH": tmp_path / "run.jsonl",
        "subprocess": SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                      TimeoutExpired=subprocess.TimeoutExpired),
    }
    if outcome == "success":
        exec(compile(code, "train-overfit", "exec"), namespace)
    else:
        with pytest.raises(KeyboardInterrupt if outcome == "interrupt" else RuntimeError):
            exec(compile(code, "train-overfit", "exec"), namespace)
    assert "step_completed" in namespace["LOG_PATH"].read_text()
    assert not namespace["OUTPUT_DIR"].exists()
    if outcome == "interrupt":
        assert calls == [("terminate",), ("wait", 10), ("kill",), ("wait", None)]


def test_cli_runs_the_cpu_diagnostic(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(replace(config, epochs=1).serialized()))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/train_bev_fusion_overfit.py"), "--config", str(path)],
        cwd=root, env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1]["event"] == "completed"
    assert events[-1]["epochs_completed"] == 1
    assert any(e["event"] == "step_completed" for e in events)
