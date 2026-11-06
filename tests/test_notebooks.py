from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


NOTEBOOK_NAMES = (
    "01_project_lidar_on_image.ipynb",
    "02_visualize_hybrid_input.ipynb",
    "03_camera_depth_training_analysis.ipynb",
    "04_lidar_bev_analysis.ipynb",
    "05_lidar_center_overfit.ipynb",
    "06_lidar_center_training_analysis.ipynb",
)


@pytest.mark.parametrize("notebook_name", NOTEBOOK_NAMES)
def test_notebook_is_syntactically_valid(notebook_name: str) -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text())

    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile(
            "".join(cell["source"]),
            f"{notebook_path.name}:cell-{cell_index}",
            "exec",
        )


@pytest.mark.parametrize("notebook_name", NOTEBOOK_NAMES)
def test_notebook_has_no_saved_execution_state(notebook_name: str) -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text())

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_training_notebook_keeps_safe_defaults_and_gpu_cache_cleanup() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / NOTEBOOK_NAMES[2]
    notebook = json.loads(notebook_path.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells[0]["id"] == "cuda-debug-settings"
    assert 'os.environ["CUDA_LAUNCH_BLOCKING"] = "1"' in cells["cuda-debug-settings"]

    flags = {"RUN_TRAINING", "RUN_SPLATTING_ABLATION", "RUN_HOLDOUT_COMPARISON", "RESUME_EXISTING"}
    defaults = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(cells["parameters"]).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in flags
    }
    assert defaults == dict.fromkeys(flags, False)

    memory_assignment = next(
        node for node in ast.parse(cells["parameters"]).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "MINIMUM_FREE_GPU_MEMORY_MIB"
    )
    assert ast.literal_eval(memory_assignment.value) == 2048

    namespace = {"ROOT": Path("/project")}
    output_assignment = next(
        node for node in ast.parse(cells["parameters"]).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "RUN_OUTPUT_DIR"
    )
    exec(compile(ast.Module(body=[output_assignment], type_ignores=[]), "parameters", "exec"), namespace)
    assert namespace["RUN_OUTPUT_DIR"] == (
        namespace["ROOT"] / "results/training/kitti_camera_depth_block_dropout_512_provenance"
    )

    preflight_call = next(
        node for node in ast.walk(ast.parse(cells["gpu-preflight"]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_gpu_preflight"
    )
    cleanup = next(keyword for keyword in preflight_call.keywords if keyword.arg == "release_cached_memory")
    assert ast.literal_eval(cleanup.value) is True


def test_lidar_notebook_keeps_cpu_inspection_and_external_actions_opt_in():
    path = Path(__file__).resolve().parents[1] / "notebooks/04_lidar_bev_analysis.ipynb"
    notebook = json.loads(path.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    flags = {"RUN_ANNOTATION_AUDIT", "RUN_GPU_PROBE"}
    defaults = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(cells["parameters"]).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in flags
    }
    assert defaults == dict.fromkeys(flags, False)
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert ".cpu().eval()" in code
    assert ".step(" not in code and "--download" not in code
    assert "manual_seed_all" not in code


def test_lidar_overfit_notebook_keeps_training_opt_in_and_separate_process():
    path = Path(__file__).resolve().parents[1] / "notebooks/05_lidar_center_overfit.ipynb"
    notebook = json.loads(path.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    defaults = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(cells["parameters"]).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"RUN_OVERFIT", "RESUME_EXISTING", "DEVICE", "EPOCHS"}
    }
    assert defaults == {"RUN_OVERFIT": False, "RESUME_EXISTING": False, "DEVICE": "cuda", "EPOCHS": 200}
    assert "subprocess.Popen" in cells["train-overfit"]
    assert "process.terminate()" in cells["train-overfit"]
    assert "process.kill()" in cells["train-overfit"]


def test_lidar_trainval_notebook_keeps_training_opt_in_and_inspection_in_validation():
    path = Path(__file__).resolve().parents[1] / "notebooks/06_lidar_center_training_analysis.ipynb"
    notebook = json.loads(path.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    defaults = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(cells["parameters"]).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"RUN_TRAINING", "RESUME_EXISTING", "DEVICE", "EPOCHS"}
    }
    assert defaults == {"RUN_TRAINING": False, "RESUME_EXISTING": False, "DEVICE": "cuda", "EPOCHS": 20}
    assert "subprocess.Popen" in cells["train-split"] and "process.terminate()" in cells["train-split"]
    assert "process.kill()" in cells["train-split"]
    assert "saved_split['validation_ids'][VALIDATION_SAMPLE_POSITION]" in cells["inspect-validation"]
    assert "assert sample_id not in saved_split['train_ids']" in cells["inspect-validation"]
