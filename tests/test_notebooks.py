from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


NOTEBOOK_NAMES = (
    "01_project_lidar_on_image.ipynb",
    "02_visualize_hybrid_input.ipynb",
    "03_camera_depth_training_analysis.ipynb",
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

    preflight_call = next(
        node for node in ast.walk(ast.parse(cells["gpu-preflight"]))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_gpu_preflight"
    )
    cleanup = next(keyword for keyword in preflight_call.keywords if keyword.arg == "release_cached_memory")
    assert ast.literal_eval(cleanup.value) is True
