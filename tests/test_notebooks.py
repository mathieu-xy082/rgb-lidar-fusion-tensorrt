from __future__ import annotations

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


@pytest.mark.parametrize("notebook_name", NOTEBOOK_NAMES[:2])
def test_exploration_notebook_has_no_saved_execution_state(notebook_name: str) -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / notebook_name
    notebook = json.loads(notebook_path.read_text())

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
