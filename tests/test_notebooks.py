from __future__ import annotations

import json
from pathlib import Path


def test_camera_depth_analysis_notebook_is_syntactically_valid() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "03_camera_depth_training_analysis.ipynb"
    )
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
