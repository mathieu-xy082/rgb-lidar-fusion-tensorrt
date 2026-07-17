from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_onnx_artifact_policy_remains_in_place_after_export_scripts_are_enabled():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_groups = pyproject.get("dependency-groups", {})
    pdm_scripts = pyproject.get("tool", {}).get("pdm", {}).get("scripts", {})

    assert dependency_groups["onnx"] == [
        "onnx>=1.16",
        "onnxruntime>=1.18",
        "onnxscript>=0.1",
    ]
    assert pdm_scripts["export_onnx"] == "python scripts/export_onnx.py"
    assert pdm_scripts["validate_onnx"] == "python scripts/validate_onnx.py"

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    results_readme = (ROOT / "results" / "README.md").read_text(encoding="utf-8")
    onnx_plan = (ROOT / "docs" / "onnx-export-validation.md").read_text(encoding="utf-8")

    assert "results/**/*.onnx" in gitignore
    assert "results/**/*.onnx.data" in gitignore
    assert "onnx/" in results_readme
    assert "NMS" in onnx_plan
    assert "outside the ONNX graph" in onnx_plan


def test_onnx_plan_names_the_reviewed_baseline_contract_for_script_validation():
    onnx_plan = (ROOT / "docs" / "onnx-export-validation.md").read_text(encoding="utf-8")

    assert "Reviewed baseline contract used by scripts" in onnx_plan
    assert "rgb" in onnx_plan
    assert "lidar_maps" in onnx_plan
    assert "prediction" in onnx_plan
    assert "no NMS" in onnx_plan
