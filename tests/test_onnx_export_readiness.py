from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_onnx_artifact_policy_is_documented_before_export_scripts_are_enabled():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependency_groups = pyproject.get("dependency-groups", {})
    pdm_scripts = pyproject.get("tool", {}).get("pdm", {}).get("scripts", {})

    assert "onnx" not in dependency_groups
    assert "export_onnx" not in pdm_scripts
    assert "validate_onnx" not in pdm_scripts

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    results_readme = (ROOT / "results" / "README.md").read_text(encoding="utf-8")
    onnx_plan = (ROOT / "docs" / "onnx-export-validation.md").read_text(encoding="utf-8")

    assert "results/**/*.onnx" in gitignore
    assert "onnx/" in results_readme
    assert "NMS" in onnx_plan
    assert "outside the ONNX graph" in onnx_plan
