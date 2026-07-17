from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pdm_exposes_onnx_dependency_group_and_scripts():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dependency_groups = pyproject["dependency-groups"]
    pdm_scripts = pyproject["tool"]["pdm"]["scripts"]

    assert dependency_groups["onnx"] == [
        "onnx>=1.16",
        "onnxruntime>=1.18",
        "onnxscript>=0.1",
    ]
    assert pdm_scripts["export_onnx"] == "python scripts/export_onnx.py"
    assert pdm_scripts["validate_onnx"] == "python scripts/validate_onnx.py"


def test_export_and_validation_scripts_keep_generated_models_under_ignored_results_path():
    export_script = (ROOT / "scripts" / "export_onnx.py").read_text(encoding="utf-8")
    validate_script = (ROOT / "scripts" / "validate_onnx.py").read_text(encoding="utf-8")

    assert "results/onnx/baseline_fusion.onnx" in export_script
    assert "results/onnx/baseline_fusion.onnx" in validate_script
    assert "BaselineFusionModel" in export_script
    assert "BaselineFusionModel" in validate_script
    assert "opset_version=18" in export_script
    assert "dynamo=False" in export_script
    assert "warnings.filterwarnings" in export_script
