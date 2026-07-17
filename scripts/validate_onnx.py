"""Validate ONNX shape metadata and PyTorch/ONNX Runtime parity."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_onnx import export_baseline_onnx

EXPECTED_INPUTS = {
    "rgb": [2, 3, 32, 48],
    "lidar_maps": [2, 8, 32, 48],
}
EXPECTED_OUTPUTS = {"prediction": [2, 4]}
DEFAULT_ONNX_PATH = Path("results/onnx/baseline_fusion.onnx")


def _shape_from_value_info(value_info) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def assert_graph_shapes(model_path: Path) -> None:
    import onnx

    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    inputs = {value.name: _shape_from_value_info(value) for value in model.graph.input}
    outputs = {value.name: _shape_from_value_info(value) for value in model.graph.output}

    if inputs != EXPECTED_INPUTS:
        raise AssertionError(f"Unexpected ONNX inputs: {inputs!r}")
    if outputs != EXPECTED_OUTPUTS:
        raise AssertionError(f"Unexpected ONNX outputs: {outputs!r}")


def assert_runtime_parity(model_path: Path, *, tolerance: float = 1e-5) -> float:
    import numpy as np
    import onnxruntime as ort
    import torch

    from rgb_lidar_fusion.baseline_model import BaselineFusionModel, ENRICHED_LIDAR_CHANNELS

    torch.manual_seed(0)
    model = BaselineFusionModel(output_dim=4, lidar_mode="enriched").eval()
    rgb = torch.linspace(0.0, 1.0, steps=2 * 3 * 32 * 48, dtype=torch.float32).reshape(2, 3, 32, 48)
    lidar_maps = torch.linspace(
        -1.0,
        1.0,
        steps=2 * ENRICHED_LIDAR_CHANNELS * 32 * 48,
        dtype=torch.float32,
    ).reshape(2, ENRICHED_LIDAR_CHANNELS, 32, 48)

    with torch.no_grad():
        expected = model(rgb, lidar_maps).detach().cpu().numpy()

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["prediction"],
        {"rgb": rgb.cpu().numpy(), "lidar_maps": lidar_maps.cpu().numpy()},
    )[0]
    max_abs_diff = float(np.max(np.abs(expected - actual)))
    if max_abs_diff > tolerance:
        raise AssertionError(f"ONNX parity failed: max_abs_diff={max_abs_diff} > tolerance={tolerance}")
    return max_abs_diff


def validate_baseline_onnx(model_path: Path = DEFAULT_ONNX_PATH, *, tolerance: float = 1e-5) -> float:
    if not model_path.exists():
        export_baseline_onnx(model_path)
    assert_graph_shapes(model_path)
    return assert_runtime_parity(model_path, tolerance=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    max_abs_diff = validate_baseline_onnx(args.model, tolerance=args.tolerance)
    print(f"ONNX shape and parity validation passed: max_abs_diff={max_abs_diff:.8g}")


if __name__ == "__main__":
    main()
