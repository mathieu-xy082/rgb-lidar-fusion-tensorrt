import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("docs/onboarding/scripts/render_kitti_sequence.py")
spec = importlib.util.spec_from_file_location("render_kitti_sequence", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
render_kitti_sequence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_kitti_sequence)


def test_render_sample_requests_structured_splatted_npz_output(tmp_path, monkeypatch):
    data_root = tmp_path / "training"
    for subdir, suffix in (("image_2", ".png"), ("velodyne", ".bin"), ("calib", ".txt")):
        path = data_root / subdir / f"000000{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    output_root = tmp_path / "results"
    commands = []

    def record(command: list[str]) -> None:
        commands.append(command)

    monkeypatch.setattr(render_kitti_sequence, "run", record)

    frame = render_kitti_sequence.render_sample("000000", data_root, output_root, alpha=0.45)

    sample_out = output_root / "kitti_000000"
    splatted_npz = sample_out / "splatted_maps_000000.npz"
    assert frame["dir"] == str(sample_out)
    smoke_command = commands[1]
    assert "--splatted-output" in smoke_command
    assert smoke_command[smoke_command.index("--splatted-output") + 1] == str(splatted_npz)
    assert "--splat-radius-px" in smoke_command
    assert "--splat-sigma-px" in smoke_command
