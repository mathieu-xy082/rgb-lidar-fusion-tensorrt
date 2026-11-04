"""Bounded memorization diagnostic for one to four explicitly selected KITTI frames.

Training and evaluation deliberately use the same IDs. This is neither a
validation split nor an estimate of generalization or 3D detection AP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .calibration import parse_kitti_calibration_file
from .camera_bev import BEVGridConfig
from .kitti_labels import load_kitti_labels
from .lidar_center_model import LidarCenterDiagnostic, decode_lidar_centers, lidar_center_loss
from .lidar_center_targets import build_lidar_center_targets
from .lidar_pillars import build_lidar_pillars
from .project_lidar import load_velodyne_bin


@dataclass(frozen=True)
class LidarCenterOverfitConfig:
    data_root: str = "data/kitti/training"
    output_dir: str = "results/training/kitti_lidar_center_overfit_4"
    sample_ids: tuple[str, ...] = ("000006", "000036", "000123", "000366")
    grid: BEVGridConfig = BEVGridConfig((0., 50.), (-25., 25.), (-3., 3.), .2)
    epochs: int = 100
    eval_every: int = 5
    learning_rate: float = .001
    seed: int = 13
    channels: int = 32
    max_points_per_pillar: int = 64
    cpu_threads: int = 2
    device: str = "cpu"
    minimum_free_memory_mib: int = 2048
    score_threshold: float = .3
    top_k: int = 50
    match_radius_m: float = 1.
    max_grad_norm: float = 10.

    def __post_init__(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda; there is no automatic fallback.")
        for name in ("data_root", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path string.")
        if not isinstance(self.grid, BEVGridConfig):
            raise ValueError("grid must be a BEVGridConfig.")
        for name in ("epochs", "eval_every", "channels", "max_points_per_pillar", "cpu_threads",
                     "minimum_free_memory_mib", "top_k", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer in its valid range.")
        if not isinstance(self.sample_ids, (list, tuple)) or not 1 <= len(self.sample_ids) <= 4:
            raise ValueError("this overfit diagnostic requires one to four explicit sample_ids.")
        if any(not isinstance(s, str) or len(s) != 6 or not s.isascii() or not s.isdigit() for s in self.sample_ids):
            raise ValueError("sample_ids must contain six-digit KITTI IDs.")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique.")
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))
        for name in ("learning_rate", "match_radius_m", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if isinstance(self.score_threshold, bool) or not isinstance(self.score_threshold, (int, float)) or not (
            math.isfinite(self.score_threshold) and 0 < self.score_threshold < 1
        ):
            raise ValueError("score_threshold must lie strictly between zero and one.")

    @classmethod
    def from_mapping(cls, values: dict):
        values = dict(values)
        try:
            if "grid" in values:
                values["grid"] = BEVGridConfig(**values["grid"])
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid overfit configuration: {exc}") from exc

    def serialized(self):
        values = asdict(self)
        values["data_root"] = str(Path(self.data_root).resolve())
        values["output_dir"] = str(Path(self.output_dir).resolve())
        # Normalize tuple/list differences before comparing resumed configurations.
        return json.loads(json.dumps(values))


@dataclass(frozen=True)
class _Frame:
    sample_id: str
    inputs: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    centers_xy: np.ndarray
    fingerprint: dict[str, str]


def _prepare_frame(config, sample_id):
    from PIL import Image

    root = Path(config.data_root)
    paths = {name: root / name / f"{sample_id}.{suffix}" for name, suffix in (
        ("image_2", "png"), ("calib", "txt"), ("velodyne", "bin"), ("label_2", "txt"),
    )}
    with Image.open(paths["image_2"]) as image:
        shape = (image.height, image.width)
    targets = build_lidar_center_targets(
        load_kitti_labels(paths["label_2"]),
        calibration=parse_kitti_calibration_file(paths["calib"]), image_shape=shape, grid=config.grid,
    )
    if not len(targets.centers_vehicle) or not targets.supervision_mask.any():
        raise ValueError(f"{sample_id}: overfit frames must contain a supervised Car center in the ROI.")
    pillars = build_lidar_pillars(
        [load_velodyne_bin(paths["velodyne"])], grid=config.grid,
        max_points_per_pillar=config.max_points_per_pillar, seed=config.seed,
    )
    if not pillars.point_mask.any():
        raise ValueError(f"{sample_id}: no retained LiDAR points in the ROI.")
    fingerprint = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            fingerprint[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return _Frame(
        sample_id, {name: torch.from_numpy(getattr(pillars, name)) for name in (
            "features", "point_mask", "indices", "point_count",
        )}, {name: torch.from_numpy(getattr(targets, name)[None]) for name in (
            "heatmap", "offset_xy", "center_mask", "supervision_mask",
        )}, targets.centers_vehicle[:, :2].copy(), fingerprint,
    )


def match_lidar_centers(predicted: np.ndarray, target_xy: np.ndarray, *, radius_m: float) -> dict:
    """Score-ordered greedy one-to-one matching within an inclusive XY radius.

    Predictions are already decoded in the annotated domain. Duplicate matches
    are false positives. This is a center diagnostic, not oriented-box AP.
    """

    if predicted.ndim != 2 or predicted.shape[1] != 3 or target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError("expected predictions [N,3] x/y/score and targets [M,2].")
    if not np.isfinite(predicted).all() or not np.isfinite(target_xy).all() or not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("matching requires finite coordinates and a positive radius.")
    available = np.ones(len(target_xy), dtype=bool)
    errors = []
    for index in np.argsort(-predicted[:, 2], kind="stable"):
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        distances = np.linalg.norm(target_xy[candidates] - predicted[index, :2], axis=1)
        best = int(np.argmin(distances))
        if distances[best] <= radius_m:
            available[candidates[best]] = False
            errors.append(float(distances[best]))
    return {"tp": len(errors), "fp": len(predicted) - len(errors), "fn": int(available.sum()),
            "matched_distance_sum_m": sum(errors)}


def _on_device(frame, device):
    return (
        {name: value.to(device) for name, value in frame.inputs.items()},
        {name: value.to(device) for name, value in frame.targets.items()},
    )


def _evaluate(model, frames, config, device):
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for frame in frames:
                inputs, targets = _on_device(frame, device)
                prediction = model(**inputs)
                loss = lidar_center_loss(prediction, **targets)
                decoded = decode_lidar_centers(
                    prediction, grid=config.grid, score_threshold=config.score_threshold,
                    top_k=config.top_k, valid_mask=targets["supervision_mask"],
                )[0].cpu().numpy()
                rows.append({"sample_id": frame.sample_id,
                             **{name: float(getattr(loss, name)) for name in ("total", "heatmap", "offset")},
                             **match_lidar_centers(decoded, frame.centers_xy, radius_m=config.match_radius_m)})
    finally:
        model.train(was_training)
    tp, fp, fn = (sum(row[name] for row in rows) for name in ("tp", "fp", "fn"))
    result = {"train_eval_loss": float(np.mean([r["total"] for r in rows])),
              "train_eval_heatmap": float(np.mean([r["heatmap"] for r in rows])),
              "train_eval_offset": float(np.mean([r["offset"] for r in rows])),
              "tp": tp, "fp": fp, "fn": fn,
              "precision": tp / (tp + fp) if tp + fp else 0.,
              "recall": tp / (tp + fn),
              "matched_mean_distance_m": sum(r["matched_distance_sum_m"] for r in rows) / tp if tp else None,
              "per_frame": rows}
    if not all(math.isfinite(result[name]) for name in ("train_eval_loss", "train_eval_heatmap", "train_eval_offset")):
        raise ValueError("non-finite evaluation loss.")
    return result


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _persist(output, model, optimizer, config, fingerprints, history):
    checkpoint = output / "checkpoints/latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"kind": "lidar_center_overfit_v1", "config": config.serialized(),
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": history[-1]["epoch"], "step": history[-1]["step"],
                "fingerprints": fingerprints, "metrics": history}, temporary)
    temporary.replace(checkpoint)
    # The checkpoint's embedded history is authoritative after an interruption.
    _write_json(output / "metrics.json", history)
    _write_json(output / "run_metadata.json", {
        "task": "lidar_center_overfit", "config": config.serialized(), "batch_size": 1,
        "train_ids": list(config.sample_ids), "evaluation_ids": list(config.sample_ids),
        "evaluation_scope": "Same training frames; memorization only, no held-out validation or 3D AP.",
        "matching": "Score-ordered greedy one-to-one XY matching within match_radius_m; masked domain.",
        "loss_aggregation": "Mean of per-frame center-normalized losses.",
        "torch_version": str(torch.__version__), "epoch": history[-1]["epoch"],
        "step": history[-1]["step"], "fingerprints": fingerprints,
    })
    return checkpoint


def _read_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("kind") != "lidar_center_overfit_v1":
        raise ValueError("not a LiDAR center overfit checkpoint.")
    return payload


def load_lidar_center_checkpoint(path: str | Path):
    """Load a local diagnostic checkpoint for CPU inspection, without CUDA."""

    payload = _read_checkpoint(path)
    config = LidarCenterOverfitConfig.from_mapping(payload["config"])
    with torch.random.fork_rng(devices=[]):
        model = LidarCenterDiagnostic(channels=config.channels).cpu().eval()
    model.load_state_dict(payload["model_state_dict"])
    return model, config, payload["metrics"]


def run_lidar_center_overfit(
    config: LidarCenterOverfitConfig, *, resume_from: str | Path | None = None,
    emit: Callable[[dict], None] | None = None,
) -> dict:
    """Train batch-1 FP32, cache at most four frames on CPU, checkpoint at evaluation.

    Fresh runs refuse nonempty output directories. Resume verifies configuration
    and input hashes. Epoch count, CPU threads and device may change on resume;
    the order for each epoch derives from the seed and epoch, not global RNG.
    Model placement precedes optimizer creation/loading to preserve Adam's device
    policy. CUDA is opt-in, guarded, and has no retry or cleanup calls on failure.
    Use a separate process for GPU runs so process exit releases resources.
    """

    output = Path(config.output_dir).resolve()
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise ValueError("output_dir is not empty; choose a new directory or explicitly resume latest.pt.")
    payload = None
    if resume_from is not None:
        if Path(resume_from).resolve() != output / "checkpoints/latest.pt":
            raise ValueError("resume_from must be this output directory's latest.pt.")
        payload = _read_checkpoint(resume_from)
        mutable = {"epochs", "cpu_threads", "device", "minimum_free_memory_mib"}
        previous = {k: v for k, v in payload["config"].items() if k not in mutable}
        current = {k: v for k, v in config.serialized().items() if k not in mutable}
        if previous != current or config.epochs < payload["epoch"]:
            raise ValueError("resume configuration differs from checkpoint or requests fewer completed epochs.")
    start = perf_counter()

    def report(event, **fields):
        if emit is not None:
            emit({"event": event, "elapsed_s": perf_counter() - start, **fields})

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        frames = [_prepare_frame(config, sample_id) for sample_id in config.sample_ids]
        fingerprints = {f.sample_id: f.fingerprint for f in frames}
        if payload is not None and payload["fingerprints"] != fingerprints:
            raise ValueError("dataset files changed since the checkpoint.")
        if config.device == "cuda":
            from .camera_depth_analysis import require_gpu_ready, run_gpu_preflight

            preflight = run_gpu_preflight(minimum_free_memory_mib=config.minimum_free_memory_mib, run_smoke_test=False)
            report("preflight", report=asdict(preflight))
            require_gpu_ready(preflight)
            torch.cuda.reset_peak_memory_stats()
        device = torch.device(config.device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            model = LidarCenterDiagnostic(channels=config.channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, foreach=False)
        history = [] if payload is None else payload["metrics"]
        step = 0 if payload is None else int(payload["step"])
        start_epoch = 0 if payload is None else int(payload["epoch"])
        if payload is not None:
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            _persist(output, model, optimizer, config, fingerprints, history)
        report("prepared", sample_ids=list(config.sample_ids), device=config.device,
               car_counts=[len(f.centers_xy) for f in frames], start_epoch=start_epoch)
        with torch.backends.cudnn.flags(benchmark=False):
            if not history:
                row = {"epoch": 0, "step": 0, "train_loss": None, **_evaluate(model, frames, config, device)}
                history.append(row)
                _persist(output, model, optimizer, config, fingerprints, history)
                report("evaluation", **row)
            for epoch in range(start_epoch + 1, config.epochs + 1):
                model.train()
                losses = []
                order = np.random.default_rng(np.random.SeedSequence([config.seed, epoch])).permutation(len(frames))
                for index in order:
                    inputs, targets = _on_device(frames[index], device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = lidar_center_loss(model(**inputs), **targets)
                    if not torch.isfinite(loss.total):
                        raise ValueError("non-finite training loss.")
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    losses.append(float(loss.total.detach()))
                report("epoch_completed", epoch=epoch, step=step, train_loss=float(np.mean(losses)))
                if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
                    row = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)),
                           **_evaluate(model, frames, config, device)}
                    history.append(row)
                    _persist(output, model, optimizer, config, fingerprints, history)
                    report("evaluation", **row)
        result = {"device": config.device, "checkpoint": str(output / "checkpoints/latest.pt"),
                  "epochs_completed": config.epochs, "metrics": history,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if config.device == "cuda" else None}
        report("completed", checkpoint=result["checkpoint"], epochs_completed=config.epochs,
               peak_allocated_mib=result["peak_allocated_mib"])
        return result
    except Exception as exc:
        # Never query/synchronize CUDA or retry after a device failure.
        report("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        torch.set_num_threads(previous_threads)
