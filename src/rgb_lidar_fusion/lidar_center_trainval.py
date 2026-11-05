"""Streaming LiDAR center training with a fixed, disjoint KITTI frame split."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .lidar_center_model import LidarCenterDiagnostic, lidar_center_loss
from .lidar_center_training import (
    LidarCenterTrainConfig, _evaluate, _on_device, _prepare_frame, _read_checkpoint, _write_json,
)
from .training import split_dataset_indices


def build_lidar_center_split(config: LidarCenterTrainConfig, *, progress=None) -> dict:
    """Select the first N sorted local KITTI IDs and fingerprint all counterparts.

    The seeded random frame split is reproducible, not guaranteed sequence- or
    scene-disjoint. Missing labels are errors, never implicit background.
    Only identifiers/hashes are retained, not padded pillars for all N frames.
    """

    root = Path(config.data_root).resolve()
    ids = sorted(p.stem for p in (root / "image_2").glob("*.png")
                 if len(p.stem) == 6 and p.stem.isascii() and p.stem.isdigit())
    if len(ids) < config.sample_count:
        raise ValueError(f"need {config.sample_count} KITTI images under {root}; found {len(ids)}.")
    ids = ids[:config.sample_count]
    train, validation = split_dataset_indices(
        len(ids), validation_fraction=config.validation_fraction, seed=config.split_seed,
    )
    fingerprints = {}
    for processed, sample_id in enumerate(ids, start=1):
        fingerprints[sample_id] = {}
        for name, suffix in (("image_2", "png"), ("calib", "txt"), ("velodyne", "bin"), ("label_2", "txt")):
            with (root / name / f"{sample_id}.{suffix}").open("rb") as stream:
                fingerprints[sample_id][name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if progress is not None:
            progress(processed, sample_id)
    return {
        "policy": "seeded_random_frame_split", "split_seed": config.split_seed,
        "validation_fraction": config.validation_fraction, "data_root": str(root),
        "source_ids": ids, "train_ids": [ids[i] for i in train],
        "validation_ids": [ids[i] for i in validation], "fingerprints": fingerprints,
        "limitation": "Disjoint frame IDs, not a guarantee of independent sequences/scenes.",
    }


def _frames(config, ids, fingerprints):
    for sample_id in ids:
        frame = _prepare_frame(config, sample_id, require_positive=False)
        if frame.fingerprint != fingerprints[sample_id]:
            raise ValueError(f"dataset files changed during the run: {sample_id}.")
        yield frame


def _evaluate_split(model, config, split, name, device, progress=None):
    ids = split["train_ids" if name == "train" else "validation_ids"]
    prefix = "train_eval" if name == "train" else "val"
    values = _evaluate(
        model, _frames(config, ids, split["fingerprints"]), config, device,
        prefix=prefix, progress=progress,
    )
    return {key if key.startswith(prefix + "_") else f"{name}_{key}": value
            for key, value in values.items()}


def _atomic_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _persist(output, config, split, model, optimizer, history, best):
    row = history[-1]
    payload = {
        "kind": "lidar_center_trainval_v1", "purpose": "resume", "config": config.serialized(),
        "split": split, "epoch": row["epoch"], "step": row["step"], "metrics": history,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best": best,
    }
    _atomic_checkpoint(output / "checkpoints/latest.pt", payload)
    # The latest checkpoint carries the best weights so interrupted sidecars can be repaired.
    _atomic_checkpoint(output / "checkpoints/best.pt", {
        "kind": payload["kind"], "purpose": "inference", "config": payload["config"],
        "split": split, "epoch": best["epoch"], "step": best["step"],
        "metrics": [r for r in history if r["epoch"] <= best["epoch"]],
        "model_state_dict": best["model_state_dict"],
    })
    _write_json(output / "metrics.json", history)
    _write_json(output / "split.json", split)
    _write_json(output / "run_metadata.json", {
        "task": "lidar_center_trainval", "config": config.serialized(), "batch_size": 1,
        "initialized_from": "fresh_random_weights; no four-frame overfit warm start",
        "train_ids": split["train_ids"], "validation_ids": split["validation_ids"],
        "split_limitation": split["limitation"], "epoch": row["epoch"], "step": row["step"],
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "selection_metric": "minimum val_loss, including the epoch-0 baseline",
        "loss_aggregation": "Mean per supervised frame; center-normalized focal/L1, including negative frames.",
        "matching": "Masked-domain, score-ordered greedy one-to-one XY matching; not 3D AP.",
        "torch_version": str(torch.__version__),
    })


def run_lidar_center_training(
    config: LidarCenterTrainConfig, *, resume_from: str | Path | None = None,
    emit: Callable[[dict], None] | None = None,
) -> dict:
    """Train only train_ids; evaluate both fixed splits in eval/no-grad mode.

    Frames are prepared on demand, never cached across the 512-frame dataset.
    Valid no-Car frames are negatives; wholly unknown supervision is skipped and
    reported. A split with no usable supervision raises. Checkpoints are written
    after every epoch; resume checks config, exact split and all file hashes.
    CUDA is explicit and guarded, with no fallback/retry/cleanup after failure.
    """

    output = Path(config.output_dir).resolve()
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise ValueError("output_dir is not empty; choose a new directory or explicitly resume latest.pt.")
    payload = None
    if resume_from is not None:
        if Path(resume_from).resolve() != output / "checkpoints/latest.pt":
            raise ValueError("resume_from must be this run's latest.pt, not best.pt or an overfit checkpoint.")
        payload = _read_checkpoint(resume_from, expected_kind="lidar_center_trainval_v1")
        mutable = {"epochs", "device", "cpu_threads", "minimum_free_memory_mib", "log_every_steps"}
        if {k: v for k, v in payload["config"].items() if k not in mutable} != {
            k: v for k, v in config.serialized().items() if k not in mutable
        } or config.epochs < payload["epoch"]:
            raise ValueError("resume configuration differs from checkpoint or requests fewer completed epochs.")
    started = perf_counter()

    def report(event, **fields):
        if emit is not None:
            emit({"event": event, "elapsed_s": perf_counter() - started, **fields})

    def progress(phase, total):
        def update(processed, sample_id):
            if processed == 1 or processed % config.log_every_steps == 0 or processed == total:
                report("progress", phase=phase, processed=processed, total=total, sample_id=sample_id)
        return update

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        report("phase_started", phase="dataset_split_and_hashes")
        split = build_lidar_center_split(config, progress=progress("dataset_checks", config.sample_count))
        if payload is not None and payload["split"] != split:
            raise ValueError("dataset files or train/validation split changed since checkpoint.")
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
        best = None if payload is None else payload["best"]
        report("prepared", device=config.device, train_count=len(split["train_ids"]),
               validation_count=len(split["validation_ids"]), start_epoch=start_epoch)

        def evaluate(epoch, train_loss, skipped_ids, negative_frames):
            nonlocal best
            row = {"epoch": epoch, "step": step, "train_loss": train_loss,
                   "train_skipped_update_ids": skipped_ids, "train_negative_update_frames": negative_frames}
            for name, ids in (("train", split["train_ids"]), ("val", split["validation_ids"])):
                report("phase_started", phase=f"{name}_evaluation", epoch=epoch)
                row.update(_evaluate_split(
                    model, config, split, name, device, progress(f"{name}_evaluation", len(ids)),
                ))
            history.append(row)
            if best is None or row["val_loss"] < best["val_loss"]:
                best = {"epoch": epoch, "step": step, "val_loss": row["val_loss"],
                        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            _persist(output, config, split, model, optimizer, history, best)
            report("evaluation", **row, best_epoch=best["epoch"], best_val_loss=best["val_loss"])

        with torch.backends.cudnn.flags(benchmark=False):
            if not history:
                evaluate(0, None, [], 0)
            else:
                model.load_state_dict(payload["model_state_dict"])
                optimizer.load_state_dict(payload["optimizer_state_dict"])
                _persist(output, config, split, model, optimizer, history, best)
            for epoch in range(start_epoch + 1, config.epochs + 1):
                model.train()
                losses, skipped_ids = [], []
                negative_frames = 0
                order = np.random.default_rng(np.random.SeedSequence([config.seed, epoch])).permutation(len(split["train_ids"]))
                ordered_ids = [split["train_ids"][i] for i in order]
                report("phase_started", phase="training", epoch=epoch)
                for position, frame in enumerate(_frames(config, ordered_ids, split["fingerprints"]), start=1):
                    if not frame.targets["supervision_mask"].any():
                        skipped_ids.append(frame.sample_id)
                        continue
                    negative_frames += not len(frame.centers_xy)
                    inputs, targets = _on_device(frame, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = lidar_center_loss(model(**inputs), **targets)
                    if not torch.isfinite(loss.total):
                        raise ValueError("non-finite training loss.")
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    losses.append(float(loss.total.detach()))
                    if position == 1 or position % config.log_every_steps == 0 or position == len(ordered_ids):
                        report("step_completed", epoch=epoch, step=step, processed=position, total=len(ordered_ids),
                               sample_id=frame.sample_id, loss=losses[-1])
                if not losses:
                    raise ValueError("training split has no supervised frames; no optimizer steps performed.")
                evaluate(epoch, float(np.mean(losses)), skipped_ids, negative_frames)
        result = {"device": config.device, "epochs_completed": config.epochs,
                  "checkpoint": str(output / "checkpoints/latest.pt"),
                  "best_checkpoint": str(output / "checkpoints/best.pt"), "best_epoch": best["epoch"],
                  "split": split, "metrics": history,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if config.device == "cuda" else None}
        report("completed", checkpoint=result["checkpoint"], best_checkpoint=result["best_checkpoint"],
               epochs_completed=config.epochs, best_epoch=best["epoch"], peak_allocated_mib=result["peak_allocated_mib"])
        return result
    except Exception as exc:
        report("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        torch.set_num_threads(previous_threads)
