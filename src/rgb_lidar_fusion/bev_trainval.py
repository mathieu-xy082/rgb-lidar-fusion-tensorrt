"""One matched BEV pilot variant on the saved, disjoint train/validation split."""

from dataclasses import asdict
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .bev_checkpoint import (
    load_bev_fusion_checkpoint, make_bev_fusion_optimizer, restore_bev_fusion_training,
    save_bev_fusion_checkpoint,
)
from .bev_data import BEVFusionFrameSource
from .bev_model import BEVFusionDiagnostic
from .bev_pilot import BEVFusionPilotConfig, prepare_bev_fusion_pilot, read_bev_pilot_split
from .lidar_center_model import lidar_center_loss
from .lidar_center_training import _evaluate, _on_device, _write_json


TASK = "bev_fusion_pilot"


def _protocol(config):
    mutable = {"epochs", "device", "cpu_threads", "minimum_free_memory_mib", "log_every_steps"}
    return {key: value for key, value in config.serialized().items() if key not in mutable}


def _read_resume(path, *, config, variant, split):
    model, payload = load_bev_fusion_checkpoint(path)
    metadata = payload.get("training_metadata", {})
    if metadata.get("task") != TASK or metadata.get("variant") != variant:
        raise ValueError("resume task/variant differs; overfit or other-variant checkpoints are not pilot resumes.")
    if not isinstance(metadata.get("runner_config"), dict):
        raise ValueError("pilot resume requires its saved runner configuration.")
    previous = BEVFusionPilotConfig.from_mapping(metadata["runner_config"])
    if (payload["purpose"] != "resume" or "optimizer_state_dict" not in payload
            or payload["config"] != config.model_for_variant(variant).serialized()
            or _protocol(previous) != _protocol(config) or payload["split"] != split
            or config.epochs < payload["epoch"]):
        raise ValueError("resume configuration, dataset/split, purpose or epoch target differs from checkpoint.")
    history = payload["metrics"]
    if (not isinstance(history, list) or len(history) != payload["epoch"] + 1
            or any(row.get("epoch") != index for index, row in enumerate(history))
            or history[-1].get("step") != payload["step"]):
        raise ValueError("pilot checkpoint history does not match its counters.")
    if any(not isinstance(row.get("val_loss"), (int, float)) or not math.isfinite(row["val_loss"])
           for row in history):
        raise ValueError("pilot history requires finite validation losses.")
    selected = min(history, key=lambda row: row["val_loss"])
    best = payload.get("best", {})
    if any(best.get(key) != selected[key] for key in ("epoch", "step", "val_loss")):
        raise ValueError("best state does not match the minimum validation-loss history.")
    pretraining = payload.get("pretraining", {})
    if pretraining.get("camera", {}).get("provenance") != "recorded_at_training":
        raise ValueError("pilot resume requires recorded camera provenance.")
    for name in ("camera", "lidar"):
        for partition in ("train", "validation"):
            if pretraining.get(name, {}).get(partition + "_ids") != split[partition + "_ids"]:
                raise ValueError("pilot source membership differs from canonical split.")
    initialization = metadata.get("initialization", {})
    if (initialization.get("camera_enabled") != (variant == "fusion")
            or initialization.get("seed") != config.seed
            or not isinstance(initialization.get("new_blocks_sha256"), str)
            or len(initialization["new_blocks_sha256"]) != 64):
        raise ValueError("pilot initialization record is missing or incompatible.")
    return model, payload


def _persist(output, *, model, optimizer, config, variant, split, pretraining,
             initialization, history, best, resources):
    row = history[-1]
    metadata = {
        "task": TASK, "variant": variant, "runner_config": config.serialized(),
        "initialization": initialization, "batch_size": 1, "precision": "float32",
        "freeze_encoders": True, "depth_holdout": False,
        "train_ids": split["train_ids"], "validation_ids": split["validation_ids"],
        "evaluation_scope": "Saved disjoint frames, not independent scenes, an unbiased test set or full 3D AP.",
        "loss_aggregation": "Mean per supervised frame; includes valid no-Car negatives; unknown skipped.",
        "matching": "Masked-domain score-ordered greedy one-to-one XY, inclusive match_radius_m.",
        "selection_metric": "minimum val_loss, including epoch 0; earliest epoch breaks ties",
        "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
        "epoch": row["epoch"], "step": row["step"], "resources": resources,
        "torch_version": str(torch.__version__),
    }
    save_bev_fusion_checkpoint(
        output / "checkpoints/latest.pt", model=model, optimizer=optimizer,
        split=split, epoch=row["epoch"], step=row["step"], pretraining=pretraining,
        metrics=history, training_metadata=metadata, best=best,
    )
    # Preserve an authoritative CPU best-state copy in latest.pt so sidecars can
    # be repaired after interruption, without changing the live GPU model.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        best_model = BEVFusionDiagnostic(model.config).float().eval()
    best_model.load_state_dict(best["model_state_dict"], strict=True)
    save_bev_fusion_checkpoint(
        output / "checkpoints/best.pt", model=best_model, split=split,
        epoch=best["epoch"], step=best["step"], pretraining=pretraining,
        metrics=[r for r in history if r["epoch"] <= best["epoch"]],
        training_metadata={**metadata, "epoch": best["epoch"], "step": best["step"]},
    )
    _write_json(output / "metrics.json", history)
    _write_json(output / "run_metadata.json", metadata)
    _write_json(output / "split.json", split)
    _write_json(output / "pretraining.json", pretraining)


def run_bev_fusion_pilot(config: BEVFusionPilotConfig, *, variant: str, resume_from=None, emit=None):
    """Stream one batch-1 FP32 variant; only train IDs ever enter Adam updates.

    Fresh models come exclusively from the paired initialization API. Resume
    checks task/variant/protocol, current data, history and best state on CPU
    before preflight. Original encoder files are not reopened during resume.
    Device placement precedes Adam creation/loading. CUDA is opt-in, guarded,
    and has no fallback/retry/cleanup on failure; use the CLI in a child process.
    """

    output = config.output_for_variant(variant)
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise ValueError("output directory is not empty; choose a new pilot root or resume explicitly.")
    if resume_from is not None and Path(resume_from).resolve() != output / "checkpoints/latest.pt":
        raise ValueError("resume_from must be this variant's latest.pt, not best.pt or overfit weights.")
    started = perf_counter()

    def report(event, **fields):
        if emit is not None:
            emit({"event": event, "variant": variant, "elapsed_s": perf_counter() - started, **fields})

    def progress(phase, total, epoch):
        def update(processed, sample_id):
            if processed == 1 or processed % config.log_every_steps == 0 or processed == total:
                report("progress", phase=phase, epoch=epoch, processed=processed,
                       total=total, sample_id=sample_id)
        return update

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        report("phase_started", phase="data_and_source_audit")
        payload = None
        if resume_from is None:
            start = prepare_bev_fusion_pilot(config, variant=variant)
            model, split = start.model, start.split
            pretraining, initialization = start.pretraining, start.initialization
        else:
            split = read_bev_pilot_split(config)
            model, payload = _read_resume(resume_from, config=config, variant=variant, split=split)
            pretraining, initialization = payload["pretraining"], payload["training_metadata"]["initialization"]
        source = BEVFusionFrameSource(config.data_root, model_config=model.config, split=split)
        if config.device == "cuda":
            from .camera_depth_analysis import require_gpu_ready, run_gpu_preflight

            preflight = run_gpu_preflight(minimum_free_memory_mib=config.minimum_free_memory_mib,
                                          run_smoke_test=False)
            report("preflight", report=asdict(preflight))
            require_gpu_ready(preflight)
            torch.cuda.reset_peak_memory_stats()
        device = torch.device(config.device)
        model.to(device)
        optimizer = make_bev_fusion_optimizer(model, learning_rate=config.learning_rate)
        history, best = [], None
        step = start_epoch = 0
        if payload is not None:
            restored = restore_bev_fusion_training(resume_from, model=model, optimizer=optimizer, split=split)
            start_epoch, step, history = restored["epoch"], restored["step"], restored["metrics"]
            best = payload["best"]
        trainable = [p for p in model.parameters() if p.requires_grad]
        center_config = config.center_config()

        def resources():
            return {
                "elapsed_s": perf_counter() - started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None,
                "scope": "Current runner invocation including audit, streaming preprocessing, training and evaluation; not inference latency. GPU peaks are PyTorch allocator only, not whole-device use.",
            }

        def persist():
            _persist(output, model=model, optimizer=optimizer, config=config, variant=variant,
                     split=split, pretraining=pretraining, initialization=initialization,
                     history=history, best=best, resources=resources())

        def evaluate(epoch, train_loss, skipped_ids, negative_frames):
            nonlocal best
            row = {"epoch": epoch, "step": step, "train_loss": train_loss,
                   "train_skipped_update_ids": skipped_ids, "train_negative_update_frames": negative_frames}
            for name, partition in (("train", "train"), ("val", "validation")):
                ids = split[partition + "_ids"]
                prefix = "train_eval" if name == "train" else "val"
                report("phase_started", phase=name + "_evaluation", epoch=epoch)
                values = _evaluate(model, source.frames(ids), center_config, device, prefix=prefix,
                                   progress=progress(name + "_evaluation", len(ids), epoch))
                row.update({key if key.startswith(prefix + "_") else name + "_" + key: value
                            for key, value in values.items()})
            history.append(row)
            if best is None or row["val_loss"] < best["val_loss"]:
                best = {"epoch": epoch, "step": step, "val_loss": row["val_loss"],
                        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            persist()
            report("evaluation", **row, best_epoch=best["epoch"], best_val_loss=best["val_loss"])

        report("prepared", device=config.device, start_epoch=start_epoch, total_epochs=config.epochs,
               train_count=len(split["train_ids"]), validation_count=len(split["validation_ids"]),
               initialization=initialization)
        with torch.backends.cudnn.flags(benchmark=False):
            if not history:
                evaluate(0, None, [], 0)
            else:
                persist()
            for epoch in range(start_epoch + 1, config.epochs + 1):
                model.train()
                losses, skipped_ids = [], []
                negative_frames = 0
                order = np.random.default_rng(np.random.SeedSequence([config.seed, epoch])).permutation(len(split["train_ids"]))
                ordered_ids = [split["train_ids"][i] for i in order]
                report("phase_started", phase="training", epoch=epoch)
                update = progress("training", len(ordered_ids), epoch)
                for position, frame in enumerate(source.frames(ordered_ids), start=1):
                    update(position, frame.sample_id)
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
                    norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    losses.append(float(loss.total.detach()))
                    if position == 1 or position % config.log_every_steps == 0 or position == len(ordered_ids):
                        report("step_completed", epoch=epoch, step=step, processed=position, total=len(ordered_ids),
                               sample_id=frame.sample_id, loss=losses[-1], grad_norm=float(norm))
                if not losses:
                    raise ValueError("training split has no supervised frames; no optimizer steps performed.")
                report("epoch_completed", epoch=epoch, step=step, train_loss=float(np.mean(losses)))
                evaluate(epoch, float(np.mean(losses)), skipped_ids, negative_frames)
        result = {"epochs_completed": config.epochs, "checkpoint": str(output / "checkpoints/latest.pt"),
                  "best_checkpoint": str(output / "checkpoints/best.pt"), "best_epoch": best["epoch"],
                  "metrics": history, "resources": resources(), "variant": variant}
        report("completed", checkpoint=result["checkpoint"], best_checkpoint=result["best_checkpoint"],
               epochs_completed=config.epochs, best_epoch=best["epoch"], resources=result["resources"])
        return result
    except Exception as exc:
        # Never synchronize/query CUDA or retry after a device failure.
        report("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        torch.set_num_threads(previous_threads)
