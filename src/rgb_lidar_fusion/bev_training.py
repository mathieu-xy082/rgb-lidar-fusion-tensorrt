"""Audited, frozen-encoder BEV fusion memorization on at most four train IDs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .bev_batch import prepare_bev_fusion_batch
from .bev_checkpoint import (
    initialize_bev_fusion_encoders, load_bev_fusion_checkpoint, make_bev_fusion_optimizer,
    restore_bev_fusion_training, save_bev_fusion_checkpoint, validate_bev_fusion_split,
)
from .bev_model import BEVFusionConfig, BEVFusionDiagnostic
from .dataset import KittiObjectDepthDataset
from .lidar_center_model import lidar_center_loss
from .lidar_center_training import (
    LidarCenterOverfitConfig, _evaluate, _on_device, _prepare_frame, _write_json,
)
from .project_lidar import load_velodyne_bin


TASK = "bev_fusion_overfit"


@dataclass(frozen=True)
class BEVFusionOverfitConfig:
    data_root: str = "data/kitti/training"
    output_dir: str = "results/training/kitti_bev_fusion_overfit_4"
    split_path: str = "results/training/kitti_lidar_center_512/split.json"
    camera_checkpoint: str = "results/training/kitti_camera_depth_block_dropout_512_provenance/checkpoints/latest.pt"
    lidar_checkpoint: str = "results/training/kitti_lidar_center_512/checkpoints/best.pt"
    model_config: BEVFusionConfig = BEVFusionConfig()
    sample_ids: tuple[str, ...] = ("000006", "000036", "000052", "000060")
    epochs: int = 100
    eval_every: int = 5
    learning_rate: float = .001
    seed: int = 13
    cpu_threads: int = 2
    device: str = "cpu"
    minimum_free_memory_mib: int = 2048
    score_threshold: float = .3
    top_k: int = 50
    match_radius_m: float = 1.
    max_grad_norm: float = 10.

    def __post_init__(self):
        if not isinstance(self.model_config, BEVFusionConfig) or not self.model_config.freeze_encoders:
            raise ValueError("this increment requires a BEVFusionConfig with frozen encoders.")
        if not self.model_config.camera_enabled:
            raise ValueError("the four-frame fusion diagnostic requires the camera branch.")
        for name in ("split_path", "camera_checkpoint", "lidar_checkpoint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path string.")
        # Share validation of schedule, IDs, matching and device with the LiDAR diagnostic.
        self.lidar_config()
        object.__setattr__(self, "sample_ids", tuple(self.sample_ids))

    @property
    def grid(self):
        return self.model_config.grid

    def lidar_config(self):
        names = ("data_root", "output_dir", "sample_ids", "epochs", "eval_every", "learning_rate",
                 "seed", "cpu_threads", "device", "minimum_free_memory_mib", "score_threshold",
                 "top_k", "match_radius_m", "max_grad_norm")
        return LidarCenterOverfitConfig(
            **{name: getattr(self, name) for name in names}, grid=self.grid,
            channels=self.model_config.lidar_channels,
            max_points_per_pillar=self.model_config.max_points_per_pillar,
        )

    def serialized(self):
        result = asdict(self)
        for name in ("data_root", "output_dir", "split_path", "camera_checkpoint", "lidar_checkpoint"):
            result[name] = str(Path(getattr(self, name)).resolve())
        return json.loads(json.dumps(result, allow_nan=False))

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        try:
            if "model_config" in values:
                values["model_config"] = BEVFusionConfig.from_mapping(values["model_config"])
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid BEV overfit configuration: {exc}") from exc


def read_bev_overfit_split(config):
    split = json.loads(Path(config.split_path).read_text())
    split["data_root"] = str(Path(config.data_root).resolve())
    split = validate_bev_fusion_split(split, verify_files=True)
    if not set(config.sample_ids) <= set(split["train_ids"]):
        raise ValueError("all memorization sample_ids must belong to the saved training split, never validation.")
    return split


def prepare_bev_overfit_frames(config, split):
    """Cache four CPU inputs/targets, using full scans and native annotation geometry.

    No depth holdout is applied. Labels only feed targets/evaluation, never inputs.
    All selected files are checked before and after preprocessing. Cached tensors
    represent that verified snapshot, not subsequent changes to files on disk.
    """

    if not set(config.sample_ids) <= set(split["train_ids"]):
        raise ValueError("sample_ids must belong to the training split.")
    model_config = config.model_config
    dataset = KittiObjectDepthDataset(
        config.data_root, image_shape=model_config.image_shape,
        max_depth_m=model_config.max_depth_m, sample_limit=len(split["source_ids"]),
    )
    if [s[0] for s in dataset.samples] != split["source_ids"]:
        raise ValueError("camera dataset population/order differs from the saved split.")
    positions = {sample[0]: index for index, sample in enumerate(dataset.samples)}
    frames = []
    for sample_id in config.sample_ids:
        native = _prepare_frame(replace(config.lidar_config(), seed=model_config.pillar_seed), sample_id)
        expected = split["fingerprints"][sample_id]
        if native.fingerprint != expected:
            raise ValueError(f"dataset files changed before preparing {sample_id}.")
        item = dataset[positions[sample_id]]
        cloud = load_velodyne_bin(item["meta"]["lidar_path"])
        batch = prepare_bev_fusion_batch([item], {sample_id: cloud}, config=model_config)
        for name, suffix in (("image_2", "png"), ("velodyne", "bin"), ("calib", "txt"), ("label_2", "txt")):
            with (Path(config.data_root) / name / f"{sample_id}.{suffix}").open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[name]:
                    raise ValueError(f"dataset files changed after preparing {sample_id}.")
        frames.append(replace(native, inputs=batch.inputs))
    return frames


def _protocol(config):
    mutable = {"epochs", "cpu_threads", "device", "minimum_free_memory_mib"}
    return {k: v for k, v in config.serialized().items() if k not in mutable}


def _persist(output, model, optimizer, config, split, pretraining, history, *, epoch, step, resources):
    metadata = {
        "task": TASK, "runner_config": config.serialized(),
        "train_ids": list(config.sample_ids), "evaluation_ids": list(config.sample_ids),
        "evaluation_scope": "Same training frames; memorization only, no held-out validation or 3D AP.",
        "batch_size": 1, "precision": "float32", "depth_holdout": False,
        "freeze_encoders": True, "torch_version": str(torch.__version__),
        "resources": resources, "epoch": epoch, "step": step,
    }
    save_bev_fusion_checkpoint(
        output / "checkpoints/latest.pt", model=model, optimizer=optimizer,
        split=split, epoch=epoch, step=step, pretraining=pretraining, metrics=history,
        training_metadata=metadata,
    )
    _write_json(output / "metrics.json", history)
    _write_json(output / "run_metadata.json", metadata)
    _write_json(output / "split.json", split)
    _write_json(output / "pretraining.json", pretraining)


def run_bev_fusion_overfit(config: BEVFusionOverfitConfig, *, resume_from=None, emit=None):
    """Audited encoder warm start, fresh fusion/head, batch-1 FP32 and no CUDA retry.

    Resume validates saved runner/model/split before CUDA. It is self-contained:
    source encoder files are not reopened and source optimizer states never load.
    Use the CLI child process for GPU execution and resource release on exit.
    """

    output = Path(config.output_dir).resolve()
    if resume_from is None and output.exists() and any(output.iterdir()):
        raise ValueError("output_dir is not empty; choose a new directory or resume explicitly.")
    start = perf_counter()

    def report(event, **fields):
        if emit is not None:
            emit({"event": event, "elapsed_s": perf_counter() - start, **fields})

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        split = read_bev_overfit_split(config)
        payload = None
        if resume_from is not None:
            if Path(resume_from).resolve() != output / "checkpoints/latest.pt":
                raise ValueError("resume_from must be this output directory's latest.pt.")
            model, payload = load_bev_fusion_checkpoint(resume_from)
            metadata = payload.get("training_metadata", {})
            saved = BEVFusionOverfitConfig.from_mapping(metadata.get("runner_config", {}))
            if (metadata.get("task") != TASK or _protocol(saved) != _protocol(config)
                    or payload["split"] != split or payload["purpose"] != "resume"
                    or "optimizer_state_dict" not in payload or config.epochs < payload["epoch"]
                    or payload["config"] != config.model_config.serialized()):
                raise ValueError("resume task, configuration, split or epoch target differs from checkpoint.")
            pretraining = payload["pretraining"]
        else:
            with torch.random.fork_rng(devices=[]), torch.device("cpu"):
                torch.random.default_generator.manual_seed(config.seed)
                model = BEVFusionDiagnostic(config.model_config)
            pretraining = initialize_bev_fusion_encoders(
                model, camera_checkpoint=config.camera_checkpoint,
                lidar_checkpoint=config.lidar_checkpoint, split=split,
            )
        if pretraining.get("camera", {}).get("provenance") != "recorded_at_training":
            raise ValueError("fusion training requires recorded camera provenance without legacy exceptions.")
        frames = prepare_bev_overfit_frames(config, split)
        report("audited", sample_ids=list(config.sample_ids),
               car_counts=[len(frame.centers_xy) for frame in frames],
               camera_provenance=pretraining["camera"]["provenance"])
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
        epoch = step = 0
        history = []
        if payload is not None:
            restored = restore_bev_fusion_training(resume_from, model=model, optimizer=optimizer, split=split)
            epoch, step, history = restored["epoch"], restored["step"], restored["metrics"]
        trainable = [p for p in model.parameters() if p.requires_grad]

        def resources():
            return {
                "elapsed_s": perf_counter() - start,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None,
                "scope": "Whole runner since start, including audit, CPU preparation and evaluation; not inference latency. GPU peaks are PyTorch allocator only, not total device use.",
            }

        def persist():
            _persist(output, model, optimizer, config, split, pretraining, history,
                     epoch=epoch, step=step, resources=resources())

        report("prepared", device=config.device, start_epoch=epoch, total_epochs=config.epochs,
               steps_per_epoch=len(frames), trainable_parameters=sum(p.numel() for p in trainable))
        with torch.backends.cudnn.flags(benchmark=False):
            if not history:
                row = {"epoch": 0, "step": 0, "train_loss": None, **_evaluate(model, frames, config, device)}
                history.append(row)
                persist()
                report("evaluation", **row)
            elif payload is not None:
                persist()  # Repair interrupted sidecars from the checkpoint's authoritative state.
            for epoch in range(epoch + 1, config.epochs + 1):
                model.train()
                losses = []
                order = np.random.default_rng(np.random.SeedSequence([config.seed, epoch])).permutation(len(frames))
                for batch_index, index in enumerate(order, start=1):
                    inputs, targets = _on_device(frames[index], device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = lidar_center_loss(model(**inputs), **targets)
                    if not torch.isfinite(loss.total):
                        raise ValueError("non-finite training loss.")
                    loss.total.backward()
                    norm = torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    losses.append(float(loss.total.detach()))
                    report("step_completed", epoch=epoch, batch=batch_index, batches=len(frames),
                           sample_id=frames[index].sample_id, step=step, loss=losses[-1], grad_norm=float(norm))
                report("epoch_completed", epoch=epoch, step=step, train_loss=float(np.mean(losses)))
                if epoch == 1 or epoch % config.eval_every == 0 or epoch == config.epochs:
                    row = {"epoch": epoch, "step": step, "train_loss": float(np.mean(losses)),
                           **_evaluate(model, frames, config, device)}
                    history.append(row)
                    persist()
                    report("evaluation", **row)
        result = {"epochs_completed": config.epochs, "checkpoint": str(output / "checkpoints/latest.pt"),
                  "metrics": history, "resources": resources()}
        report("completed", checkpoint=result["checkpoint"], epochs_completed=config.epochs,
               resources=result["resources"])
        return result
    except Exception as exc:
        # Do not query/synchronize CUDA or retry after a device failure.
        report("failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        torch.set_num_threads(previous_threads)
