"""CPU-first pretrained-weight audits and distinct multimodal checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .bev_model import BEVFusionConfig, BEVFusionDiagnostic
from .camera_bev import BEVGridConfig
from .camera_depth_manifest import CAMERA_DEPTH_CHECKPOINT_KIND, validate_camera_depth_manifest
from .training import split_dataset_indices


CHECKPOINT_KIND = "bev_fusion_center_v1"
_FILES = {"image_2": "png", "calib": "txt", "velodyne": "bin", "label_2": "txt"}


def _ids(values, name):
    if not isinstance(values, (list, tuple)) or not values or any(
        not isinstance(s, str) or len(s) != 6 or not s.isascii() or not s.isdigit() for s in values
    ) or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain non-empty unique six-digit IDs.")
    return list(values)


def validate_bev_fusion_split(split: dict, *, verify_files: bool = False) -> dict:
    """Validate saved membership/hashes; optionally verify every local input file."""

    if not isinstance(split, dict) or not isinstance(split.get("data_root"), str) or not split["data_root"].strip():
        raise ValueError("split must declare a non-empty data_root path.")
    source = _ids(split.get("source_ids"), "source_ids")
    train = _ids(split.get("train_ids"), "train_ids")
    validation = _ids(split.get("validation_ids"), "validation_ids")
    if set(train) & set(validation) or set(train) | set(validation) != set(source):
        raise ValueError("train/validation must be disjoint and partition source_ids.")
    fingerprints = split.get("fingerprints", {})
    if set(fingerprints) != set(source):
        raise ValueError("split fingerprints must cover all source_ids.")
    for sample_id in source:
        hashes = fingerprints[sample_id]
        if set(hashes) != set(_FILES) or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in hashes.values()
        ):
            raise ValueError("each fingerprint must contain four SHA-256 hashes.")
        if verify_files:
            root = Path(split["data_root"])
            for name, suffix in _FILES.items():
                path = root / name / f"{sample_id}.{suffix}"
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != hashes[name]:
                    raise ValueError(f"dataset file changed: {path}.")
    return json.loads(json.dumps(split))


def _read(path):
    # Hash and deserialize the same open file, even if latest.pt is atomically replaced.
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("checkpoint must contain model_state_dict.")
    return payload, digest


def _validate_state(module, state):
    expected = module.state_dict()
    if set(expected) != set(state):
        raise ValueError("checkpoint state keys differ from the expected module.")
    for name, target in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"checkpoint tensor shape/dtype differs: {name}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"checkpoint tensor must be finite: {name}.")


def initialize_bev_fusion_encoders(
    model: BEVFusionDiagnostic, *, camera_checkpoint: str | Path,
    lidar_checkpoint: str | Path, split: dict,
    camera_source_ids: list[str] | tuple[str, ...] | None = None,
    allow_legacy_camera: bool = False,
) -> dict:
    """Warm-start encoders on CPU, never source heads or optimizer states.

    Legacy camera checkpoints omit source IDs/file hashes. Their ID mapping
    requires allow_legacy_camera=True and an explicit original sorted ID list.
    New checkpoints use recorded IDs and hashes, without a caller-declared map.
    All checks complete before any destination module is changed.
    """

    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("initialize encoders on an FP32 CPU model before device placement/optimizer creation.")
    split = validate_bev_fusion_split(split, verify_files=True)
    camera, camera_sha = _read(camera_checkpoint)
    lidar, lidar_sha = _read(lidar_checkpoint)
    config = model.config
    expected = {"dataset": "kitti_camera_depth", "lidar_mode": "enriched",
                "height": config.image_shape[0], "width": config.image_shape[1],
                "max_depth_m": config.max_depth_m, "splat_radius_px": config.splat_radius_px,
                "splat_sigma_px": config.splat_sigma_px}
    if camera.get("kind") == CAMERA_DEPTH_CHECKPOINT_KIND:
        recorded = validate_camera_depth_manifest(camera.get("data_manifest"))
        source = recorded["source_ids"]
        if source != split["source_ids"] or any(
            digest != split["fingerprints"][sample_id][name]
            for sample_id, row in recorded["fingerprints"].items() for name, digest in row.items()
        ):
            raise ValueError("camera checkpoint source IDs or file hashes differ from the fusion dataset.")
        if camera_source_ids is not None and list(camera_source_ids) != source:
            raise ValueError("camera_source_ids conflict with the recorded camera manifest.")
        actual = recorded["protocol"]
        camera_train, camera_validation = recorded["train_ids"], recorded["validation_ids"]
        id_mapping = "recorded camera-depth training manifest"
        provenance = "recorded_at_training"
        limitation = "Recorded frame membership/content, not independent scenes or an unbiased test set."
    else:
        if camera.get("kind") is not None or "data_manifest" in camera:
            raise ValueError("unsupported camera checkpoint kind/manifest combination.")
        if allow_legacy_camera is not True:
            raise ValueError("legacy camera provenance is unverified; explicit allow_legacy_camera=True "
                             "and camera_source_ids are required for this exception.")
        actual = camera.get("config", {})
        source = _ids(camera_source_ids, "camera_source_ids")
        if source != split["source_ids"] or actual.get("sample_count") != len(source) or source != sorted(source):
            raise ValueError("camera_source_ids must match the saved sorted source_ids and sample_count.")
        train_indices, validation_indices = split_dataset_indices(
            len(source), validation_fraction=actual.get("validation_fraction", .2),
            seed=actual.get("split_seed", actual.get("seed", 0)),
        )
        camera_train = [source[i] for i in train_indices]
        camera_validation = [source[i] for i in validation_indices]
        id_mapping = "caller-declared original sorted source IDs"
        provenance = "legacy_unverified"
        limitation = "Legacy camera checkpoint has no saved source IDs/file hashes; " \
                     "historical membership/content cannot be independently proven."
    if any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("camera checkpoint preprocessing differs from the fusion configuration.")
    if set(camera_train) & set(split["validation_ids"]):
        raise ValueError("camera pretraining includes fusion validation IDs.")
    if lidar.get("kind") != "lidar_center_trainval_v1":
        raise ValueError("LiDAR warm start requires a train/validation checkpoint, not four-frame overfit weights.")
    lidar_split = validate_bev_fusion_split(lidar.get("split", {}))
    if set(lidar_split["train_ids"]) & set(split["validation_ids"]):
        raise ValueError("LiDAR pretraining includes fusion validation IDs.")
    if lidar_split["source_ids"] != split["source_ids"] or lidar_split["fingerprints"] != split["fingerprints"]:
        raise ValueError("LiDAR checkpoint source IDs or file hashes differ from the fusion dataset.")
    lidar_config = lidar.get("config", {})
    if not isinstance(lidar_config.get("grid"), dict):
        raise ValueError("LiDAR checkpoint must declare its grid.")
    if (
        BEVGridConfig(**lidar_config["grid"]).shape != config.grid.shape
        or json.loads(json.dumps(lidar_config["grid"])) != config.serialized()["grid"]
        or lidar_config.get("channels") != config.lidar_channels
        or lidar_config.get("max_points_per_pillar") != config.max_points_per_pillar
        or lidar_config.get("seed") != config.pillar_seed
    ):
        raise ValueError("LiDAR checkpoint preprocessing/architecture differs from the fusion configuration.")
    camera_state = camera["model_state_dict"]
    pfn_state = {k.removeprefix("pfn."): v for k, v in lidar["model_state_dict"].items() if k.startswith("pfn.")}
    context_state = {k.removeprefix("context."): v for k, v in lidar["model_state_dict"].items()
                     if k.startswith("context.")}
    for module, state in ((model.camera, camera_state), (model.pfn, pfn_state), (model.context, context_state)):
        _validate_state(module, state)
    model.camera.load_state_dict(camera_state, strict=True)
    model.pfn.load_state_dict(pfn_state, strict=True)
    model.context.load_state_dict(context_state, strict=True)
    return {
        "camera": {"path": str(Path(camera_checkpoint).resolve()), "sha256": camera_sha,
                   "epoch": camera.get("epoch"), "step": camera.get("step"),
                   "train_ids": camera_train, "validation_ids": camera_validation,
                   "source_ids": source, "id_mapping": id_mapping,
                   "provenance": provenance, "limitation": limitation},
        "lidar": {"path": str(Path(lidar_checkpoint).resolve()), "sha256": lidar_sha,
                  "epoch": lidar.get("epoch"), "step": lidar.get("step"),
                  "train_ids": lidar_split["train_ids"], "validation_ids": lidar_split["validation_ids"]},
        "loaded": ["camera", "pfn", "context"], "source_head_and_optimizer_loaded": False,
    }


def make_bev_fusion_optimizer(model: BEVFusionDiagnostic, *, learning_rate: float = .001):
    """Create Adam after model placement; frozen encoders/depth head stay excluded."""

    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=learning_rate, foreach=False)


def _check_optimizer(model, optimizer):
    if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1:
        raise ValueError("expected the single-group Adam optimizer for this model.")
    group = optimizer.param_groups[0]
    if group["foreach"] is not False or [id(p) for p in group["params"]] != [
        id(p) for p in model.parameters() if p.requires_grad
    ]:
        raise ValueError("optimizer must contain exactly the ordered trainable model parameters, foreach=False.")


def save_bev_fusion_checkpoint(
    path: str | Path, *, model: BEVFusionDiagnostic, split: dict, epoch: int, step: int,
    optimizer=None, pretraining: dict | None = None, metrics: list[dict] | None = None,
    training_metadata: dict | None = None,
    best: dict | None = None,
) -> None:
    """Atomic new-kind checkpoint; optimizer present means resumable, otherwise inference."""

    for value in (epoch, step):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("epoch and step must be non-negative integers.")
    split = validate_bev_fusion_split(split)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("multimodal checkpoints currently require FP32 model parameters.")
    _validate_state(model, model.state_dict())
    if optimizer is not None:
        _check_optimizer(model, optimizer)
    payload = {
        "kind": CHECKPOINT_KIND, "purpose": "resume" if optimizer is not None else "inference",
        "config": model.config.serialized(), "split": split, "epoch": epoch, "step": step,
        "pretraining": json.loads(json.dumps(pretraining)), "metrics": metrics or [],
        "model_state_dict": model.state_dict(),
        "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if training_metadata is not None:
        payload["training_metadata"] = json.loads(json.dumps(training_metadata, allow_nan=False))
    if best is not None:
        _validate_best_state(model, best, epoch=epoch, step=step)
        payload["best"] = best
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_bev_fusion_checkpoint(path: str | Path):
    """Load a self-contained multimodal checkpoint on CPU in eval mode, without old files."""

    payload, _ = _read(path)
    if payload.get("kind") != CHECKPOINT_KIND or payload.get("purpose") not in ("resume", "inference"):
        raise ValueError("not a multimodal BEV center checkpoint.")
    if any(isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int) or payload[key] < 0
           for key in ("epoch", "step")):
        raise ValueError("checkpoint epoch/step must be non-negative integers.")
    validate_bev_fusion_split(payload["split"])
    config = BEVFusionConfig.from_mapping(payload["config"])
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = BEVFusionDiagnostic(config).float().eval()
    _validate_state(model, payload["model_state_dict"])
    if payload.get("trainable_names") != [n for n, p in model.named_parameters() if p.requires_grad]:
        raise ValueError("checkpoint trainable parameter policy differs from its configuration.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if "best" in payload:
        _validate_best_state(model, payload["best"], epoch=payload["epoch"], step=payload["step"])
    return model, payload


def _validate_best_state(model, best, *, epoch, step):
    if not isinstance(best, dict) or any(
        isinstance(best.get(key), bool) or not isinstance(best.get(key), int)
        or not 0 <= best[key] <= upper for key, upper in (("epoch", epoch), ("step", step))
    ):
        raise ValueError("invalid best checkpoint counters.")
    value = best.get("val_loss")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("best checkpoint val_loss must be finite.")
    _validate_state(model, best.get("model_state_dict", {}))


def restore_bev_fusion_training(path: str | Path, *, model: BEVFusionDiagnostic, optimizer, split: dict) -> dict:
    """Resume only the new format with identical model config, split and data.

    Caller must place the destination model before creating its optimizer.
    Old encoder checkpoints are warm starts only; inference/best checkpoints
    cannot resume Adam. No epochs/optimizer step or device allocation is run here.
    """

    _, payload = load_bev_fusion_checkpoint(path)
    _check_optimizer(model, optimizer)
    if payload["purpose"] != "resume" or "optimizer_state_dict" not in payload:
        raise ValueError("inference checkpoint has no resumable optimizer state.")
    if model.config.serialized() != payload["config"]:
        raise ValueError("resume model configuration changed.")
    if validate_bev_fusion_split(split, verify_files=True) != payload["split"]:
        raise ValueError("resume split or dataset fingerprints changed.")
    _validate_state(model, payload["model_state_dict"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {key: payload[key] for key in ("epoch", "step", "pretraining", "metrics")}
