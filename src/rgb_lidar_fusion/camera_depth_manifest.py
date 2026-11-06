"""Recorded camera-depth data membership, content hashes and training protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CAMERA_DEPTH_CHECKPOINT_KIND = "camera_depth_v2"
MANIFEST_SCHEMA = "camera-depth-data-v1"
CAMERA_FILES = ("image_2", "velodyne", "calib")


def validate_camera_depth_manifest(manifest: dict) -> dict:
    """Check IDs, complete disjoint index partitions and three-file SHA-256 rows."""

    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("invalid camera-depth data manifest schema.")
    ids = manifest.get("source_ids")
    if not isinstance(ids, list) or len(ids) < 2 or any(
        not isinstance(s, str) or not s for s in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError("manifest source_ids must be non-empty unique sample IDs.")
    partitions = []
    for name in ("train", "validation"):
        indices = manifest.get(f"{name}_indices")
        if not isinstance(indices, list) or not indices or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(ids) for i in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError("manifest indices must be non-empty unique in-range integers.")
        if manifest.get(f"{name}_ids") != [ids[i] for i in indices]:
            raise ValueError("manifest sample IDs do not match their saved indices.")
        partitions.append(set(indices))
    if partitions[0] & partitions[1] or partitions[0] | partitions[1] != set(range(len(ids))):
        raise ValueError("manifest train/validation must form a complete disjoint partition.")
    fingerprints = manifest.get("fingerprints", {})
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(ids):
        raise ValueError("manifest fingerprints must cover all sample IDs.")
    for row in fingerprints.values():
        if not isinstance(row, dict) or set(row) != set(CAMERA_FILES) or any(
            not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in row.values()
        ):
            raise ValueError("camera fingerprints require three SHA-256 hashes.")
    if not isinstance(manifest.get("protocol"), dict) or not manifest["protocol"]:
        raise ValueError("manifest must record its training protocol.")
    return json.loads(json.dumps(manifest, allow_nan=False))


def _sample_fingerprints(sample) -> dict:
    _, image, velodyne, calibration = sample
    result = {}
    for name, path in zip(CAMERA_FILES, (image, velodyne, calibration), strict=True):
        with Path(path).open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def build_camera_depth_manifest(dataset, *, train_indices, validation_indices, protocol: dict) -> dict:
    """Snapshot the actual dataset.samples ordering; annotations are not depth inputs."""

    ids = [sample[0] for sample in dataset.samples]
    train = [int(i) for i in train_indices]
    validation = [int(i) for i in validation_indices]
    return validate_camera_depth_manifest({
        "schema_version": MANIFEST_SCHEMA, "data_root": str(dataset.root.resolve()),
        "source_ids": ids, "train_indices": train, "validation_indices": validation,
        "train_ids": [ids[i] for i in train], "validation_ids": [ids[i] for i in validation],
        "fingerprints": {sample[0]: _sample_fingerprints(sample) for sample in dataset.samples},
        "protocol": protocol,
        "limitation": "Disjoint frame IDs, not a guarantee of independent sequences/scenes.",
    })


def verify_camera_depth_dataset(dataset, manifest: dict) -> None:
    """Refuse changed order/membership/content; permit relocation of identical data."""

    manifest = validate_camera_depth_manifest(manifest)
    if [sample[0] for sample in dataset.samples] != manifest["source_ids"]:
        raise ValueError("camera dataset IDs or order changed since checkpoint.")
    current = build_camera_depth_manifest(
        dataset, train_indices=manifest["train_indices"],
        validation_indices=manifest["validation_indices"], protocol=manifest["protocol"],
    )
    if {k: v for k, v in current.items() if k != "data_root"} != {
        k: v for k, v in manifest.items() if k != "data_root"
    }:
        raise ValueError("camera dataset IDs, order or file hashes changed since checkpoint.")


class VerifiedCameraDepthDataset:
    """Check each triplet before and after reading it, including during validation."""

    def __init__(self, dataset, manifest):
        self.dataset = dataset
        self.manifest = validate_camera_depth_manifest(manifest)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.samples[index]
        sample_id = self.manifest["source_ids"][index]
        expected = self.manifest["fingerprints"][sample_id]
        for phase in ("before", "after"):
            if sample[0] != sample_id or _sample_fingerprints(sample) != expected:
                raise ValueError(f"camera dataset changed {phase} reading {sample_id}.")
            if phase == "before":
                item = self.dataset[index]
        return item


def read_camera_depth_resume(path, *, dataset, protocol: dict, epochs: int) -> dict:
    """Validate on CPU before model/device initialization; never promote legacy weights."""

    import torch

    if not Path(path).is_file():
        raise ValueError(f"resume checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != CAMERA_DEPTH_CHECKPOINT_KIND or "data_manifest" not in payload:
        raise ValueError("legacy camera checkpoint has no recorded data manifest; it remains readable "
                         "for inference but cannot resume as a verified run. Start a new output directory.")
    manifest = validate_camera_depth_manifest(payload["data_manifest"])
    if manifest["protocol"] != json.loads(json.dumps(protocol)):
        raise ValueError("camera resume training/preprocessing protocol changed.")
    if any(isinstance(payload.get(k), bool) or not isinstance(payload.get(k), int) or payload[k] < 0
           for k in ("epoch", "step")) or epochs < payload["epoch"]:
        raise ValueError("invalid camera checkpoint counters or fewer requested epochs.")
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError("camera resume requires model and optimizer state.")
    verify_camera_depth_dataset(dataset, manifest)
    return payload


def write_camera_depth_manifest(path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(validate_camera_depth_manifest(manifest), indent=2) + "\n")
    temporary.replace(path)
