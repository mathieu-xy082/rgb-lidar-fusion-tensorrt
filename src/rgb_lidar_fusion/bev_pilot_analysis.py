"""CPU-only pilot curves, validation inspection and matched-run comparison."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .bev_analysis import BEVFusionSample, analyze_bev_fusion_frame
from .bev_checkpoint import load_bev_fusion_checkpoint, validate_bev_fusion_split
from .bev_data import BEVFusionFrameSource
from .bev_pilot import BEVFusionPilotConfig, VARIANTS
from .bev_trainval import TASK, validate_bev_pilot_resume


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_evaluation(row, split, name):
    prefix = "train_eval" if name == "train" else "val"
    partition = "train_ids" if name == "train" else "validation_ids"
    frames, skipped = row.get(name + "_per_frame"), row.get(name + "_skipped_ids")
    if not isinstance(frames, list) or not frames or not isinstance(skipped, list):
        raise ValueError("pilot history requires per-frame evaluation and explicit skipped IDs.")
    ids = [frame.get("sample_id") for frame in frames]
    skipped_set = set(skipped)
    if (len(skipped_set) != len(skipped) or not skipped_set <= set(split[partition])
            or ids != [s for s in split[partition] if s not in skipped_set]):
        raise ValueError("evaluation IDs do not match the saved partition/domain.")
    for metric in ("loss", "heatmap", "offset"):
        if not _finite(row.get(prefix + "_" + metric)) or row[prefix + "_" + metric] < 0:
            raise ValueError("pilot evaluation losses must be finite and nonnegative.")
    for metric in ("tp", "fp", "fn"):
        counts = [frame.get(metric) for frame in frames]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
            raise ValueError("pilot evaluation counts must be nonnegative integers.")
        if row.get(name + "_" + metric) != sum(counts):
            raise ValueError("aggregate center counts differ from per-frame evaluation.")
    tp, fp, fn = (row[name + "_" + key] for key in ("tp", "fp", "fn"))
    expected = {"precision": tp / (tp + fp) if tp + fp else 0.,
                "recall": tp / (tp + fn) if tp + fn else None}
    distances = [frame.get("matched_distance_sum_m") for frame in frames]
    if any(not _finite(value) or value < 0 for value in distances):
        raise ValueError("matched distances must be finite and nonnegative.")
    expected["matched_mean_distance_m"] = sum(distances) / tp if tp else None
    for metric, value in expected.items():
        actual = row.get(name + "_" + metric)
        if value is None:
            if actual is not None:
                raise ValueError("undefined recall/distance must remain None, not zero.")
        elif not _finite(actual) or not math.isclose(actual, value, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("evaluation ratios/distances differ from their counts.")
    return ids, skipped


def load_bev_pilot_run(path, *, verify_files=True):
    """Load an authoritative latest/resume checkpoint for analysis, never Adam.

    The full saved history, rather than possibly stale JSON sidecars or a best
    checkpoint's truncated history, describes completed budget and selection.
    Original pretrained encoder files are not required. Current dataset content
    is verified by default; callers doing offline metadata analysis can explicitly
    disable only this live-file check, not provenance or compatibility checks.
    """

    model, payload = load_bev_fusion_checkpoint(path)
    metadata = payload.get("training_metadata", {})
    if metadata.get("task") != TASK or metadata.get("variant") not in VARIANTS:
        raise ValueError("expected a BEV pilot checkpoint, not the four-frame diagnostic.")
    if not isinstance(metadata.get("runner_config"), dict):
        raise ValueError("pilot checkpoint must record its runner configuration.")
    config = BEVFusionPilotConfig.from_mapping(metadata["runner_config"])
    split = validate_bev_fusion_split(payload["split"], verify_files=verify_files)
    validate_bev_pilot_resume(payload, config=config, variant=metadata["variant"], split=split)
    expected = {"batch_size": 1, "precision": "float32", "freeze_encoders": True, "depth_holdout": False}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported pilot precision/batch/freeze/holdout policy.")
    for key in ("matching", "selection_metric", "loss_aggregation", "torch_version"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError("pilot must record evaluation and software policy.")
    for name in ("camera", "lidar"):
        digest = payload["pretraining"][name].get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("pilot must record source checkpoint SHA-256 hashes.")
    domains = None
    for row in payload["metrics"]:
        current = tuple(_validate_evaluation(row, split, name) for name in ("train", "val"))
        if domains is not None and current != domains:
            raise ValueError("evaluation domain changed between epochs.")
        domains = current
    return model, config, payload


def _comparison_protocol(config):
    operational = {"data_root", "output_dir", "split_path", "camera_checkpoint", "lidar_checkpoint",
                   "device", "cpu_threads", "minimum_free_memory_mib", "log_every_steps"}
    return {key: value for key, value in config.serialized().items() if key not in operational}


def _validate_pair(runs, *, require_complete):
    if set(runs) != set(VARIANTS):
        raise ValueError("comparison requires both lidar_only and fusion runs.")
    first, second = (runs[name] for name in VARIANTS)
    first_meta, second_meta = (run["training_metadata"] for run in (first, second))
    configs = [BEVFusionPilotConfig.from_mapping(meta["runner_config"]) for meta in (first_meta, second_meta)]
    if _comparison_protocol(configs[0]) != _comparison_protocol(configs[1]):
        raise ValueError("pilot learning, preprocessing or evaluation protocol differs.")
    if any(first["split"][key] != second["split"][key]
           for key in ("source_ids", "train_ids", "validation_ids", "fingerprints")):
        raise ValueError("pilot dataset/split membership or hashes differ.")
    for name in ("camera", "lidar"):
        if first["pretraining"][name]["sha256"] != second["pretraining"][name]["sha256"]:
            raise ValueError("pilot pretrained encoder sources differ.")
    if (first_meta["initialization"]["new_blocks_sha256"] != second_meta["initialization"]["new_blocks_sha256"]
            or first["trainable_names"] != second["trainable_names"]):
        raise ValueError("pilot fresh new-block initialization or trainable policy differs.")
    for key in ("batch_size", "precision", "freeze_encoders", "depth_holdout", "matching",
                "selection_metric", "loss_aggregation", "torch_version"):
        if first_meta.get(key) != second_meta.get(key):
            raise ValueError(f"pilot recorded {key} differs.")
    for name, run in runs.items():
        if run["training_metadata"]["variant"] != name:
            raise ValueError("pilot variant labels do not match their checkpoints.")
    for name in ("train", "val"):
        if ([row["sample_id"] for row in first["metrics"][0][name + "_per_frame"]]
                != [row["sample_id"] for row in second["metrics"][0][name + "_per_frame"]]
                or first["metrics"][0][name + "_skipped_ids"] != second["metrics"][0][name + "_skipped_ids"]):
            raise ValueError("pilot evaluated domains differ.")
    if require_complete:
        if any(run["epoch"] != config.epochs for run, config in zip((first, second), configs, strict=True)):
            raise ValueError("final comparison requires both runs to complete the agreed epoch budget.")
        if first["step"] != second["step"]:
            raise ValueError("completed pilot optimizer-step budgets differ.")


def _summary(row):
    return {"epoch": row["epoch"], "step": row["step"],
            **{key: row["val_" + key] for key in ("loss", "heatmap", "offset", "tp", "fp", "fn",
                                                 "precision", "recall", "matched_mean_distance_m")}}


def compare_bev_pilot_runs(lidar_only_path, fusion_path, *, verify_files=True):
    """Summarize equal completed budgets, with final AND independently selected best scores."""

    paths = {"lidar_only": lidar_only_path, "fusion": fusion_path}
    runs = {name: load_bev_pilot_run(path, verify_files=verify_files)[2] for name, path in paths.items()}
    _validate_pair(runs, require_complete=True)
    return {
        "task": "bev_fusion_pilot_comparison", "complete": True,
        "live_dataset_files_verified": verify_files,
        "selection": "Best: minimum validation loss per run, including epoch 0. Final: equal completed epoch budget.",
        "limitation": "Frame-disjoint validation used for selection, not independent scenes, an unbiased test set or 3D AP. Matched distance only concerns each model's matched centers.",
        "train_count": len(runs["fusion"]["split"]["train_ids"]),
        "validation_count": len(runs["fusion"]["split"]["validation_ids"]),
        "variants": {name: {"checkpoint": str(Path(paths[name]).resolve()),
                             "epochs_completed": run["epoch"], "optimizer_steps": run["step"],
                             "execution_device": run["training_metadata"]["runner_config"]["device"],
                             "best": _summary(min(run["metrics"], key=lambda row: row["val_loss"])),
                             "final": _summary(run["metrics"][-1])} for name, run in runs.items()},
    }


def plot_bev_pilot_curves(runs):
    """Allow partial curves, but reject mismatched protocols before overlaying two runs."""

    import matplotlib.pyplot as plt

    if not runs or not set(runs) <= set(VARIANTS):
        raise ValueError("provide at least one pilot variant.")
    if any(run["training_metadata"]["variant"] != name for name, run in runs.items()):
        raise ValueError("curve variant labels do not match their checkpoints.")
    if len(runs) == 2:
        _validate_pair(runs, require_complete=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"lidar_only": "tab:orange", "fusion": "tab:blue"}
    for name, run in runs.items():
        rows = run["metrics"]
        epochs = [row["epoch"] for row in rows]
        for key, style in (("train_eval_loss", "--"), ("val_loss", "-")):
            axes[0, 0].plot(epochs, [row[key] for row in rows], style, color=colors[name], label=name + " " + key)
        for key, style in (("precision", "-"), ("recall", "--")):
            axes[0, 1].plot(epochs, [row["val_" + key] for row in rows], style,
                            color=colors[name], label=name + " " + key)
        for key, style in (("tp", "-"), ("fp", ":"), ("fn", "--")):
            axes[1, 0].plot(epochs, [row["val_" + key] for row in rows], style,
                            color=colors[name], label=name + " " + key)
        axes[1, 1].plot(epochs, [np.nan if row["val_matched_mean_distance_m"] is None else
                                row["val_matched_mean_distance_m"] for row in rows],
                       color=colors[name], label=name)
    axes[0, 0].set(title="Train / validation loss", ylabel="Loss", yscale="log")
    axes[0, 1].set(title="Validation precision / recall", ylabel="Fraction", ylim=(-.05, 1.05))
    axes[1, 0].set(title="Validation center counts", ylabel="Count")
    axes[1, 1].set(title="Matched centers only", ylabel="Mean XY error (m)")
    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class BEVPilotValidation:
    sample: BEVFusionSample
    config: BEVFusionPilotConfig
    variant: str
    selection: str
    checkpoint_epoch: int


def analyze_bev_pilot_validation(path, *, validation_position=0, selection="best") -> BEVPilotValidation:
    """Inspect ONLY a saved validation ID on CPU; source encoder files are unnecessary."""

    if selection not in ("best", "latest"):
        raise ValueError("selection must be best or latest.")
    model, config, payload = load_bev_pilot_run(path)
    ids = payload["split"]["validation_ids"]
    if (isinstance(validation_position, bool) or not isinstance(validation_position, int)
            or not 0 <= validation_position < len(ids)):
        raise ValueError("validation_position must index the saved validation partition.")
    epoch = payload["epoch"]
    if selection == "best":
        model.load_state_dict(payload["best"]["model_state_dict"], strict=True)
        epoch = payload["best"]["epoch"]
    source = BEVFusionFrameSource(config.data_root, model_config=model.config, split=payload["split"])
    sample = analyze_bev_fusion_frame(
        model, source.read(ids[validation_position]), score_threshold=config.score_threshold,
        top_k=config.top_k, match_radius_m=config.match_radius_m,
    )
    return BEVPilotValidation(sample, config, payload["training_metadata"]["variant"], selection, epoch)
