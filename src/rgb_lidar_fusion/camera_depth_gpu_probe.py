"""Bounded, synthetic camera-depth diagnostics for a standalone process."""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from .camera_depth_analysis import require_gpu_ready, run_gpu_preflight
from .camera_depth_model import CameraDepthModel, masked_depth_loss
from .training import load_checkpoint, select_device, set_deterministic_seed


def nvidia_snapshot() -> dict[str, Any]:
    """Collect driver diagnostics without issuing CUDA work, even after failure."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


def run_camera_depth_probe(
    *,
    emit: Callable[[dict[str, Any]], None],
    steps: int = 5,
    height: int = 320,
    width: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    checkpoint: Path | None = None,
    cudnn_enabled: bool = True,
) -> None:
    """Exercise FP32 training without saving weights or reading KITTI data.

    CPU mode is for testing this diagnostic, not for validating GPU health.
    Events precede synchronized phases so a partial log identifies the last
    attempted operation. Never retry CUDA or collect CUDA counters on failure.
    """

    for name, value in {
        "steps": steps,
        "height": height,
        "width": width,
        "batch_size": batch_size,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    phase = "environment"
    step = 0
    start = time.monotonic()

    def record(event: str, **fields: Any) -> None:
        emit({"event": event, "elapsed_s": time.monotonic() - start, **fields})

    def begin(name: str) -> None:
        nonlocal phase
        phase = name
        record("phase_started", step=step, phase=phase)

    try:
        record(
            "environment",
            python=platform.python_version(),
            pytorch=str(torch.__version__),
            cuda_build=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            requested_device=device,
            steps=steps,
            input_shape=[batch_size, 3, height, width],
            lidar_channels=8,
            dtype="float32",
            checkpoint=str(checkpoint) if checkpoint else None,
            cudnn_enabled=cudnn_enabled,
            cudnn_benchmark=False,
            nvidia_smi=nvidia_snapshot(),
        )
        begin("preflight")
        if device == "cuda":
            report = run_gpu_preflight(run_smoke_test=False)
            record("preflight", report=asdict(report))
            require_gpu_ready(report)
        selected_device = select_device(device)
        set_deterministic_seed(13)

        def synchronize() -> None:
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)

        with torch.backends.cudnn.flags(
            enabled=cudnn_enabled, benchmark=False, deterministic=False,
        ):
            begin("model_setup")
            model = CameraDepthModel(lidar_mode="enriched").to(selected_device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            if checkpoint is not None:
                epoch, saved_step = load_checkpoint(
                    path=checkpoint, model=model, optimizer=optimizer,
                    device=selected_device,
                )
                record("checkpoint_loaded", epoch=epoch, saved_step=saved_step)
            model.train()
            synchronize()
            if selected_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(selected_device)

            # Reuse a fixed synthetic batch to isolate kernels from data loading.
            generator = torch.Generator(device="cpu").manual_seed(13)
            shape = (batch_size, 1, height, width)
            rgb_cpu = torch.rand((batch_size, 3, height, width), generator=generator)
            lidar_cpu = torch.rand((batch_size, 8, height, width), generator=generator)
            target_cpu = torch.rand(shape, generator=generator)
            mask_cpu = torch.rand(shape, generator=generator) < 0.02
            mask_cpu[:, :, 0, 0] = True

            for step in range(1, steps + 1):
                step_start = time.monotonic()
                begin("input_transfer")
                rgb = rgb_cpu.to(selected_device)
                lidar = lidar_cpu.to(selected_device)
                target = target_cpu.to(selected_device)
                mask = mask_cpu.to(selected_device)
                synchronize()
                optimizer.zero_grad(set_to_none=True)

                begin("forward")
                prediction = model(rgb, lidar)
                loss = masked_depth_loss(prediction, target, mask)
                synchronize()
                if not torch.isfinite(loss):
                    raise ValueError("Probe loss must be finite.")

                begin("backward")
                loss.backward()
                synchronize()

                begin("gradient_check")
                grad_norm_sq = 0.0
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise ValueError("Probe gradients must be finite.")
                        grad_norm_sq += float(parameter.grad.detach().square().sum().cpu())
                if grad_norm_sq <= 0.0:
                    raise ValueError("Probe gradients must not all be zero.")

                begin("optimizer")
                optimizer.step()
                synchronize()
                memory = {}
                if selected_device.type == "cuda":
                    free, total = torch.cuda.mem_get_info(selected_device)
                    memory = {
                        "allocated_mib": torch.cuda.memory_allocated(selected_device) / 2**20,
                        "reserved_mib": torch.cuda.memory_reserved(selected_device) / 2**20,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated(selected_device) / 2**20,
                        "free_mib": free / 2**20,
                        "total_mib": total / 2**20,
                    }
                record(
                    "step_completed", step=step, loss=float(loss.detach().cpu()),
                    grad_norm=grad_norm_sq**0.5,
                    duration_s=time.monotonic() - step_start, memory=memory,
                )
                del prediction, loss
            record("completed", steps=steps, nvidia_smi=nvidia_snapshot())
    except Exception as exc:
        record(
            "failed", step=step, phase=phase,
            error_type=type(exc).__name__, error=str(exc),
            nvidia_smi=nvidia_snapshot(),
        )
        raise
