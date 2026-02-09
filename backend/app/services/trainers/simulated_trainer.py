"""
Simulated training backend for AI Hub.

This trainer exists to make the AI Hub "happy path" work in environments where
optional ML dependencies (torch/transformers/peft) and/or GPUs aren't available.
It emits realistic-looking progress callbacks and produces a tiny "adapter"
artifact directory.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from loguru import logger

from .base_trainer import BaseTrainer, DeviceInfo, TrainingProgress, TrainingResult


class SimulatedTrainer(BaseTrainer):
    """A lightweight trainer that simulates LoRA training."""

    def __init__(self) -> None:
        self._cancelled_jobs: set[str] = set()

    def is_available(self) -> bool:
        return True

    def get_device_info(self) -> DeviceInfo:
        return DeviceInfo(device="cpu", device_name="Simulated CPU", memory_total_gb=None)

    async def cancel(self, job_id: UUID) -> bool:
        self._cancelled_jobs.add(str(job_id))
        return True

    def get_supported_models(self) -> List[str]:
        return ["*"]

    def _is_cancelled(self, job_id: UUID) -> bool:
        return str(job_id) in self._cancelled_jobs

    async def train(
        self,
        job_id: UUID,
        dataset_path: str,
        base_model: str,
        output_path: str,
        hyperparameters: Dict[str, Any],
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TrainingResult:
        start = time.time()
        job_id_str = str(job_id)

        num_epochs = int(hyperparameters.get("num_epochs", 3) or 3)
        batch_size = int(hyperparameters.get("batch_size", 4) or 4)
        learning_rate = float(hyperparameters.get("learning_rate", 2e-4) or 2e-4)

        # Roughly infer dataset size from JSONL line count (best-effort).
        total_samples = 100
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                total_samples = max(1, sum(1 for _ in f))
        except Exception:
            pass

        total_steps = max(10, int((total_samples / max(1, batch_size)) * num_epochs))
        # Keep demos responsive regardless of dataset size.
        total_steps = min(total_steps, 400)

        logger.info(
            f"SimulatedTrainer starting job={job_id_str} base_model={base_model} "
            f"samples~={total_samples} steps={total_steps} epochs={num_epochs}"
        )

        best_loss = 9.0
        for step in range(1, total_steps + 1):
            if self._is_cancelled(job_id) or (cancel_check and cancel_check()):
                self._cancelled_jobs.discard(job_id_str)
                return TrainingResult(
                    success=False,
                    error="Cancelled",
                    total_steps=total_steps,
                    total_epochs=num_epochs,
                    training_time_seconds=int(time.time() - start),
                    metrics={"cancelled": True},
                )

            # Make loss decay smoothly + a tiny bit of noise.
            progress_frac = step / total_steps
            loss = max(0.05, 6.0 * (1.0 - progress_frac) + 0.2 * (0.5 - (step % 10) / 10.0))
            best_loss = min(best_loss, loss)

            epoch = max(1, int((step - 1) / max(1, total_steps / num_epochs)) + 1)
            eta = int((total_steps - step) * 0.15)

            if progress_callback and (step == 1 or step == total_steps or step % max(1, total_steps // 50) == 0):
                progress_callback(
                    TrainingProgress(
                        job_id=job_id_str,
                        step=step,
                        total_steps=total_steps,
                        epoch=epoch,
                        total_epochs=num_epochs,
                        loss=float(loss),
                        learning_rate=learning_rate,
                        samples_processed=min(total_samples * num_epochs, step * batch_size),
                        eta_seconds=eta,
                        metrics={"best_loss": float(best_loss)},
                        message="Simulated training step",
                    )
                )

            await asyncio.sleep(0.08)

        # Produce a tiny "adapter" artifact directory
        adapter_dir = Path(output_path) / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "trainer": "simulated",
            "job_id": job_id_str,
            "base_model": base_model,
            "created_at": time.time(),
            "hyperparameters": hyperparameters,
            "final_loss": float(best_loss),
            "note": "This is a simulated adapter artifact (no real weights).",
        }
        with open(adapter_dir / "adapter.json", "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

        with open(adapter_dir / "README.txt", "w", encoding="utf-8") as f:
            f.write("Simulated AI Hub adapter artifact. No real weights were produced.\n")

        adapter_size = 0
        try:
            adapter_size = sum(
                (adapter_dir / p).stat().st_size
                for p in os.listdir(adapter_dir)
                if (adapter_dir / p).is_file()
            )
        except Exception:
            pass

        self._cancelled_jobs.discard(job_id_str)
        return TrainingResult(
            success=True,
            adapter_path=str(adapter_dir),
            adapter_size=int(adapter_size),
            final_loss=float(best_loss),
            total_steps=total_steps,
            total_epochs=num_epochs,
            training_time_seconds=int(time.time() - start),
            samples_trained=int(total_samples * num_epochs),
            metrics={
                "training_loss": float(best_loss),
                "trainer": "simulated",
                "samples": int(total_samples),
                "epochs": int(num_epochs),
                "batch_size": int(batch_size),
            },
        )
