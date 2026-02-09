"""
Local training backend using PEFT/transformers.

This trainer runs on the local machine using PyTorch, PEFT, and transformers.
Dependencies are optional and loaded lazily.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from loguru import logger

from app.core.config import settings
from .base_trainer import BaseTrainer, DeviceInfo, TrainingProgress, TrainingResult


class LocalTrainer(BaseTrainer):
    """
    Local LoRA/QLoRA training using PEFT and transformers.

    This trainer requires optional dependencies:
    - torch
    - transformers
    - peft
    - datasets
    - bitsandbytes (for QLoRA)
    - accelerate
    """

    def __init__(self):
        self._torch_available = False
        self._peft_available = False
        self._transformers_available = False
        self._device = None
        self._cancelled_jobs: set = set()
        self._check_dependencies()

    def _check_dependencies(self):
        """Check for optional training dependencies."""
        try:
            import torch
            self._torch_available = True
            self._torch = torch
        except ImportError:
            logger.warning("PyTorch not available for local training")

        try:
            import peft
            self._peft_available = True
        except ImportError:
            logger.warning("PEFT not available for local training")

        try:
            import transformers
            self._transformers_available = True
        except ImportError:
            logger.warning("transformers not available for local training")

    def is_available(self) -> bool:
        """Check if local training is available."""
        return (
            self._torch_available and
            self._peft_available and
            self._transformers_available
        )

    def get_device_info(self) -> DeviceInfo:
        """Get information about available compute devices."""
        if not self._torch_available:
            return DeviceInfo(device="cpu", device_name="CPU (PyTorch not installed)")

        torch = self._torch

        # Check for CUDA
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_free = (
                torch.cuda.get_device_properties(0).total_memory -
                torch.cuda.memory_allocated(0)
            ) / (1024**3)

            cuda_version = torch.version.cuda
            compute_cap = torch.cuda.get_device_capability(0)

            return DeviceInfo(
                device="cuda",
                device_name=device_name,
                memory_total_gb=round(memory_total, 1),
                memory_available_gb=round(memory_free, 1),
                cuda_version=cuda_version,
                compute_capability=f"{compute_cap[0]}.{compute_cap[1]}",
            )

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceInfo(
                device="mps",
                device_name="Apple Silicon",
            )

        # CPU fallback
        return DeviceInfo(device="cpu", device_name="CPU")

    def get_supported_models(self) -> List[str]:
        """Get list of supported base models."""
        return [
            "llama3.2:1b",
            "llama3.2:3b",
            "llama3.1:8b",
            "mistral:7b",
            "phi3:mini",
            "gemma:2b",
            "gemma:7b",
            "qwen2:0.5b",
            "qwen2:1.5b",
            "qwen2:7b",
        ]

    async def cancel(self, job_id: UUID) -> bool:
        """Request cancellation of a training job."""
        self._cancelled_jobs.add(str(job_id))
        logger.info(f"Cancellation requested for job {job_id}")
        return True

    def _is_cancelled(self, job_id: UUID) -> bool:
        """Check if a job has been cancelled."""
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
        """
        Execute local LoRA training.

        Args:
            job_id: Unique job identifier
            dataset_path: Path to training dataset JSONL file
            base_model: Base model name from Ollama
            output_path: Directory to save the trained adapter
            hyperparameters: Training hyperparameters
            progress_callback: Callback for progress updates
            cancel_check: Function to check for cancellation

        Returns:
            TrainingResult with success status and adapter path
        """
        if not self.is_available():
            return TrainingResult(
                success=False,
                error=(
                    "Local training dependencies not available. "
                    "Install with: pip install peft transformers datasets bitsandbytes accelerate"
                ),
            )

        start_time = time.time()
        job_id_str = str(job_id)

        try:
            # Run training in thread pool to not block async loop
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._train_sync(
                    job_id_str,
                    dataset_path,
                    base_model,
                    output_path,
                    hyperparameters,
                    progress_callback,
                    cancel_check,
                ),
            )

            result.training_time_seconds = int(time.time() - start_time)
            return result

        except Exception as e:
            logger.exception(f"Training failed for job {job_id}: {e}")
            return TrainingResult(
                success=False,
                error=str(e),
                training_time_seconds=int(time.time() - start_time),
            )
        finally:
            # Clean up cancellation flag
            self._cancelled_jobs.discard(job_id_str)

    def _train_sync(
        self,
        job_id: str,
        dataset_path: str,
        base_model: str,
        output_path: str,
        hyperparameters: Dict[str, Any],
        progress_callback: Optional[Callable[[TrainingProgress], None]],
        cancel_check: Optional[Callable[[], bool]],
    ) -> TrainingResult:
        """
        Synchronous training implementation.

        This runs in a thread pool executor.
        """
        # Late imports to avoid loading heavy libraries until needed
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset

        # Determine device
        device_config = hyperparameters.get("device", settings.TRAINING_LOCAL_DEVICE)
        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = device_config

        logger.info(f"Training on device: {device}")

        # Get hyperparameters with defaults
        lora_r = hyperparameters.get("lora_r", 16)
        lora_alpha = hyperparameters.get("lora_alpha", 32)
        lora_dropout = hyperparameters.get("lora_dropout", 0.05)
        target_modules = hyperparameters.get("target_modules", ["q_proj", "v_proj"])
        learning_rate = hyperparameters.get("learning_rate", 2e-4)
        num_epochs = hyperparameters.get("num_epochs", 3)
        batch_size = hyperparameters.get("batch_size", 4)
        gradient_accumulation = hyperparameters.get("gradient_accumulation_steps", 4)
        warmup_steps = hyperparameters.get("warmup_steps", 100)
        max_seq_length = hyperparameters.get("max_seq_length", 2048)
        training_method = hyperparameters.get("training_method", "lora")

        # Resolve model path from Ollama
        # Ollama stores models in ~/.ollama/models/blobs
        # For now, we'll assume the model is from HuggingFace
        # In production, we'd need to convert or use the Ollama API
        model_name = self._resolve_model_name(base_model)
        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization if QLoRA
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
        }

        if training_method == "qlora" and device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Apply PEFT
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        logger.info(f"Loaded {len(dataset)} samples")

        # Tokenize dataset
        def tokenize_function(examples):
            # Assuming alpaca format
            prompts = []
            for i in range(len(examples["instruction"])):
                instruction = examples["instruction"][i]
                input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
                output = examples["output"][i]

                if input_text:
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                prompts.append(prompt)

            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Training arguments
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=10,
            save_steps=settings.TRAINING_CHECKPOINT_INTERVAL_STEPS,
            save_total_limit=3,
            fp16=(device == "cuda"),
            bf16=False,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            report_to="none",
            disable_tqdm=True,
        )

        # Custom callback for progress
        from transformers import TrainerCallback

        class ProgressCallback(TrainerCallback):
            def __init__(self, total_steps, progress_callback, cancel_check):
                self.total_steps = total_steps
                self.progress_callback = progress_callback
                self.cancel_check = cancel_check
                self.loss_history = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    self.loss_history.append(logs["loss"])

                    if self.progress_callback:
                        progress = TrainingProgress(
                            job_id=job_id,
                            step=state.global_step,
                            total_steps=self.total_steps,
                            epoch=int(state.epoch) if state.epoch else 0,
                            total_epochs=num_epochs,
                            loss=logs["loss"],
                            learning_rate=logs.get("learning_rate", 0),
                            samples_processed=state.global_step * batch_size * gradient_accumulation,
                            metrics={"loss": logs["loss"]},
                        )
                        self.progress_callback(progress)

                # Check for cancellation
                if self.cancel_check and self.cancel_check():
                    control.should_training_stop = True
                    logger.info("Training cancelled by user")

        # Calculate total steps
        total_steps = (len(tokenized_dataset) // (batch_size * gradient_accumulation)) * num_epochs

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[ProgressCallback(total_steps, progress_callback, cancel_check)],
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save final adapter
        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        # Calculate adapter size
        adapter_size = sum(
            f.stat().st_size for f in adapter_path.rglob("*") if f.is_file()
        )

        # Get final metrics
        final_loss = train_result.training_loss

        logger.info(f"Training complete. Final loss: {final_loss}")

        return TrainingResult(
            success=True,
            adapter_path=str(adapter_path),
            adapter_size=adapter_size,
            final_loss=final_loss,
            total_steps=train_result.global_step,
            total_epochs=num_epochs,
            samples_trained=len(tokenized_dataset) * num_epochs,
            metrics={
                "training_loss": final_loss,
                "runtime": train_result.metrics.get("train_runtime", 0),
                "samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            },
        )

    def _resolve_model_name(self, ollama_model: str) -> str:
        """
        Resolve Ollama model name to HuggingFace model name.

        In production, this would use Ollama's model manifest or
        a mapping table. For now, we use common mappings.
        """
        mappings = {
            "llama3.2:1b": "meta-llama/Llama-3.2-1B",
            "llama3.2:3b": "meta-llama/Llama-3.2-3B",
            "llama3.1:8b": "meta-llama/Meta-Llama-3.1-8B",
            "mistral:7b": "mistralai/Mistral-7B-v0.1",
            "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",
            "gemma:2b": "google/gemma-2b",
            "gemma:7b": "google/gemma-7b",
            "qwen2:0.5b": "Qwen/Qwen2-0.5B",
            "qwen2:1.5b": "Qwen/Qwen2-1.5B",
            "qwen2:7b": "Qwen/Qwen2-7B",
        }

        # Try exact match
        if ollama_model in mappings:
            return mappings[ollama_model]

        # Try without version tag
        base_name = ollama_model.split(":")[0]
        for key, value in mappings.items():
            if key.startswith(base_name):
                return value

        # Assume it's already a HuggingFace model name
        return ollama_model
