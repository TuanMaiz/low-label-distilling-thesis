"""Train a config-selected compact student for WDC Entity Matching."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import get_scheduler

from experiments.trainer import Trainer
from models.classification_student import ERClassificationDataset, load_sequence_classifier
from models.seq2seq_student import ERSeq2SeqDataset, load_seq2seq, load_target_rows
from models.student_config import StudentConfig, load_student_config
from utils.torch_runtime import (
    PRECISION_CHOICES,
    resolve_validation_batch_size as resolve_runtime_validation_batch_size,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model_and_datasets(
    config: StudentConfig,
    train_rows: list[dict],
    validation_rows: list[dict],
    max_input_length: int,
    max_target_length: int,
):
    if config.architecture == "seq2seq":
        tokenizer, model = load_seq2seq(
            config.model_name,
            use_fast=config.tokenizer_use_fast,
        )
        train_dataset = ERSeq2SeqDataset(
            train_rows,
            tokenizer,
            max_input_length,
            max_target_length,
        )
        validation_dataset = ERSeq2SeqDataset(
            validation_rows,
            tokenizer,
            max_input_length,
            max_target_length,
        )
        return tokenizer, model, train_dataset, validation_dataset
    if config.architecture == "sequence_classification":
        tokenizer, model = load_sequence_classifier(config)
        train_dataset = ERClassificationDataset(
            train_rows,
            tokenizer,
            config.label_to_id,
            max_input_length,
        )
        validation_dataset = ERClassificationDataset(
            validation_rows,
            tokenizer,
            config.label_to_id,
            max_input_length,
        )
        return tokenizer, model, train_dataset, validation_dataset
    raise ValueError(f"Unsupported student architecture: {config.architecture}")


def train_configured_student(
    student_config: Path,
    train_targets: Path,
    validation_targets: Path,
    output_dir: Path,
    batch_size: int = 4,
    num_epochs: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 0,
    max_input_length: int = 512,
    max_target_length: int = 8,
    seed: int = 42,
    device: str | None = None,
    use_wandb: bool = False,
    early_stopping_patience: int = 3,
    validation_batch_size: int | None = None,
    precision: str = "auto",
) -> dict:
    config = load_student_config(student_config)
    set_seed(seed)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_validation_batch_size = resolve_runtime_validation_batch_size(
        resolved_device,
        batch_size,
        validation_batch_size,
        precision,
    )
    train_rows = load_target_rows(train_targets)
    validation_rows = load_target_rows(validation_targets)
    tokenizer, model, train_dataset, validation_dataset = _load_model_and_datasets(
        config,
        train_rows,
        validation_rows,
        max_input_length,
        max_target_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=resolved_validation_batch_size,
        shuffle=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        precision=precision,
        wandb_project="distiller-wdc-er" if use_wandb else None,
    )
    trainer.scheduler = get_scheduler(
        "linear",
        optimizer=trainer.optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, len(train_loader) * num_epochs),
    )
    training_device = torch.device(resolved_device)
    if training_device.type == "cuda":
        torch.cuda.synchronize(training_device)
    training_started = time.perf_counter()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=validation_loader,
        num_epochs=num_epochs,
        save_dir=str(output_dir),
        early_stopping_patience=early_stopping_patience,
    )
    if training_device.type == "cuda":
        torch.cuda.synchronize(training_device)
    training_wall_seconds = time.perf_counter() - training_started

    summary = {
        "student_id": config.student_id,
        "model_name": config.model_name,
        "architecture": config.architecture,
        "student_config": str(student_config),
        "train_targets": str(train_targets),
        "validation_targets": str(validation_targets),
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "batch_size": batch_size,
        "validation_batch_size": resolved_validation_batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length if config.architecture == "seq2seq" else None,
        "seed": seed,
        "early_stopping_patience": early_stopping_patience,
        "precision_requested": precision,
        "precision": trainer.precision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(training_device)
            if training_device.type == "cuda"
            else None
        ),
        "training_wall_seconds": training_wall_seconds,
        "training_gpu_hours": training_wall_seconds / 3600.0,
        "training_time_scope": (
            "trainer.train, including epoch validation and checkpointing; "
            "excluding model loading and dataset tokenization"
        ),
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "training_summary.json"
    temporary_summary_path = output_dir / "training_summary.json.tmp"
    temporary_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_summary_path.replace(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-config", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--precision", choices=PRECISION_CHOICES, default="auto")
    args = parser.parse_args()
    summary = train_configured_student(
        student_config=args.student_config,
        train_targets=args.train_targets,
        validation_targets=args.validation_targets,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        seed=args.seed,
        device=args.device,
        use_wandb=args.use_wandb,
        early_stopping_patience=args.early_stopping_patience,
        precision=args.precision,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
