"""Train compact seq2seq student models for WDC Entity Matching."""
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
from models.seq2seq_student import (
    DEFAULT_SEQ2SEQ_MODEL,
    ERSeq2SeqDataset,
    load_seq2seq,
    load_target_rows,
)
from supervision.build_targets import build_targets
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


def resolve_validation_batch_size(
    device: str,
    train_batch_size: int,
    requested: int | None = None,
    precision: str = "auto",
) -> int:
    """Choose a larger evaluation-only batch without changing training updates."""
    return resolve_runtime_validation_batch_size(
        device,
        train_batch_size,
        requested,
        precision,
    )


def train_student(
    train_targets: Path,
    validation_targets: Path,
    output_dir: Path,
    model_name: str = DEFAULT_SEQ2SEQ_MODEL,
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
    set_seed(seed)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_validation_batch_size = resolve_validation_batch_size(
        resolved_device,
        batch_size,
        validation_batch_size,
        precision,
    )
    tokenizer, model = load_seq2seq(model_name)
    train_rows = load_target_rows(train_targets)
    validation_rows = load_target_rows(validation_targets)

    train_dataset = ERSeq2SeqDataset(train_rows, tokenizer, max_input_length, max_target_length)
    validation_dataset = ERSeq2SeqDataset(validation_rows, tokenizer, max_input_length, max_target_length)
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
        "model_name": model_name,
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
        "max_target_length": max_target_length,
        "seed": seed,
        "early_stopping_patience": early_stopping_patience,
        "precision_requested": precision,
        "precision": trainer.precision,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(resolved_device))
            if torch.device(resolved_device).type == "cuda"
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
    parser = argparse.ArgumentParser(description="Train a seq2seq ER student")
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--validation-targets", type=Path)
    parser.add_argument("--validation-pairs", type=Path, default=Path("data/cache/wdc_products/serialized/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_SEQ2SEQ_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        help="Validation-only batch size; defaults to 32 on BF16 CUDA, 16 on other CUDA",
    )
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

    validation_targets = args.validation_targets
    if validation_targets is None:
        validation_targets = args.output_dir / "validation.gold_label.targets.jsonl"
        build_targets(
            pairs_path=args.validation_pairs,
            output_path=validation_targets,
            variant="gold_label",
        )

    summary = train_student(
        train_targets=args.train_targets,
        validation_targets=validation_targets,
        output_dir=args.output_dir,
        model_name=args.model_name,
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
