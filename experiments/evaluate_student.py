"""Evaluate seq2seq ER student checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

from models.classification_student import (
    ERClassificationPredictionDataset,
    ensure_padding_token,
)
from models.seq2seq_student import iter_jsonl, tokenize_seq2seq_inputs
from models.generative_reranker_student import (
    ERGenerativeRerankerDataset,
    RerankerDataCollator,
    load_merged_reranker,
)
from models.student_config import StudentConfig, load_student_config
from utils.metrics import compute_metrics
from utils.classification_threshold import load_decision_threshold


def parse_decision(text: str) -> Optional[bool]:
    """Parse generated text into True=match, False=non-match, or None=invalid."""
    normalized = text.strip().lower()
    if not normalized:
        return None

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "decision" in payload:
        return parse_decision(str(payload["decision"]))

    delimited = re.search(
        r"\[\[decision\]\]\s*(non-match|non match|no match|match)\b",
        normalized,
    )
    if delimited:
        return parse_decision(delimited.group(1))

    first = normalized.splitlines()[0].strip()
    if first.startswith("non-match") or first.startswith("non match") or first.startswith("no match"):
        return False
    if first.startswith("match"):
        return True
    if '"decision": "non-match"' in normalized or "'decision': 'non-match'" in normalized:
        return False
    if '"decision": "match"' in normalized or "'decision': 'match'" in normalized:
        return True
    return None


def _label_from_row(row: dict) -> bool:
    label = row["label"]
    if isinstance(label, str):
        normalized = label.strip().lower()
        if normalized in {"1", "true", "match"}:
            return True
        if normalized in {"0", "false", "non-match", "non_match", "no match"}:
            return False
        raise ValueError(f"Unsupported label value: {label}")
    return bool(label)


def inference_timing_metrics(inference_seconds: float, row_count: int) -> dict:
    """Return structured throughput metrics without applying a pricing assumption."""
    return {
        "student_inference_seconds": inference_seconds,
        "student_inference_rows_per_second": (
            row_count / inference_seconds if inference_seconds > 0 else None
        ),
        "student_inference_seconds_per_pair": (
            inference_seconds / row_count if row_count else None
        ),
    }


class PredictionDataset:
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_input_length: int,
        truncate_inputs: bool = True,
    ) -> None:
        self.rows = rows
        self.max_input_length = max_input_length
        encoded = tokenize_seq2seq_inputs(
            rows,
            tokenizer,
            max_input_length,
            truncate_inputs,
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def generate_predictions(
    checkpoint: Path,
    input_path: Path,
    output_path: Path,
    batch_size: int = 8,
    max_input_length: int = 512,
    max_new_tokens: int = 64,
    device: str | None = None,
    precision: str = "auto",
    truncate_inputs: bool = True,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from utils.torch_runtime import autocast_context, resolve_precision

    rows = list(iter_jsonl(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_precision = resolve_precision(device, precision)
    model.to(device)
    model.eval()

    dataset = PredictionDataset(
        rows,
        tokenizer,
        max_input_length=max_input_length,
        truncate_inputs=truncate_inputs,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[dict] = []
    cursor = 0
    device_object = torch.device(device)
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    inference_started = time.perf_counter()
    with torch.inference_mode(), autocast_context(device, resolved_precision):
        for batch in tqdm(loader, desc="Generating"):
            batch = {key: value.to(device) for key, value in batch.items()}
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for text in texts:
                row = rows[cursor]
                parsed = parse_decision(text)
                predictions.append(
                    {
                        "pair_id": row["pair_id"],
                        "label": row["label"],
                        "prediction_text": text,
                        "prediction": None if parsed is None else int(parsed),
                        "is_valid": parsed is not None,
                    }
                )
                cursor += 1
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    inference_seconds = time.perf_counter() - inference_started

    temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_output_path.replace(output_path)

    metrics = evaluate_prediction_rows(predictions)
    metrics["predictions"] = str(output_path)
    metrics["precision"] = resolved_precision
    metrics.update(inference_timing_metrics(inference_seconds, len(rows)))
    metrics.update(
        {
            "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
            "inference_device": str(device_object),
            "inference_device_name": (
                torch.cuda.get_device_name(device_object)
                if device_object.type == "cuda"
                else "cpu"
            ),
            "inference_batch_size": batch_size,
            "inference_max_input_length": max_input_length,
            "inference_max_new_tokens": max_new_tokens,
            "input_truncation": truncate_inputs,
        }
    )
    return metrics


def classify_predictions(
    config: StudentConfig,
    checkpoint: Path,
    input_path: Path,
    output_path: Path,
    batch_size: int = 8,
    max_input_length: int = 2400,
    device: str | None = None,
    precision: str = "auto",
) -> dict:
    """Evaluate a sequence-classification checkpoint with textual label serialization."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from utils.torch_runtime import autocast_context, resolve_precision

    rows = list(iter_jsonl(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        use_fast=config.tokenizer_use_fast,
    )
    ensure_padding_token(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    ensure_padding_token(tokenizer, model)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_precision = resolve_precision(device, precision)
    model.to(device)
    model.eval()
    dataset = ERClassificationPredictionDataset(
        rows,
        tokenizer,
        max_input_length=max_input_length,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    decision_threshold, threshold_source, threshold_payload = load_decision_threshold(
        checkpoint
    )

    predictions: list[dict] = []
    cursor = 0
    device_object = torch.device(device)
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    inference_started = time.perf_counter()
    with torch.inference_mode(), autocast_context(device, resolved_precision):
        for batch in tqdm(loader, desc="Classifying"):
            batch = {key: value.to(device) for key, value in batch.items()}
            probabilities = torch.softmax(model(**batch).logits.float(), dim=-1)
            match_probabilities = probabilities[:, config.label_to_id["match"]]
            predicted_matches = match_probabilities >= decision_threshold
            predicted_ids = torch.where(
                predicted_matches,
                torch.tensor(config.label_to_id["match"], device=probabilities.device),
                torch.tensor(config.label_to_id["non-match"], device=probabilities.device),
            )
            for row_index in range(predicted_ids.shape[0]):
                row = rows[cursor]
                prediction_id = int(predicted_ids[row_index].item())
                prediction_text = config.id_to_label[prediction_id]
                predictions.append(
                    {
                        "pair_id": row["pair_id"],
                        "label": row["label"],
                        "prediction_text": prediction_text,
                        "prediction": int(prediction_text == "match"),
                        "is_valid": True,
                        "non_match_probability": float(
                            probabilities[row_index, config.label_to_id["non-match"]].item()
                        ),
                        "match_probability": float(
                            probabilities[row_index, config.label_to_id["match"]].item()
                        ),
                    }
                )
                cursor += 1
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    inference_seconds = time.perf_counter() - inference_started

    temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_output_path.replace(output_path)

    metrics = evaluate_prediction_rows(predictions)
    metrics["predictions"] = str(output_path)
    metrics["precision"] = resolved_precision
    metrics.update(inference_timing_metrics(inference_seconds, len(rows)))
    metrics.update(
        {
            "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
            "inference_device": str(device_object),
            "inference_device_name": (
                torch.cuda.get_device_name(device_object)
                if device_object.type == "cuda"
                else "cpu"
            ),
            "inference_batch_size": batch_size,
            "inference_max_input_length": max_input_length,
            "inference_max_new_tokens": None,
            "decision_threshold": decision_threshold,
            "decision_threshold_source": threshold_source,
            "decision_threshold_selection_metric": (
                threshold_payload.get("selection_metric") if threshold_payload else None
            ),
        }
    )
    return metrics


def rerank_predictions(
    config: StudentConfig,
    checkpoint: Path,
    input_path: Path,
    output_path: Path,
    batch_size: int = 1,
    max_input_length: int = 4096,
    device: str | None = None,
    precision: str = "auto",
) -> dict:
    """Evaluate a merged causal-LM reranker through final yes/no token logits."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from utils.torch_runtime import autocast_context, resolve_precision

    rows = list(iter_jsonl(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_started = time.perf_counter()
    tokenizer, model = load_merged_reranker(config, checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_precision = resolve_precision(device, precision)
    model.to(device)
    model.eval()
    dataset = ERGenerativeRerankerDataset(
        rows,
        tokenizer,
        config,
        max_input_length,
        include_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=RerankerDataCollator(tokenizer),
    )
    decision_threshold, threshold_source, threshold_payload = load_decision_threshold(
        checkpoint
    )

    predictions: list[dict] = []
    cursor = 0
    device_object = torch.device(device)
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    inference_started = time.perf_counter()
    with torch.inference_mode(), autocast_context(device, resolved_precision):
        for batch in tqdm(loader, desc="Reranking"):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.float()
            probabilities = torch.softmax(logits, dim=-1)
            match_probabilities = probabilities[:, config.label_to_id["match"]]
            predicted_matches = match_probabilities >= decision_threshold
            for row_index in range(probabilities.shape[0]):
                row = rows[cursor]
                is_match = bool(predicted_matches[row_index].item())
                predictions.append(
                    {
                        "pair_id": row["pair_id"],
                        "label": row["label"],
                        "prediction_text": "match" if is_match else "non-match",
                        "prediction": int(is_match),
                        "is_valid": True,
                        "non_match_probability": float(
                            probabilities[
                                row_index,
                                config.label_to_id["non-match"],
                            ].item()
                        ),
                        "match_probability": float(
                            probabilities[
                                row_index,
                                config.label_to_id["match"],
                            ].item()
                        ),
                    }
                )
                cursor += 1
    if device_object.type == "cuda":
        torch.cuda.synchronize(device_object)
    inference_seconds = time.perf_counter() - inference_started

    temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_output_path.replace(output_path)

    metrics = evaluate_prediction_rows(predictions)
    metrics["predictions"] = str(output_path)
    metrics["precision"] = resolved_precision
    metrics.update(inference_timing_metrics(inference_seconds, len(rows)))
    metrics.update(
        {
            "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
            "inference_device": str(device_object),
            "inference_device_name": (
                torch.cuda.get_device_name(device_object)
                if device_object.type == "cuda"
                else "cpu"
            ),
            "inference_batch_size": batch_size,
            "inference_max_input_length": max_input_length,
            "inference_max_new_tokens": None,
            "input_truncation": False,
            "decision_threshold": decision_threshold,
            "decision_threshold_source": threshold_source,
            "decision_threshold_selection_metric": (
                threshold_payload.get("selection_metric")
                if threshold_payload
                else None
            ),
        }
    )
    return metrics


def evaluate_prediction_rows(rows: list[dict]) -> dict:
    labels = [_label_from_row(row) for row in rows]
    parsed = [None if row.get("prediction") is None else bool(row["prediction"]) for row in rows]
    valid_predictions = [prediction if prediction is not None else False for prediction in parsed]
    metrics = compute_metrics(valid_predictions, labels)
    invalid = sum(prediction is None for prediction in parsed)
    metrics.update(
        {
            "total": len(rows),
            "valid": len(rows) - invalid,
            "invalid": invalid,
            "invalid_output_rate": invalid / len(rows) if rows else 0.0,
        }
    )
    return metrics


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def append_summary_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "budget",
        "split",
        "same_precision",
        "same_recall",
        "same_f1",
        "macro_f1",
        "accuracy",
        "invalid_output_rate",
        "total",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a compact ER student checkpoint")
    parser.add_argument("--student-config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Serialized validation/test JSONL")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--variant", default="unknown")
    parser.add_argument("--budget", default="unknown")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    args = parser.parse_args()

    config = load_student_config(args.student_config) if args.student_config else None
    resolved_max_input_length = args.max_input_length
    if resolved_max_input_length is None:
        resolved_max_input_length = config.max_input_length if config else 512
    if config is not None and config.architecture == "sequence_classification":
        metrics = classify_predictions(
            config=config,
            checkpoint=args.checkpoint,
            input_path=args.input,
            output_path=args.predictions,
            batch_size=args.batch_size,
            max_input_length=resolved_max_input_length,
            device=args.device,
            precision=args.precision,
        )
    elif config is not None and config.architecture == "generative_reranker":
        metrics = rerank_predictions(
            config=config,
            checkpoint=args.checkpoint,
            input_path=args.input,
            output_path=args.predictions,
            batch_size=args.batch_size,
            max_input_length=resolved_max_input_length,
            device=args.device,
            precision=args.precision,
        )
    else:
        metrics = generate_predictions(
            checkpoint=args.checkpoint,
            input_path=args.input,
            output_path=args.predictions,
            batch_size=args.batch_size,
            max_input_length=resolved_max_input_length,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            precision=args.precision,
            truncate_inputs=config.input_truncation if config else True,
        )
    metrics.update(
        {
            "variant": args.variant,
            "budget": args.budget,
            "split": args.split,
            "student_id": config.student_id if config else "flan-t5-base",
            "model_name": config.model_name if config else "google/flan-t5-base",
            "architecture": config.architecture if config else "seq2seq",
        }
    )
    write_metrics(args.metrics, metrics)
    if args.summary_csv:
        append_summary_csv(args.summary_csv, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
