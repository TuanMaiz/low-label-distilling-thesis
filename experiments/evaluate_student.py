"""Evaluate Phase 03 seq2seq ER student checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

from models.mt5_student import iter_jsonl
from utils.metrics import compute_metrics


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


class PredictionDataset:
    def __init__(self, rows: list[dict], tokenizer, max_input_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.rows[idx]["input_text"],
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def generate_predictions(
    checkpoint: Path,
    input_path: Path,
    output_path: Path,
    batch_size: int = 8,
    max_input_length: int = 512,
    max_new_tokens: int = 64,
    device: str | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = list(iter_jsonl(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = PredictionDataset(rows, tokenizer, max_input_length=max_input_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    predictions: list[dict] = []
    cursor = 0
    with torch.no_grad():
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

    with output_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = evaluate_prediction_rows(predictions)
    metrics["predictions"] = str(output_path)
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
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Evaluate an mT5 ER student checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Serialized validation/test JSONL")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--variant", default="unknown")
    parser.add_argument("--budget", default="unknown")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device")
    args = parser.parse_args()

    metrics = generate_predictions(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_path=args.predictions,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    metrics.update({"variant": args.variant, "budget": args.budget, "split": args.split})
    write_metrics(args.metrics, metrics)
    if args.summary_csv:
        append_summary_csv(args.summary_csv, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
