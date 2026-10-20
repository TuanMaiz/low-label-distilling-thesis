"""Record-order swap probe for the Qwen generative reranker.

Entity identity is symmetric, so a matcher should score (A, B) and (B, A)
alike. The Qwen reranker places Record A in ``<Query>`` and Record B in
``<Document>`` under a relevance-style system prompt, and every training pair
was seen in one order only. This probe scores each pair in its original order
and with the two record bodies exchanged, then summarizes how much and in which
direction the score moves.

Two subcommands:

``score``
    Inference only. Loads the base model, a LoRA adapter directory, or a merged
    checkpoint; writes one JSON line per (pair, order) to ``scores.jsonl``.
    Re-running the same command resumes and skips completed rows.

``summarize``
    CPU only. Joins the scores with gold labels, checks reproduction against
    saved predictions, and reports asymmetry by outcome group, a vaguer-record
    direction test, and metrics for order-fused scores.

Gold labels are used only for evaluation and error grouping. The probe trains
nothing, calls no LLM, and changes no labels or thresholds of existing runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from models.generative_reranker_student import (
    RECORD_A_MARKER,
    RECORD_B_MARKER,
    split_product_records,
)
from models.seq2seq_student import iter_jsonl
from utils.metrics import compute_metrics

ORDERS = ("original", "swapped")
FUSIONS = ("mean", "min")
SCORES_FILENAME = "scores.jsonl"
RUN_FILENAME = "run.json"


# ---------------------------------------------------------------------------
# Input orientation
# ---------------------------------------------------------------------------

def swap_record_order(input_text: str) -> str:
    """Exchange the Record A and Record B bodies; keep header and labels fixed."""
    if RECORD_A_MARKER not in input_text or RECORD_B_MARKER not in input_text:
        raise ValueError("Input is missing a Record A or Record B boundary")
    header, pair_text = input_text.split(RECORD_A_MARKER, maxsplit=1)
    body_a, body_b = pair_text.split(RECORD_B_MARKER, maxsplit=1)
    return f"{header}{RECORD_A_MARKER}{body_b}{RECORD_B_MARKER}{body_a}"


def orient_input(input_text: str, order: str) -> str:
    if order == "original":
        return input_text
    if order == "swapped":
        return swap_record_order(input_text)
    raise ValueError(f"Unknown order {order!r}; choose one of {ORDERS}")


def record_title(record_text: str) -> str:
    match = re.search(r"^- title: (.*)$", record_text, flags=re.MULTILINE)
    if not match or match.group(1).strip() == "<missing>":
        return ""
    return match.group(1).strip()


def title_token_count(title: str) -> int:
    return len(set(re.findall(r"[a-z0-9]+", title.lower())))


# ---------------------------------------------------------------------------
# Score files
# ---------------------------------------------------------------------------

def load_scores(path: Path) -> list[dict]:
    """Read complete score rows; drop a partial trailing line left by a crash."""
    if not path.is_file():
        return []
    valid: list[dict] = []
    damaged = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                valid.append(json.loads(line))
            except json.JSONDecodeError:
                damaged = True
    if damaged:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in valid:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    return valid


def append_scores(path: Path, records: Iterable[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------
# Model loading and scoring
# ---------------------------------------------------------------------------

def load_scorer(config, model_spec: str):
    """Return (tokenizer, GenerativeRerankerModel, identity) for a model spec.

    ``base`` loads the pretrained reranker. A directory with
    ``adapter_config.json`` loads the base model and merges that LoRA adapter.
    Any other directory is treated as a merged checkpoint.
    """
    from transformers import AutoModelForCausalLM

    from models.generative_reranker_student import (
        GenerativeRerankerModel,
        ensure_reranker_padding_token,
        load_merged_reranker,
        load_reranker_tokenizer,
        resolve_answer_token_ids,
    )

    if model_spec == "base":
        tokenizer = load_reranker_tokenizer(config)
        causal_lm = AutoModelForCausalLM.from_pretrained(
            config.model_name, revision=config.model_revision
        )
        identity = {"kind": "base", "model_name": config.model_name}
    else:
        path = Path(model_spec)
        adapter_config = path / "adapter_config.json"
        if adapter_config.is_file():
            from peft import PeftModel

            declared_base = json.loads(adapter_config.read_text(encoding="utf-8")).get(
                "base_model_name_or_path"
            )
            if declared_base and declared_base != config.model_name:
                raise ValueError(
                    f"Adapter base {declared_base!r} differs from config model "
                    f"{config.model_name!r}"
                )
            tokenizer = load_reranker_tokenizer(config)
            base = AutoModelForCausalLM.from_pretrained(
                config.model_name, revision=config.model_revision
            )
            causal_lm = PeftModel.from_pretrained(base, str(path)).merge_and_unload()
            weights = path / "adapter_model.safetensors"
            identity = {
                "kind": "lora_adapter",
                "path": str(path),
                "adapter_sha256": file_sha256(weights) if weights.is_file() else None,
            }
        else:
            tokenizer, model = load_merged_reranker(config, path)
            weights = path / "model.safetensors"
            identity = {
                "kind": "merged_checkpoint",
                "path": str(path),
                "model_sha256": file_sha256(weights) if weights.is_file() else None,
            }
            return tokenizer, model, identity

    ensure_reranker_padding_token(tokenizer, causal_lm)
    negative_id, positive_id = resolve_answer_token_ids(
        tokenizer,
        str(config.reranker_negative_token),
        str(config.reranker_positive_token),
    )
    model = GenerativeRerankerModel(
        causal_lm, negative_id, positive_id, config.label_to_id
    )
    return tokenizer, model, identity


def score_rows(
    model,
    tokenizer,
    config,
    rows: Sequence[dict],
    order: str,
    batch_size: int,
    device: str,
    precision: str,
    max_input_length: int,
    sink: Callable[[list[dict]], None],
    progress: bool = False,
) -> int:
    """Score rows in one order and pass each finished batch to ``sink``."""
    import torch

    from models.generative_reranker_student import (
        ERGenerativeRerankerDataset,
        RerankerDataCollator,
    )
    from utils.torch_runtime import autocast_context

    if not rows:
        return 0
    oriented = [
        {**row, "input_text": orient_input(str(row["input_text"]), order)}
        for row in rows
    ]
    dataset = ERGenerativeRerankerDataset(
        oriented, tokenizer, config, max_input_length, include_labels=False
    )
    collator = RerankerDataCollator(tokenizer)
    # Length-sorted batches waste less padding; results are keyed by pair ID.
    indices = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["input_ids"]))
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    if progress:
        from tqdm import tqdm

        batches = tqdm(batches, desc=f"Scoring {order}")
    match_index = config.label_to_id["match"]
    non_match_index = config.label_to_id["non-match"]
    written = 0
    with torch.inference_mode(), autocast_context(device, precision):
        for batch_indices in batches:
            batch = collator([dataset[i] for i in batch_indices])
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.float()
            records = []
            for position, row_index in enumerate(batch_indices):
                logit_match = float(logits[position, match_index].item())
                logit_non_match = float(logits[position, non_match_index].item())
                log_odds = logit_match - logit_non_match
                records.append(
                    {
                        "pair_id": rows[row_index]["pair_id"],
                        "order": order,
                        "label": int(rows[row_index]["label"]),
                        "logit_match": logit_match,
                        "logit_non_match": logit_non_match,
                        "log_odds": log_odds,
                        "match_probability": sigmoid(log_odds),
                        "token_count": len(dataset[row_index]["input_ids"]),
                    }
                )
            sink(records)
            written += len(records)
    return written


def run_score(args: argparse.Namespace) -> dict:
    import torch

    from models.student_config import load_student_config
    from utils.torch_runtime import resolve_precision

    config = load_student_config(Path(args.config))
    if config.architecture != "generative_reranker":
        raise ValueError("The order-swap probe supports generative_reranker configs only")
    input_path = Path(args.input)
    rows = list(iter_jsonl(input_path))
    seen_ids: set[str] = set()
    for row in rows:
        if row["pair_id"] in seen_ids:
            raise ValueError(f"Duplicate pair_id in input: {row['pair_id']}")
        seen_ids.add(row["pair_id"])
    if args.pair_ids:
        wanted = {
            line.strip()
            for line in Path(args.pair_ids).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing = wanted - seen_ids
        if missing:
            raise ValueError(f"{len(missing)} requested pair IDs are not in the input")
        rows = [row for row in rows if row["pair_id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / SCORES_FILENAME
    done = {(row["pair_id"], row["order"]) for row in load_scores(scores_path)}

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    precision = resolve_precision(device, args.precision)
    load_started = time.perf_counter()
    tokenizer, model, identity = load_scorer(config, args.model)
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    session = {
        "started_unix": time.time(),
        "device": device,
        "device_name": torch.cuda.get_device_name(device)
        if torch.device(device).type == "cuda"
        else "cpu",
        "precision": precision,
        "batch_size": args.batch_size,
        "model_load_seconds": load_seconds,
        "orders": {},
    }
    for order in args.orders:
        pending = [row for row in rows if (row["pair_id"], order) not in done]
        started = time.perf_counter()
        written = score_rows(
            model,
            tokenizer,
            config,
            pending,
            order,
            args.batch_size,
            device,
            precision,
            args.max_input_length,
            lambda records: append_scores(scores_path, records),
            progress=True,
        )
        seconds = time.perf_counter() - started
        session["orders"][order] = {
            "requested_rows": len(rows),
            "already_done": len(rows) - len(pending),
            "scored_now": written,
            "seconds": seconds,
            "seconds_per_pair": seconds / written if written else None,
        }

    run_path = output_dir / RUN_FILENAME
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    fixed = {
        "probe": "record_order_swap",
        "model": identity,
        "student_config": args.config,
        "reranker_instruction": config.reranker_instruction,
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "pair_ids_file": args.pair_ids,
        "limit": args.limit,
        "max_input_length": args.max_input_length,
        "input_truncation": False,
    }
    for key, value in fixed.items():
        if key in run and run[key] != value:
            raise ValueError(
                f"{run_path} records {key}={run[key]!r}; this run uses {value!r}. "
                "Use a separate --output-dir."
            )
    run.update(fixed)
    import transformers

    session["torch_version"] = torch.__version__
    session["transformers_version"] = transformers.__version__
    session["git_commit"] = git_commit()
    run.setdefault("sessions", []).append(session)
    temporary = run_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(run_path)
    return session


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def fuse(log_odds_original: float, log_odds_swapped: float, fusion: str) -> float:
    if fusion == "mean":
        return (log_odds_original + log_odds_swapped) / 2.0
    if fusion == "min":
        return min(log_odds_original, log_odds_swapped)
    raise ValueError(f"Unknown fusion {fusion!r}; choose one of {FUSIONS}")


def outcome(label: int, decision: bool) -> str:
    if decision:
        return "TP" if label == 1 else "FP"
    return "FN" if label == 1 else "TN"


def describe(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def read_threshold(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["decision_threshold"])


def build_pairs(score_rows_: Sequence[dict], input_rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """Join both orders per pair; report pairs scored in only one order."""
    by_pair: dict[str, dict] = {}
    for row in score_rows_:
        entry = by_pair.setdefault(row["pair_id"], {})
        if row["order"] in entry:
            raise ValueError(f"Duplicate score for {row['pair_id']} ({row['order']})")
        entry[row["order"]] = row
    inputs = {row["pair_id"]: row for row in input_rows}
    pairs: list[dict] = []
    single_order = 0
    for pair_id, entry in by_pair.items():
        if pair_id not in inputs:
            raise ValueError(f"Scored pair {pair_id} is not in the input file")
        if not all(order in entry for order in ORDERS):
            single_order += 1
            continue
        record_a, record_b = split_product_records(str(inputs[pair_id]["input_text"]))
        title_a, title_b = record_title(record_a), record_title(record_b)
        label = int(inputs[pair_id]["label"])
        if entry["original"]["label"] != label:
            raise ValueError(f"Score label for {pair_id} differs from the input label")
        pairs.append(
            {
                "pair_id": pair_id,
                "label": label,
                "log_odds_original": float(entry["original"]["log_odds"]),
                "log_odds_swapped": float(entry["swapped"]["log_odds"]),
                "title_a": title_a,
                "title_b": title_b,
                "title_tokens_a": title_token_count(title_a),
                "title_tokens_b": title_token_count(title_b),
            }
        )
    pairs.sort(key=lambda item: item["pair_id"])
    coverage = {
        "input_pairs": len(inputs),
        "pairs_with_both_orders": len(pairs),
        "pairs_with_one_order": single_order,
    }
    return pairs, coverage


def summarize(
    pairs: Sequence[dict],
    threshold: float,
    saved_predictions: dict[str, dict] | None = None,
    select_fused_thresholds: bool = False,
    fused_thresholds: dict[str, float] | None = None,
) -> dict:
    """Asymmetry, direction, and order-fusion metrics for paired scores."""
    from utils.classification_threshold import select_decision_threshold

    if not pairs:
        raise ValueError("No pairs have scores in both orders")
    labels = [pair["label"] == 1 for pair in pairs]
    probabilities = {
        "original": [sigmoid(p["log_odds_original"]) for p in pairs],
        "swapped": [sigmoid(p["log_odds_swapped"]) for p in pairs],
    }
    for fusion in FUSIONS:
        probabilities[fusion] = [
            sigmoid(fuse(p["log_odds_original"], p["log_odds_swapped"], fusion))
            for p in pairs
        ]

    summary: dict = {
        "pairs": len(pairs),
        "saved_threshold": threshold,
        "notes": [
            "Delta = swapped log-odds minus original log-odds.",
            "Groups use the original-order outcome at the saved threshold.",
            "Direction delta = log-odds with the vaguer title as Query minus "
            "log-odds with the more specific title as Query; vaguer means fewer "
            "distinct title tokens, and ties are excluded. A relevance prior "
            "predicts positive values.",
        ],
    }

    if saved_predictions is not None:
        missing = [p["pair_id"] for p in pairs if p["pair_id"] not in saved_predictions]
        if missing:
            raise ValueError(f"{len(missing)} scored pairs lack saved predictions")
        differences = [
            abs(prob - float(saved_predictions[p["pair_id"]]["match_probability"]))
            for p, prob in zip(pairs, probabilities["original"])
        ]
        agreement = sum(
            (prob >= threshold) == bool(saved_predictions[p["pair_id"]]["prediction"])
            for p, prob in zip(pairs, probabilities["original"])
        )
        summary["reproduction_check"] = {
            "abs_probability_difference": describe(differences),
            "decision_agreement": agreement,
            "decision_disagreement": len(pairs) - agreement,
        }

    summary["metrics_at_saved_threshold"] = {
        name: compute_metrics([p >= threshold for p in values], labels)
        for name, values in probabilities.items()
    }

    if select_fused_thresholds:
        selected = {
            name: select_decision_threshold(probabilities[name], labels)
            for name in ("original", *FUSIONS)
        }
        summary["selected_thresholds"] = {
            name: {
                "decision_threshold": payload["decision_threshold"],
                "metrics": payload["validation_metrics"],
            }
            for name, payload in selected.items()
        }
        summary["fused_thresholds"] = {
            name: selected[name]["decision_threshold"] for name in FUSIONS
        }
        summary["notes"].append(
            "Selected thresholds are chosen and scored on the same split; "
            "their metrics are optimistic and need confirmation on held-out data."
        )
    if fused_thresholds is not None:
        summary["metrics_at_supplied_fused_thresholds"] = {
            name: {
                "decision_threshold": fused_thresholds[name],
                "metrics": compute_metrics(
                    [p >= fused_thresholds[name] for p in probabilities[name]], labels
                ),
            }
            for name in FUSIONS
        }

    groups: dict[str, dict] = {}
    for pair, prob_original, prob_swapped in zip(
        pairs, probabilities["original"], probabilities["swapped"]
    ):
        delta = pair["log_odds_swapped"] - pair["log_odds_original"]
        decision_original = prob_original >= threshold
        decision_swapped = prob_swapped >= threshold
        pair["delta"] = delta
        pair["decision_original"] = int(decision_original)
        pair["decision_swapped"] = int(decision_swapped)
        pair["group"] = outcome(pair["label"], decision_original)
        if pair["title_tokens_a"] == pair["title_tokens_b"]:
            pair["direction_delta"] = None
        elif pair["title_tokens_a"] < pair["title_tokens_b"]:
            pair["direction_delta"] = -delta
        else:
            pair["direction_delta"] = delta
        bucket = groups.setdefault(
            pair["group"], {"abs_delta": [], "delta": [], "flips": 0, "direction": []}
        )
        bucket["abs_delta"].append(abs(delta))
        bucket["delta"].append(delta)
        bucket["flips"] += int(decision_original != decision_swapped)
        if pair["direction_delta"] is not None:
            bucket["direction"].append(pair["direction_delta"])

    summary["groups"] = {}
    for name in ("TP", "FP", "TN", "FN"):
        bucket = groups.get(name)
        if bucket is None:
            summary["groups"][name] = {"n": 0}
            continue
        direction = bucket["direction"]
        summary["groups"][name] = {
            "n": len(bucket["delta"]),
            "abs_delta": describe(bucket["abs_delta"]),
            "signed_delta": describe(bucket["delta"]),
            "decision_flips": bucket["flips"],
            "direction_delta": {
                **describe(direction),
                "share_positive": (
                    sum(value > 0 for value in direction) / len(direction)
                    if direction
                    else None
                ),
            },
        }
    return summary


def write_pairs_csv(path: Path, pairs: Sequence[dict]) -> None:
    import csv

    fields = [
        "pair_id", "label", "group", "log_odds_original", "log_odds_swapped",
        "delta", "decision_original", "decision_swapped", "direction_delta",
        "title_tokens_a", "title_tokens_b", "title_a", "title_b",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)


def run_summarize(args: argparse.Namespace) -> dict:
    if args.select_fused_thresholds and args.fused_thresholds_json:
        raise ValueError("Use either --select-fused-thresholds or --fused-thresholds-json")
    score_rows_ = load_scores(Path(args.scores))
    input_rows = list(iter_jsonl(Path(args.input)))
    pairs, coverage = build_pairs(score_rows_, input_rows)
    if coverage["pairs_with_both_orders"] < coverage["input_pairs"] and not args.allow_partial:
        raise ValueError(
            f"Only {coverage['pairs_with_both_orders']}/{coverage['input_pairs']} pairs "
            "have both orders; pass --allow-partial for a subset summary"
        )
    saved = None
    if args.saved_predictions:
        saved = {row["pair_id"]: row for row in iter_jsonl(Path(args.saved_predictions))}
    fused_thresholds = None
    if args.fused_thresholds_json:
        payload = json.loads(Path(args.fused_thresholds_json).read_text(encoding="utf-8"))
        fused_thresholds = {name: float(payload["fused_thresholds"][name]) for name in FUSIONS}
    if (args.threshold is None) == (args.threshold_json is None):
        raise ValueError("Pass exactly one of --threshold or --threshold-json")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else read_threshold(Path(args.threshold_json))
    )
    summary = summarize(
        pairs,
        threshold,
        saved_predictions=saved,
        select_fused_thresholds=args.select_fused_thresholds,
        fused_thresholds=fused_thresholds,
    )
    summary["coverage"] = coverage
    summary["inputs"] = {
        "scores": args.scores,
        "input": args.input,
        "threshold": args.threshold,
        "threshold_json": args.threshold_json,
        "saved_predictions": args.saved_predictions,
        "fused_thresholds_json": args.fused_thresholds_json,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.pairs_csv:
        write_pairs_csv(Path(args.pairs_csv), pairs)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score pairs in original and swapped order")
    score.add_argument("--config", default="configs/students/qwen3_reranker_0_6b.json")
    score.add_argument(
        "--model",
        required=True,
        help="'base', a LoRA adapter directory, or a merged checkpoint directory",
    )
    score.add_argument("--input", required=True, help="Serialized pair JSONL")
    score.add_argument("--output-dir", required=True)
    score.add_argument("--orders", nargs="+", choices=ORDERS, default=list(ORDERS))
    score.add_argument("--pair-ids", help="Optional file with one pair ID per line")
    score.add_argument("--limit", type=int, help="Score only the first N pairs (timing)")
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--max-input-length", type=int, default=4096)
    score.add_argument("--device")
    score.add_argument("--precision", default="auto")

    summ = sub.add_parser("summarize", help="Summarize asymmetry and fused metrics")
    summ.add_argument("--scores", required=True)
    summ.add_argument("--input", required=True, help="Same serialized JSONL as scoring")
    summ.add_argument(
        "--threshold-json",
        help="decision_threshold.json of the scored run",
    )
    summ.add_argument(
        "--threshold",
        type=float,
        help="Explicit match-probability threshold (e.g. 0.5 for the base model)",
    )
    summ.add_argument("--saved-predictions", help="Saved predictions for a reproduction check")
    summ.add_argument(
        "--select-fused-thresholds",
        action="store_true",
        help="Select fused-score thresholds on this split (validation only)",
    )
    summ.add_argument(
        "--fused-thresholds-json",
        help="Validation summary.json whose fused thresholds are applied here",
    )
    summ.add_argument("--allow-partial", action="store_true")
    summ.add_argument("--output", required=True)
    summ.add_argument("--pairs-csv")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "score":
        result = run_score(args)
    else:
        result = run_summarize(args)
        result = {key: result[key] for key in ("coverage", "metrics_at_saved_threshold", "groups")}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
