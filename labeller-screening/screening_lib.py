from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from supervision.llm_providers import OpenRouterHTTPError, OpenRouterTransportError


VALID_LABELS = {"match", "non_match"}
BLINDED_KEYS = ("pair_id", "input_text")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class NonRetryableResponseError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(row) + "\n" for row in rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _normalize_gold(value: Any) -> str:
    if value is True or value == 1 or value == "1" or value == "match":
        return "match"
    if value is False or value == 0 or value == "0" or value in {"non_match", "non-match"}:
        return "non_match"
    raise ValueError(f"Unsupported gold label: {value!r}")


def prepare_sample(source: Path, output_dir: Path, count: int = 300, seed: int = 42) -> dict[str, Any]:
    rows = load_jsonl(source)
    if len(rows) < count:
        raise ValueError(f"Source has {len(rows)} rows, fewer than requested {count}")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id")
        input_text = row.get("input_text")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Row {index} has an invalid pair_id")
        if pair_id in seen:
            raise ValueError(f"Duplicate pair_id in source: {pair_id}")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"Row {index} has invalid input_text")
        if row.get("split") != "train":
            raise ValueError(f"Non-training row found: {pair_id}")
        gold = _normalize_gold(row.get("label"))
        seen.add(pair_id)
        validated.append({"pair_id": pair_id, "input_text": input_text, "gold_label": gold})

    selected = random.Random(seed).sample(validated, count)
    selected.sort(key=lambda item: item["pair_id"])
    blinded = [{"pair_id": row["pair_id"], "input_text": row["input_text"]} for row in selected]
    gold_rows = [(row["pair_id"], row["gold_label"]) for row in selected]

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "wdc_300.inputs.jsonl"
    gold_path = output_dir / "wdc_300.gold.csv"
    manifest_path = output_dir / "wdc_300.manifest.json"

    inputs_text = _jsonl_text(blinded)
    csv_lines = ["pair_id,gold_label\n"]
    csv_lines.extend(f"{pair_id},{label}\n" for pair_id, label in gold_rows)
    gold_text = "".join(csv_lines)

    inputs_hash = hashlib.sha256(inputs_text.encode("utf-8")).hexdigest()
    gold_hash = hashlib.sha256(gold_text.encode("utf-8")).hexdigest()
    ids_hash = hashlib.sha256("\n".join(row["pair_id"] for row in selected).encode("utf-8")).hexdigest()
    class_counts = {
        "match": sum(row["gold_label"] == "match" for row in selected),
        "non_match": sum(row["gold_label"] == "non_match" for row in selected),
    }
    manifest = {
        "schema_version": 1,
        "dataset": "wdc-products",
        "split": "train",
        "sampling": "uniform_without_replacement",
        "seed": seed,
        "count": count,
        "class_counts": class_counts,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "inputs_sha256": inputs_hash,
        "gold_sha256": gold_hash,
        "sampled_ids_sha256": ids_hash,
        "blinded_fields": list(BLINDED_KEYS),
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    desired = {inputs_path: inputs_text, gold_path: gold_text, manifest_path: manifest_text}
    existing = [path for path in desired if path.exists()]
    if existing:
        mismatches = [path for path, text in desired.items() if not path.exists() or path.read_text(encoding="utf-8") != text]
        if mismatches:
            names = ", ".join(path.name for path in mismatches)
            raise FileExistsError(f"Refusing to overwrite a different frozen sample: {names}")
        return manifest

    for path, text in desired.items():
        _atomic_text(path, text)
    return manifest


def prepare_full_training_inputs(source: Path, output_dir: Path) -> dict[str, Any]:
    """Freeze every source training pair into a gold-free model input file."""
    rows = load_jsonl(source)
    seen: set[str] = set()
    blinded: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id")
        input_text = row.get("input_text")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"Row {index} has an invalid pair_id")
        if pair_id in seen:
            raise ValueError(f"Duplicate pair_id in source: {pair_id}")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"Row {index} has invalid input_text")
        if row.get("split") != "train":
            raise ValueError(f"Non-training row found: {pair_id}")
        seen.add(pair_id)
        blinded.append({"pair_id": pair_id, "input_text": input_text})

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "wdc_train_full.inputs.jsonl"
    manifest_path = output_dir / "wdc_train_full.manifest.json"
    inputs_text = _jsonl_text(blinded)
    manifest = {
        "schema_version": 1,
        "dataset": "wdc-products",
        "split": "train",
        "selection": "complete_official_split",
        "count": len(blinded),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "inputs_sha256": hashlib.sha256(inputs_text.encode("utf-8")).hexdigest(),
        "ids_sha256": hashlib.sha256(
            "\n".join(row["pair_id"] for row in blinded).encode("utf-8")
        ).hexdigest(),
        "blinded_fields": list(BLINDED_KEYS),
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    desired = {inputs_path: inputs_text, manifest_path: manifest_text}
    if any(path.exists() for path in desired):
        mismatches = [
            path for path, text in desired.items()
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        if mismatches:
            names = ", ".join(path.name for path in mismatches)
            raise FileExistsError(f"Refusing to overwrite different frozen full inputs: {names}")
        return manifest
    for path, text in desired.items():
        _atomic_text(path, text)
    return manifest


def load_settings(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("provider") != "openrouter":
        raise ValueError("Screening config must use OpenRouter")
    if config.get("api_url") != OPENROUTER_BASE_URL:
        raise ValueError(f"API URL must be pinned to {OPENROUTER_BASE_URL}")
    settings = config.get("settings")
    if not isinstance(settings, dict) or set(settings) != {"sol_high", "sol_max", "sol_pro_max"}:
        raise ValueError("Config must define exactly sol_high, sol_max, and sol_pro_max")
    expected_settings = {
        "sol_high": {
            "model": "openai/gpt-5.6-sol",
            "reasoning": {"effort": "high", "exclude": True},
        },
        "sol_max": {
            "model": "openai/gpt-5.6-sol",
            "reasoning": {"effort": "max", "exclude": True},
        },
        "sol_pro_max": {
            "model": "openai/gpt-5.6-sol-pro",
            "reasoning": {"effort": "max", "exclude": True},
        },
    }
    for name, expected in expected_settings.items():
        if settings[name] != expected:
            raise ValueError(f"{name} must use its frozen OpenRouter model and reasoning configuration")
    expected_routing = {
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "max_price": {"prompt": 2.0, "completion": 10.0},
    }
    if config.get("provider_routing") != expected_routing:
        raise ValueError("OpenRouter routing must pin OpenAI, disable fallbacks, and enforce parameters/prices")
    pricing = config.get("pricing_snapshot")
    for key in ("input_usd_per_million_tokens", "output_usd_per_million_tokens"):
        value = pricing.get(key) if isinstance(pricing, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"pricing_snapshot.{key} must be positive and finite")
    for key in ("max_attempts", "max_output_tokens", "request_timeout_seconds"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return config


def build_request_payload(config: Mapping[str, Any], setting: str, row: Mapping[str, str]) -> dict[str, Any]:
    if set(row) != set(BLINDED_KEYS):
        raise ValueError(f"Blinded input must contain exactly {BLINDED_KEYS}")
    if setting not in config["settings"]:
        raise ValueError(f"Unknown setting: {setting}")
    setting_config = config["settings"][setting]
    return {
        "model": setting_config["model"],
        "messages": [
            {"role": "system", "content": config["instructions"]},
            {"role": "user", "content": row["input_text"]},
        ],
        "reasoning": setting_config["reasoning"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "entity_match_label",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"label": {"type": "string", "enum": ["match", "non_match"]}},
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
        },
        "provider": config["provider_routing"],
        "max_tokens": config["max_output_tokens"],
        "stream": False,
    }


def _response_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("OpenRouter response must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter response contains no output text")
    return content


def parse_response(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("OpenRouter response must contain exactly one choice")
    message = choices[0].get("message")
    if isinstance(message, dict) and message.get("refusal"):
        raise NonRetryableResponseError("Model refused the classification request")
    finish_reason = choices[0].get("finish_reason")
    if finish_reason != "stop":
        raise NonRetryableResponseError(f"OpenRouter response did not finish normally: {finish_reason!r}")
    parsed = json.loads(_response_text(response))
    if not isinstance(parsed, dict) or set(parsed) != {"label"} or parsed["label"] not in VALID_LABELS:
        raise ValueError("Response violates the answer-only label schema")
    return parsed["label"]


def validate_blinded_inputs(
    path: Path,
    expected_count: int | None = 300,
) -> list[dict[str, str]]:
    rows = load_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} blinded rows, found {len(rows)}")
    ids: set[str] = set()
    for row in rows:
        if set(row) != set(BLINDED_KEYS):
            raise ValueError(f"Blinded row {row.get('pair_id')} has forbidden or missing fields")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id or pair_id in ids:
            raise ValueError(f"Missing or duplicate blinded pair ID: {pair_id!r}")
        if not isinstance(row["input_text"], str) or not row["input_text"].strip():
            raise ValueError(f"Invalid input_text for {pair_id}")
        ids.add(pair_id)
    return rows  # type: ignore[return-value]


def _append_journal(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_attempt_journal(path: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if not path.exists():
        return {}, []
    predictions: dict[str, str] = {}
    rows = load_jsonl(path)
    for row in rows:
        pair_id, result, status = row.get("pair_id"), row.get("result"), row.get("status")
        if not isinstance(pair_id, str) or status not in {"valid", "invalid", "error"}:
            raise ValueError(f"Invalid attempt journal row in {path}")
        if status == "valid":
            if result not in VALID_LABELS or pair_id in predictions:
                raise ValueError(f"Invalid or duplicate completed prediction in {path}")
            predictions[pair_id] = result
    return predictions, rows


def validate_reuse_artifacts(
    full_rows: list[dict[str, str]],
    config: Mapping[str, Any],
    setting: str,
    reuse_attempts_path: Path,
    reuse_inputs_path: Path,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, str]]:
    """Validate that completed attempts are exact reusable requests for this run."""
    reuse_run_path = reuse_attempts_path.with_name(
        reuse_attempts_path.name.replace(".attempts.jsonl", ".run.json")
    )
    if not reuse_run_path.exists():
        raise ValueError(f"Reuse run manifest is required: {reuse_run_path}")
    reuse_rows = validate_blinded_inputs(reuse_inputs_path, expected_count=None)
    reuse_by_id = {row["pair_id"]: row for row in reuse_rows}
    full_by_id = {row["pair_id"]: row for row in full_rows}
    if not set(reuse_by_id).issubset(full_by_id):
        raise ValueError("Reuse inputs contain IDs outside the full training input")
    for pair_id, reuse_row in reuse_by_id.items():
        if reuse_row != full_by_id[pair_id]:
            raise ValueError(f"Reuse input differs from full input for {pair_id}")

    reuse_run = json.loads(reuse_run_path.read_text(encoding="utf-8"))
    expected_payload_hash = hashlib.sha256(
        _canonical_json([
            build_request_payload(config, setting, row) for row in reuse_rows
        ]).encode("utf-8")
    ).hexdigest()
    if (
        reuse_run.get("setting") != setting
        or reuse_run.get("model") != config["settings"][setting]["model"]
        or reuse_run.get("prompt_version") != config["prompt_version"]
        or reuse_run.get("inputs_sha256") != sha256_file(reuse_inputs_path)
        or reuse_run.get("request_payloads_sha256") != expected_payload_hash
    ):
        raise ValueError("Reuse artifacts do not match the frozen labeler configuration")

    predictions, attempts = _read_attempt_journal(reuse_attempts_path)
    if set(predictions) != set(reuse_by_id):
        raise ValueError("Reuse journal does not provide complete valid coverage of reuse inputs")
    expected_model = config["settings"][setting]["model"]
    for attempt_row in attempts:
        if attempt_row.get("status") == "valid" and (
            attempt_row.get("requested_model") != expected_model
            or attempt_row.get("returned_model") != expected_model
        ):
            raise ValueError("Reuse journal contains a valid result from the wrong model")
    provenance = {
        "attempts_path": str(reuse_attempts_path),
        "attempts_sha256": sha256_file(reuse_attempts_path),
        "inputs_path": str(reuse_inputs_path),
        "inputs_sha256": sha256_file(reuse_inputs_path),
        "run_path": str(reuse_run_path),
        "run_sha256": sha256_file(reuse_run_path),
    }
    return predictions, attempts, provenance


def _observed_cost(rows: Iterable[Mapping[str, Any]], pricing: Mapping[str, Any]) -> float:
    return sum(_accounted_attempt_cost(row, pricing) for row in rows)


def _accounted_attempt_cost(row: Mapping[str, Any], pricing: Mapping[str, Any]) -> float:
    usage = row.get("usage", {})
    provider_cost = _provider_usage_cost(usage)
    if provider_cost is not None:
        return provider_cost
    if _has_complete_usage(usage):
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        return (
            input_tokens * float(pricing["input_usd_per_million_tokens"])
            + output_tokens * float(pricing["output_usd_per_million_tokens"])
        ) / 1_000_000
    if row.get("reserved_cost_usd") is not None:
        return float(row["reserved_cost_usd"])
    return 0.0


def _largest_observed_attempt_cost(
    rows: Iterable[Mapping[str, Any]],
    pricing: Mapping[str, Any],
) -> float:
    return max((_accounted_attempt_cost(row, pricing) for row in rows), default=0.0)


def _has_complete_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    for primary, fallback in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        value = usage.get(primary, usage.get(fallback))
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            return False
    return True


def _provider_usage_cost(usage: Any) -> float | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get("cost")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _has_billable_usage(usage: Any) -> bool:
    return _provider_usage_cost(usage) is not None or _has_complete_usage(usage)


def _reserved_request_cost(payload: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    pricing = config["pricing_snapshot"]
    conservative_input_tokens = len(_canonical_json(payload).encode("utf-8"))
    return (
        conservative_input_tokens * float(pricing["input_usd_per_million_tokens"])
        + int(config["max_output_tokens"]) * float(pricing["output_usd_per_million_tokens"])
    ) / 1_000_000


def run_setting(
    inputs_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    setting: str,
    client: Any,
    spend_ceiling_usd: float,
    sleep: Any = time.sleep,
    expected_count: int | None = 300,
    reuse_attempts_path: Path | None = None,
    reuse_inputs_path: Path | None = None,
    run_provenance: Mapping[str, Any] | None = None,
) -> Path:
    if not isinstance(spend_ceiling_usd, (int, float)) or isinstance(spend_ceiling_usd, bool) or not math.isfinite(spend_ceiling_usd) or spend_ceiling_usd <= 0:
        raise ValueError("spend_ceiling_usd must be a positive number")
    rows = validate_blinded_inputs(inputs_path, expected_count=expected_count)
    ids = [row["pair_id"] for row in rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / f"{setting}.attempts.jsonl"
    audit_path = output_dir / f"{setting}.audit.jsonl"
    run_manifest_path = output_dir / f"{setting}.run.json"
    final_path = output_dir / f"{setting}.csv"
    reuse_provenance = None
    if reuse_attempts_path is not None or reuse_inputs_path is not None:
        if reuse_attempts_path is None or reuse_inputs_path is None:
            raise ValueError("Both reuse_attempts_path and reuse_inputs_path are required")
        _, _, reuse_provenance = validate_reuse_artifacts(
            rows, config, setting, reuse_attempts_path, reuse_inputs_path
        )
    run_manifest = {
        "schema_version": 1,
        "setting": setting,
        "model": config["settings"][setting]["model"],
        "prompt_version": config["prompt_version"],
        "inputs_sha256": sha256_file(inputs_path),
        "request_payloads_sha256": hashlib.sha256(_canonical_json([
            build_request_payload(config, setting, row) for row in rows
        ]).encode("utf-8")).hexdigest(),
        "api_url": config["api_url"],
        "max_attempts": config["max_attempts"],
        "runner_code_sha256": sha256_file(Path(__file__)),
        "provider_client_code_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "supervision/llm_providers.py"
        ),
        "spend_ceiling_usd": float(spend_ceiling_usd),
        "pricing_snapshot": config["pricing_snapshot"],
        "reused_attempts": reuse_provenance,
        "run_provenance": dict(run_provenance or {}),
    }
    run_manifest_text = json.dumps(run_manifest, indent=2, sort_keys=True) + "\n"
    if run_manifest_path.exists() and run_manifest_path.read_text(encoding="utf-8") != run_manifest_text:
        raise ValueError("Refusing to resume: frozen input, config, pricing, or spend ceiling changed")
    if journal_path.exists() and not run_manifest_path.exists():
        raise ValueError("Refusing to resume an attempt journal without its frozen run manifest")
    if not run_manifest_path.exists():
        _atomic_text(run_manifest_path, run_manifest_text)
    if reuse_provenance is not None and not journal_path.exists():
        _, reused_attempts, _ = validate_reuse_artifacts(
            rows, config, setting, reuse_attempts_path, reuse_inputs_path
        )
        full_by_id = {row["pair_id"]: row for row in rows}
        enriched_attempts = []
        for attempt_row in reused_attempts:
            payload = build_request_payload(config, setting, full_by_id[attempt_row["pair_id"]])
            payload_sha256 = hashlib.sha256(
                _canonical_json(payload).encode("utf-8")
            ).hexdigest()
            request_identity_sha256 = hashlib.sha256(
                _canonical_json({
                    "pair_id": attempt_row["pair_id"],
                    "setting": setting,
                    "attempt": attempt_row["attempt"],
                    "payload_sha256": payload_sha256,
                }).encode("utf-8")
            ).hexdigest()
            enriched_attempts.append({
                **attempt_row,
                "request_payload_sha256": payload_sha256,
                "request_identity_sha256": request_identity_sha256,
                "provenance_schema": "screening_v1_grandfathered",
                "raw_response_unavailable_reason": (
                    "The completed screening-v1 attempt predates raw-response retention."
                ),
            })
        _atomic_text(journal_path, _jsonl_text(enriched_attempts))
    completed, attempts = _read_attempt_journal(journal_path)
    id_set = set(ids)
    rows_by_id = {row["pair_id"]: row for row in rows}
    attempt_counts: dict[str, int] = {}
    for journal_row in attempts:
        pair_id = journal_row["pair_id"]
        if pair_id not in id_set or journal_row.get("setting") != setting:
            raise ValueError("Attempt journal contains an out-of-sample ID or wrong setting")
        expected_attempt = attempt_counts.get(pair_id, 0) + 1
        if journal_row.get("attempt") != expected_attempt:
            raise ValueError(f"Non-sequential attempt number for {pair_id}")
        payload = build_request_payload(config, setting, rows_by_id[pair_id])
        expected_payload_sha256 = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        expected_request_identity_sha256 = hashlib.sha256(
            _canonical_json({
                "pair_id": pair_id,
                "setting": setting,
                "attempt": expected_attempt,
                "payload_sha256": expected_payload_sha256,
            }).encode("utf-8")
        ).hexdigest()
        if (
            journal_row.get("request_payload_sha256") != expected_payload_sha256
            or journal_row.get("request_identity_sha256")
            != expected_request_identity_sha256
        ):
            raise ValueError(f"Attempt journal request identity mismatch for {pair_id}")
        if journal_row.get("status") == "valid" and (
            journal_row.get("requested_model") != config["settings"][setting]["model"]
            or journal_row.get("returned_model") != config["settings"][setting]["model"]
        ):
            raise ValueError(f"Attempt journal model identity mismatch for {pair_id}")
        attempt_counts[pair_id] = expected_attempt

    max_attempts = int(config["max_attempts"])
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in completed:
            continue
        last_error: Exception | None = None
        first_attempt = attempt_counts.get(pair_id, 0) + 1
        if first_attempt > max_attempts:
            raise RuntimeError(f"Cumulative retry cap already reached for {pair_id}")
        for attempt in range(first_attempt, max_attempts + 1):
            payload = build_request_payload(config, setting, row)
            payload_sha256 = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
            request_identity_sha256 = hashlib.sha256(
                _canonical_json({
                    "pair_id": pair_id,
                    "setting": setting,
                    "attempt": attempt,
                    "payload_sha256": payload_sha256,
                }).encode("utf-8")
            ).hexdigest()
            reserved_cost = max(
                _reserved_request_cost(payload, config),
                _largest_observed_attempt_cost(attempts, config["pricing_snapshot"]),
            )
            accounted_cost = _observed_cost(attempts, config["pricing_snapshot"])
            if accounted_cost + reserved_cost > spend_ceiling_usd:
                raise RuntimeError(f"Next request could exceed the ${spend_ceiling_usd:.2f} setting ceiling")
            started = time.time()
            response: Mapping[str, Any] | None = None
            try:
                response = client.create(payload)
                if response.get("model") != config["settings"][setting]["model"]:
                    raise NonRetryableResponseError(
                        "OpenRouter returned a model different from the frozen requested model"
                    )
                result = parse_response(response)
                usage = response.get("usage", {}) if isinstance(response, dict) else {}
                journal_row = {
                    "pair_id": pair_id,
                    "setting": setting,
                    "requested_model": config["settings"][setting]["model"],
                    "returned_model": response.get("model"),
                    "response_id": response.get("id"),
                    "provider": "openrouter",
                    "attempt": attempt,
                    "status": "valid",
                    "result": result,
                    "raw_response": response,
                    "request_payload_sha256": payload_sha256,
                    "request_identity_sha256": request_identity_sha256,
                    "usage": usage,
                    "latency_seconds": time.time() - started,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                }
                if not _has_billable_usage(usage):
                    journal_row["reserved_cost_usd"] = reserved_cost
                _append_journal(journal_path, journal_row)
                attempts.append(journal_row)
                completed[pair_id] = result
                break
            except (OpenRouterHTTPError, OpenRouterTransportError, ValueError) as error:
                last_error = error
                retryable = not isinstance(error, (OpenRouterHTTPError, NonRetryableResponseError)) or (
                    isinstance(error, OpenRouterHTTPError) and error.retryable
                )
                journal_row = {
                    "pair_id": pair_id,
                    "setting": setting,
                    "attempt": attempt,
                    "status": "invalid" if isinstance(error, ValueError) else "error",
                    "error_type": type(error).__name__,
                    "error": str(error)[:1000],
                    "latency_seconds": time.time() - started,
                    "provider": "openrouter",
                    "request_payload_sha256": payload_sha256,
                    "request_identity_sha256": request_identity_sha256,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                if response is not None:
                    usage = response.get("usage", {})
                    journal_row.update({
                        "response_id": response.get("id"),
                        "returned_model": response.get("model"),
                        "finish_reason": (
                            response.get("choices", [{}])[0].get("finish_reason")
                            if isinstance(response.get("choices"), list) and response.get("choices")
                            else None
                        ),
                        "usage": usage,
                        "raw_response": response,
                    })
                    if not _has_billable_usage(usage):
                        journal_row["reserved_cost_usd"] = reserved_cost
                else:
                    journal_row["reserved_cost_usd"] = reserved_cost
                _append_journal(journal_path, journal_row)
                attempts.append(journal_row)
                if not retryable:
                    raise RuntimeError(f"Non-retryable failure for {pair_id}: {error}") from error
                if attempt < max_attempts:
                    retry_after = (
                        error.retry_after_seconds
                        if isinstance(error, OpenRouterHTTPError)
                        else None
                    )
                    sleep(min(retry_after, 60.0) if retry_after is not None else min(2 ** (attempt - 1), 4))
        else:
            raise RuntimeError(f"Failed {pair_id} after {max_attempts} attempts: {last_error}") from last_error

    if set(completed) != set(ids):
        raise RuntimeError("Run ended without complete sample coverage")
    lines = ["pair_id,result\n"]
    lines.extend(f"{row['pair_id']},{completed[row['pair_id']]}\n" for row in rows)
    _atomic_text(final_path, "".join(lines))
    audit_rows = [{key: value for key, value in row.items() if key != "result"} for row in attempts]
    _atomic_text(audit_path, _jsonl_text(audit_rows))
    return final_path


def load_label_csv(path: Path, label_column: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["pair_id", label_column]:
            raise ValueError(f"{path} must have exactly pair_id,{label_column}")
        for row in reader:
            pair_id, label = row["pair_id"], row[label_column]
            if not pair_id or pair_id in values:
                raise ValueError(f"Missing or duplicate pair_id in {path}: {pair_id!r}")
            if label not in VALID_LABELS:
                raise ValueError(f"Invalid label in {path}: {label!r}")
            values[pair_id] = label
    return values


def compute_metrics(gold: Mapping[str, str], predicted: Mapping[str, str]) -> dict[str, Any]:
    if set(gold) != set(predicted):
        missing = sorted(set(gold) - set(predicted))
        extra = sorted(set(predicted) - set(gold))
        raise ValueError(f"Prediction IDs do not match gold; missing={missing[:5]}, extra={extra[:5]}")
    tp = sum(gold[key] == "match" and predicted[key] == "match" for key in gold)
    fp = sum(gold[key] == "non_match" and predicted[key] == "match" for key in gold)
    tn = sum(gold[key] == "non_match" and predicted[key] == "non_match" for key in gold)
    fn = sum(gold[key] == "match" and predicted[key] == "non_match" for key in gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    match_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    negative_f1 = 2 * negative_precision * negative_recall / (negative_precision + negative_recall) if negative_precision + negative_recall else 0.0
    return {
        "count": len(gold),
        "match_precision": precision,
        "match_recall": recall,
        "match_f1": match_f1,
        "macro_f1": (match_f1 + negative_f1) / 2,
        "accuracy": (tp + tn) / len(gold) if gold else 0.0,
        "invalid_rate": 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compare_all(gold_path: Path, predictions_dir: Path, output_dir: Path, settings: Iterable[str]) -> dict[str, Any]:
    gold = load_label_csv(gold_path, "gold_label")
    if len(gold) != 300:
        raise ValueError(f"Expected exactly 300 gold rows, found {len(gold)}")
    setting_names = list(settings)
    predictions: dict[str, dict[str, str]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for setting in setting_names:
        values = load_label_csv(predictions_dir / f"{setting}.csv", "result")
        metrics[setting] = compute_metrics(gold, values)
        predictions[setting] = values
    disagreements: dict[str, int] = {}
    for left_index, left in enumerate(setting_names):
        for right in setting_names[left_index + 1 :]:
            disagreements[f"{left}__vs__{right}"] = sum(predictions[left][key] != predictions[right][key] for key in gold)
    ranking = sorted(setting_names, key=lambda name: (-metrics[name]["match_f1"], -metrics[name]["match_recall"], name))
    report = {
        "schema_version": 1,
        "sample_count": len(gold),
        "primary_metric": "match_f1",
        "metrics": metrics,
        "ranking": ranking,
        "pairwise_disagreements": disagreements,
        "limitations": [
            "One deterministic random sample of 300 WDC training pairs.",
            "Natural class prevalence; this is not a balanced screening sample.",
            "Preliminary labeler calibration evidence, not a final-test thesis result.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "comparison.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    columns = ["setting", "count", "match_precision", "match_recall", "match_f1", "macro_f1", "accuracy", "invalid_rate", "tp", "fp", "tn", "fn"]
    lines = [",".join(columns) + "\n"]
    for setting in setting_names:
        row = {"setting": setting, **metrics[setting]}
        lines.append(",".join(str(row[column]) for column in columns) + "\n")
    _atomic_text(output_dir / "comparison.csv", "".join(lines))
    return report
