"""Offline-first full-training labeling runner for frozen DBLP-ACM inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from data.dataset_profiles import load_dataset_profile
from supervision.full_label_protocol import (
    FullLabelConfig,
    build_label_request,
    load_full_label_config,
    parse_label_response,
)
from supervision.prepare_full_label_inputs import prepare_blinded_inputs, sha256_file


class JSONSchemaClient(Protocol):
    offline: bool

    def create(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class FakeJSONSchemaClient:
    """Deterministic zero-network client used only for integration verification."""

    offline = True

    def __init__(self, *, fail_after: int | None = None, malformed_after: int | None = None):
        self.fail_after = fail_after
        self.malformed_after = malformed_after
        self.payloads: list[dict[str, Any]] = []

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        dispatch_index = len(self.payloads)
        self.payloads.append(payload)
        if self.fail_after is not None and dispatch_index >= self.fail_after:
            raise RuntimeError("simulated dispatch crash")
        input_text = payload["messages"][1]["content"]
        label = "match" if int(hashlib.sha256(input_text.encode("utf-8")).hexdigest(), 16) % 2 else "non_match"
        if self.malformed_after is not None and dispatch_index >= self.malformed_after:
            content = "malformed"
        else:
            content = json.dumps({"label": label})
        return {
            "id": "offline-" + hashlib.sha256(input_text.encode("utf-8")).hexdigest()[:16],
            "model": payload["model"],
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
        }


@dataclass(frozen=True)
class FullLabelArtifacts:
    root: Path
    predictions: Path
    attempts: Path
    audit: Path
    run: Path
    inputs: Path
    input_manifest: Path
    settings: Path
    inflight: Path
    completion: Path

    def as_publisher_kwargs(self) -> dict[str, Path]:
        return {
            "predictions_path": self.predictions,
            "attempts_path": self.attempts,
            "audit_path": self.audit,
            "labeler_run_path": self.run,
            "blinded_inputs_path": self.inputs,
            "blinded_inputs_manifest_path": self.input_manifest,
            "labeler_settings_path": self.settings,
        }


def _artifacts(root: Path) -> FullLabelArtifacts:
    return FullLabelArtifacts(
        root=root,
        predictions=root / "predictions.csv",
        attempts=root / "attempts.jsonl",
        audit=root / "audit.jsonl",
        run=root / "run.json",
        inputs=root / "inputs.jsonl",
        input_manifest=root / "inputs.manifest.json",
        settings=root / "settings.json",
        inflight=root / "inflight.jsonl",
        completion=root / "completion.json",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _write_fsynced(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _append_fsynced(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor) if path.is_absolute() else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_output(output: Path, workspace: Path, version: str) -> Path:
    if ".." in output.parts:
        raise ValueError("output traversal is not allowed")
    absolute = output if output.is_absolute() else workspace / output
    if _has_symlink_component(absolute):
        raise ValueError("output symlink aliases are not allowed")
    resolved = absolute.resolve(strict=False)
    protected = (workspace / "data/cache/wdc_products").resolve(strict=False)
    if resolved == protected or _is_relative_to(resolved, protected):
        raise ValueError("output overlaps protected WDC artifacts")
    allowed = (workspace / f"data/cache/dblp_acm/{version}/teacher_labels").resolve(strict=False)
    if not _is_relative_to(resolved, allowed):
        raise ValueError("output is outside the frozen DBLP teacher-label root")
    return resolved


def _runtime_settings(config: FullLabelConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen",
        "prompt_version": config.prompt_version,
        "instructions": config.instructions,
        "response_schema": {
            "name": config.response_schema.name,
            "strict": config.response_schema.strict,
            "schema": config.response_schema.schema_,
        },
        "settings": {
            "sol_high": {
                "model": config.model,
                "max_attempts": config.max_attempts,
                "reasoning": config.reasoning.model_dump(),
            }
        },
        "provider_routing": config.provider_routing.model_dump(),
        "source_config_sha256": sha256_file(config.config_path),
    }


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "runner": sha256_file(Path(__file__).resolve()),
        "protocol": sha256_file(root / "full_label_protocol.py"),
        "request_client": sha256_file(root / "openrouter_json_schema_client.py"),
        "input_builder": sha256_file(root / "prepare_full_label_inputs.py"),
        "provider_client": sha256_file(root / "llm_providers.py"),
    }


def _cache_identity(
    *,
    config: FullLabelConfig,
    profile_path: Path,
    dataset_id: str,
    dataset_version: str,
    pairs_path: Path,
    input_manifest_path: Path,
    settings_path: Path,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "source_train_sha256": sha256_file(pairs_path),
        "profile_sha256": sha256_file(profile_path.resolve(strict=True)),
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "settings_sha256": sha256_file(settings_path),
        "model": config.model,
        "reasoning": config.reasoning.model_dump(),
        "provider_routing": config.provider_routing.model_dump(),
        "prompt_version": config.prompt_version,
        "instructions_sha256": hashlib.sha256(config.instructions.encode("utf-8")).hexdigest(),
        "parser_version": "strict-json-label-v1",
        "schema_sha256": hashlib.sha256(_canonical_json(config.response_schema.schema_).encode("utf-8")).hexdigest(),
        "code_sha256": _code_hashes(),
    }


def _unresolved_inflight(path: Path) -> list[str]:
    if not path.is_file():
        return []
    state: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state[row["request_id"]] = row["status"]
    return sorted(request_id for request_id, status in state.items() if status != "resolved")


def _verify_existing(root: Path, expected_identity: dict[str, Any]) -> FullLabelArtifacts:
    artifacts = _artifacts(root)
    if not artifacts.completion.is_file():
        raise ValueError("existing labeler output has no completion manifest")
    completion = json.loads(artifacts.completion.read_text(encoding="utf-8"))
    if completion.get("cache_identity") != expected_identity:
        raise ValueError("existing labeler cache identity differs from the frozen run")
    for relative, expected_hash in completion.get("artifacts", {}).items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"existing labeler artifact differs: {relative}")
    return artifacts


def _inventory(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"labeler output contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            values[path.relative_to(root).as_posix()] = sha256_file(path)
    return values


def run_full_labeling(
    *,
    pairs_path: Path,
    dataset_profile_path: Path,
    labeler_config_path: Path,
    output_dir: Path,
    expected_count: int,
    client: JSONSchemaClient,
    workspace_root: Path,
) -> FullLabelArtifacts:
    """Run one deterministic offline labeling integration and publish its artifacts."""
    if type(client) is not FakeJSONSchemaClient:
        raise PermissionError("Phase 3 requires the concrete deterministic fake client")
    workspace = workspace_root.resolve(strict=True)
    profile = load_dataset_profile(dataset_profile_path)
    config = load_full_label_config(labeler_config_path)
    if profile.dataset_id != config.dataset_id:
        raise ValueError("dataset profile and labeler config differ")
    output = _safe_output(output_dir, workspace, profile.logical_version)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging"
    if staging.exists():
        unresolved = _unresolved_inflight(staging / "inflight.jsonl")
        if unresolved:
            raise RuntimeError(f"unresolved inflight requests require manual reconciliation: {unresolved[:5]}")
        raise RuntimeError(f"orphan labeler staging directory requires manual inspection: {staging}")

    # Build expected identities in isolated staging before accepting any cache.
    staging.mkdir()
    staged = _artifacts(staging)
    try:
        prepare_blinded_inputs(
            pairs_path=pairs_path,
            dataset_profile_path=dataset_profile_path,
            inputs_path=staged.inputs,
            manifest_path=staged.input_manifest,
            expected_count=expected_count,
            workspace_root=workspace,
        )
        _write_fsynced(staged.settings, _json_bytes(_runtime_settings(config)))
        identity = _cache_identity(
            config=config,
            profile_path=dataset_profile_path,
            dataset_id=profile.dataset_id,
            dataset_version=profile.logical_version,
            pairs_path=pairs_path.resolve(strict=True),
            input_manifest_path=staged.input_manifest,
            settings_path=staged.settings,
        )
        inputs = [json.loads(line) for line in staged.inputs.read_text(encoding="utf-8").splitlines()]
        predictions: list[tuple[str, str]] = []
        attempts: list[dict[str, Any]] = []
        for row in inputs:
            pair_id = row["pair_id"]
            payload = build_label_request(config, row["input_text"])
            request_sha = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
            request_id = hashlib.sha256(f"{pair_id}:{request_sha}".encode("utf-8")).hexdigest()
            _append_fsynced(staged.inflight, {
                "request_id": request_id,
                "pair_id": pair_id,
                "request_sha256": request_sha,
                "reserved_cost_usd": 0.0,
                "status": "inflight",
            })
            response = client.create(payload)
            _append_fsynced(staged.inflight, {
                "request_id": request_id,
                "provider_response_id": response.get("id"),
                "response_sha256": hashlib.sha256(_canonical_json(response).encode("utf-8")).hexdigest(),
                "response": response,
                "status": "response_received",
            })
            parsed = parse_label_response(config, response)
            predictions.append((pair_id, parsed.label))
            attempt = {
                "pair_id": pair_id,
                "setting": "sol_high",
                "requested_model": config.model,
                "returned_model": parsed.returned_model,
                "attempt": 1,
                "status": "valid",
                "result": parsed.label,
                "usage": parsed.usage,
                "provider_response_id": parsed.response_id,
                "request_sha256": request_sha,
                "reserved_cost_usd": 0.0,
            }
            attempts.append(attempt)
            _append_fsynced(staged.attempts, attempt)
            _append_fsynced(staged.inflight, {
                "request_id": request_id,
                "attempt_sha256": hashlib.sha256(_canonical_json(attempt).encode("utf-8")).hexdigest(),
                "status": "resolved",
            })

        with staged.predictions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["pair_id", "result"])
            writer.writerows(predictions)
            handle.flush()
            os.fsync(handle.fileno())
        _write_fsynced(
            staged.audit,
            _jsonl_bytes([{key: value for key, value in row.items() if key != "result"} for row in attempts]),
        )
        run = {
            "schema_version": 1,
            "setting": "sol_high",
            "model": config.model,
            "prompt_version": config.prompt_version,
            "max_attempts": config.max_attempts,
            "inputs_sha256": sha256_file(staged.inputs),
            "cache_identity": identity,
            "run_provenance": {
                "dataset_id": profile.dataset_id,
                "dataset_version": profile.logical_version,
                "source_train_sha256": sha256_file(pairs_path.resolve(strict=True)),
                "full_input_manifest_sha256": sha256_file(staged.input_manifest),
                "settings_sha256": sha256_file(staged.settings),
                "profile_sha256": sha256_file(dataset_profile_path.resolve(strict=True)),
                "code_sha256": _code_hashes(),
            },
        }
        _write_fsynced(staged.run, _json_bytes(run))
        artifact_paths = [
            staged.predictions, staged.attempts, staged.audit, staged.run,
            staged.inputs, staged.input_manifest, staged.settings, staged.inflight,
        ]
        completion = {
            "schema_version": 1,
            "mode": "offline_fake",
            "row_count": expected_count,
            "api_call_count": 0,
            "fake_dispatch_count": expected_count,
            "estimated_request_count": expected_count,
            "total_cost_usd": 0.0,
            "cost_estimate_status": "blocked_pending_current_pricing_review",
            "pricing_inputs": None,
            "pricing_review_required": config.paid_execution.require_current_pricing_review,
            "paid_execution_authorized": config.paid_execution.authorized,
            "confirmation_flag_required": config.paid_execution.require_confirmation_flag,
            "spend_ceiling_usd": config.paid_execution.spend_ceiling_usd,
            "cache_identity": identity,
            "artifacts": {
                path.relative_to(staging).as_posix(): sha256_file(path)
                for path in artifact_paths
            },
        }
        _write_fsynced(staged.completion, _json_bytes(completion))
        if output.exists():
            existing = _verify_existing(output, identity)
            if _inventory(output) != _inventory(staging):
                raise ValueError("existing labeler artifacts differ from deterministic fake regeneration")
            shutil.rmtree(staging)
            return existing
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, output)
        return _artifacts(output)
    except Exception:
        if staging.exists() and not (staging / "inflight.jsonl").exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--dataset-profile", type=Path, required=True)
    parser.add_argument("--labeler-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    if not args.fake:
        raise SystemExit("Phase 3 permits only --fake; paid execution is not authorized")
    artifacts = run_full_labeling(
        pairs_path=args.pairs,
        dataset_profile_path=args.dataset_profile,
        labeler_config_path=args.labeler_config,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        client=FakeJSONSchemaClient(),
        workspace_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps({"output_dir": str(artifacts.root), "mode": "offline_fake"}, indent=2))


if __name__ == "__main__":
    main()
