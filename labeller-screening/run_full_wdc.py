from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from screening_lib import (  # noqa: E402
    load_settings,
    prepare_full_training_inputs,
    run_setting,
    sha256_file,
    validate_blinded_inputs,
    validate_reuse_artifacts,
)
from supervision.llm_providers import (  # noqa: E402
    OpenRouterAnswerOnlyClient,
    resolve_openrouter_api_key,
)


DEFAULT_SOURCE = ROOT / "data/cache/wdc_products/serialized/train.jsonl"
DEFAULT_INPUT_DIR = ROOT / "data/cache/wdc_products/teacher_labels/full_sol_high"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "predictions"
DEFAULT_REUSE_INPUTS = HERE / "artifacts/sample/wdc_300.inputs.jsonl"
DEFAULT_REUSE_ATTEMPTS = HERE / "artifacts/predictions/sol_high.attempts.jsonl"
EXPECTED_WDC_TRAIN_COUNT = 2500
VERTICAL_CONTRACT = (
    ROOT
    / "plans/260820-1507-full-label-er-migration/research/wdc-sol-high-vertical-slice-contract.md"
)
MAIN_CONTRACT = (
    ROOT / "plans/260820-1507-full-label-er-migration/research/experiment-contract.md"
)


def _git_provenance() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_commit": commit,
        "dirty_worktree": bool(status),
        "dirty_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_worktree_policy": (
            "Allowed for this vertical slice because every executable/config/contract/input "
            "used by the paid run is independently SHA-256 bound in the run manifest."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label the complete official WDC training split with the screened Sol-high setting."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--settings", type=Path, default=HERE / "settings.json")
    parser.add_argument("--reuse-inputs", type=Path, default=DEFAULT_REUSE_INPUTS)
    parser.add_argument("--reuse-attempts", type=Path, default=DEFAULT_REUSE_ATTEMPTS)
    parser.add_argument("--confirm-paid-labeling", action="store_true")
    parser.add_argument("--spend-ceiling-usd", type=float)
    args = parser.parse_args()

    config = load_settings(args.settings)
    manifest = prepare_full_training_inputs(args.source, args.input_dir)
    inputs_path = args.input_dir / "wdc_train_full.inputs.jsonl"
    rows = validate_blinded_inputs(inputs_path, expected_count=EXPECTED_WDC_TRAIN_COUNT)
    if manifest["count"] != EXPECTED_WDC_TRAIN_COUNT:
        raise SystemExit(
            f"Expected official WDC train count {EXPECTED_WDC_TRAIN_COUNT}, found {manifest['count']}"
        )
    manifest_path = args.input_dir / "wdc_train_full.manifest.json"
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen_manifest.get("inputs_sha256") != sha256_file(inputs_path):
        raise SystemExit("Full blinded input hash does not match its frozen manifest")

    validate_blinded_inputs(args.reuse_inputs, expected_count=300)
    reuse_predictions, _, _ = validate_reuse_artifacts(
        rows, config, "sol_high", args.reuse_attempts, args.reuse_inputs
    )
    reuse_count = len(reuse_predictions)
    full_manifest_path = manifest_path
    run_provenance = {
        "dataset_id": "wdc_products_80cc_small_100un",
        "dataset_release": "2022-12-22",
        "source_train_sha256": manifest["source_sha256"],
        "full_input_manifest_path": str(full_manifest_path),
        "full_input_manifest_sha256": sha256_file(full_manifest_path),
        "settings_path": str(args.settings),
        "settings_sha256": sha256_file(args.settings),
        "runner_entrypoint_path": str(Path(__file__).resolve()),
        "runner_entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "vertical_contract_path": str(VERTICAL_CONTRACT),
        "vertical_contract_sha256": sha256_file(VERTICAL_CONTRACT),
        "main_contract_path": str(MAIN_CONTRACT),
        "main_contract_sha256": sha256_file(MAIN_CONTRACT),
        **_git_provenance(),
    }
    dry_run = {
        "dry_run": not args.confirm_paid_labeling,
        "setting": "sol_high",
        "model": config["settings"]["sol_high"]["model"],
        "full_training_rows": len(rows),
        "reused_completed_rows": reuse_count,
        "new_request_count": len(rows) - reuse_count,
        "spend_ceiling_usd": args.spend_ceiling_usd,
        "inputs_sha256": sha256_file(inputs_path),
        "pricing_snapshot": config["pricing_snapshot"],
    }
    if not args.confirm_paid_labeling:
        print(json.dumps(dry_run, indent=2))
        return

    if (
        args.spend_ceiling_usd is None
        or not math.isfinite(args.spend_ceiling_usd)
        or args.spend_ceiling_usd <= 0
    ):
        raise SystemExit(
            "A positive --spend-ceiling-usd is required for confirmed paid labeling"
        )
    api_key = resolve_openrouter_api_key(env_file=ROOT / ".env")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for confirmed paid labeling")
    client = OpenRouterAnswerOnlyClient(
        model=config["settings"]["sol_high"]["model"],
        api_key=api_key,
        base_url=config["api_url"],
        timeout=int(config["request_timeout_seconds"]),
        max_tokens=int(config["max_output_tokens"]),
    )
    output = run_setting(
        inputs_path=inputs_path,
        output_dir=args.output_dir,
        config=config,
        setting="sol_high",
        client=client,
        spend_ceiling_usd=args.spend_ceiling_usd,
        expected_count=EXPECTED_WDC_TRAIN_COUNT,
        reuse_attempts_path=args.reuse_attempts,
        reuse_inputs_path=args.reuse_inputs,
        run_provenance=run_provenance,
    )
    print(json.dumps({**dry_run, "dry_run": False, "complete_result": str(output)}, indent=2))


if __name__ == "__main__":
    main()
