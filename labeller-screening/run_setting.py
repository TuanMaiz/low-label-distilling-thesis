from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from screening_lib import load_settings, run_setting, sha256_file, validate_blinded_inputs
from supervision.llm_providers import OpenRouterAnswerOnlyClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one paid gpt-5.6-sol screening setting.")
    parser.add_argument("--setting", required=True, choices=("sol_high", "sol_max", "sol_pro_max"))
    parser.add_argument("--inputs", type=Path, default=HERE / "artifacts/sample/wdc_300.inputs.jsonl")
    parser.add_argument("--settings", type=Path, default=HERE / "settings.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts/predictions")
    parser.add_argument("--confirm-paid-screening", action="store_true")
    parser.add_argument("--spend-ceiling-usd", type=float)
    args = parser.parse_args()

    config = load_settings(args.settings)
    rows = validate_blinded_inputs(args.inputs)
    sample_manifest_path = args.inputs.with_name("wdc_300.manifest.json")
    if not sample_manifest_path.exists():
        raise SystemExit(f"Frozen sample manifest is required: {sample_manifest_path}")
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    if sample_manifest.get("inputs_sha256") != sha256_file(args.inputs):
        raise SystemExit("Blinded input hash does not match the frozen sample manifest")
    if not args.confirm_paid_screening:
        print(json.dumps({
            "dry_run": True,
            "setting": args.setting,
            "model": config["settings"][args.setting]["model"],
            "request_count": len(rows),
            "pricing_snapshot": config["pricing_snapshot"],
            "message": "No API calls made. Review the sample, prompt, setting, and current pricing; rerun with --confirm-paid-screening and --spend-ceiling-usd to authorize this paid run.",
        }, indent=2))
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for a confirmed paid screening run")
    if args.spend_ceiling_usd is None or not math.isfinite(args.spend_ceiling_usd) or args.spend_ceiling_usd <= 0:
        raise SystemExit("A positive --spend-ceiling-usd is required for a confirmed paid screening run")
    client = OpenRouterAnswerOnlyClient(
        model=config["settings"][args.setting]["model"],
        api_key=api_key,
        base_url=config["api_url"],
        timeout=int(config["request_timeout_seconds"]),
        max_tokens=int(config["max_output_tokens"]),
    )
    output = run_setting(args.inputs, args.output_dir, config, args.setting, client, args.spend_ceiling_usd)
    print(f"Complete result: {output}")


if __name__ == "__main__":
    main()
