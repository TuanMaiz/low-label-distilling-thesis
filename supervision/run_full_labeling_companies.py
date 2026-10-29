"""Teacher labelling for MaDI-Bench Companies (dataset 3), three passes + majority.

Uses the same engine as the WDC runner (``labeller-screening/screening_lib.py``:
answer-only OpenRouter client, attempt journal, spend ceiling) on the prepared,
gold-free inputs ``teacher_labels/inputs.jsonl`` (pair_id and input_text only),
and the existing consensus builder for the majority vote. Outputs stay in the
git-ignored Companies cache.

    python -m supervision.run_full_labeling_companies plan
    python -m supervision.run_full_labeling_companies label --pass-index 1 \\
        --confirm-paid-labeling --spend-ceiling-usd 5
    python -m supervision.run_full_labeling_companies majority

Without ``--confirm-paid-labeling`` the ``label`` action is a dry run and makes
no API call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCREENING = ROOT / "labeller-screening"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCREENING))

from screening_lib import load_settings, run_setting, sha256_file, validate_blinded_inputs  # noqa: E402
from supervision.build_teacher_consensus import build_teacher_consensus  # noqa: E402

DATASET_ID = "madi_companies"
DATA = ROOT / "data/cache/madi_companies/madi-bench-companies-base-062058d"
SETTINGS = ROOT / "configs/labelers/madi_companies_sol_high.json"
SETTING = "sol_high"
EXPECTED_COUNT = 1968
PASSES = (1, 2, 3)


def inputs_path(data_dir: Path) -> Path:
    """The prepared gold-free inputs, checked against the preparation manifest."""
    path = data_dir / "teacher_labels/inputs.jsonl"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest["outputs"]["teacher_labels/inputs.jsonl"]
    if manifest.get("dataset_id") != DATASET_ID or sha256_file(path) != contract["sha256"]:
        raise ValueError("teacher inputs differ from the prepared Companies manifest; run data.prepare_madi_companies verify")
    validate_blinded_inputs(path, expected_count=EXPECTED_COUNT)   # only pair_id and input_text allowed
    return path


def pass_dir(data_dir: Path, index: int) -> Path:
    if index not in PASSES:
        raise ValueError("pass index must be 1, 2 or 3")
    return data_dir / f"teacher_labels/pass_{index:02d}"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(data_dir: Path, index: int, settings_path: Path) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    return {
        "dataset_id": DATASET_ID, "pass_index": index,
        "inputs_sha256": sha256_file(inputs_path(data_dir)),
        "preparation_manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "settings_path": str(settings_path), "settings_sha256": sha256_file(settings_path),
        "runner_entrypoint_path": str(runner), "runner_entrypoint_sha256": sha256_file(runner),
        "git_commit": _git_commit(),
    }


def label_pass(data_dir: Path, index: int, config: dict, client: Any, spend_ceiling_usd: float,
               settings_path: Path = SETTINGS) -> Path:
    """Run one independent pass. Refuses to touch a pass that already has a final result."""
    out = pass_dir(data_dir, index)
    if (out / f"{SETTING}.csv").exists():
        raise FileExistsError(f"pass {index} already has a final result: {out}")
    return run_setting(inputs_path=inputs_path(data_dir), output_dir=out, config=config, setting=SETTING,
                       client=client, spend_ceiling_usd=spend_ceiling_usd, expected_count=EXPECTED_COUNT,
                       run_provenance=provenance(data_dir, index, settings_path))


def build_majority(data_dir: Path) -> dict[str, Any]:
    """Three-pass majority → arm-A training file and manifest (gold-free)."""
    out = data_dir / "majority_labels"
    if (out / "train.jsonl").exists():
        raise FileExistsError(f"majority labels already built: {out}")
    source = inputs_path(data_dir)
    passes = [pass_dir(data_dir, i) / f"{SETTING}.csv" for i in PASSES]
    summary = build_teacher_consensus(inputs_path=source, pass_paths=passes, output_dir=out,
                                      expected_count=EXPECTED_COUNT)
    labels = {}
    for line in (out / "llm_unrefined.csv").read_text(encoding="utf-8").splitlines()[1:]:
        pair_id, result = line.split(",")
        labels[pair_id] = result
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append({"dataset_id": DATASET_ID, "pair_id": r["pair_id"], "input_text": r["input_text"],
                     "label_source": "llm_majority", "split": "train",
                     "target_text": labels[r["pair_id"]].replace("_", "-")})
    train_bytes = b"".join(json.dumps(r, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for r in rows)
    (out / "train.jsonl").write_bytes(train_bytes)
    manifest = {
        "arm": "A", "policy": "llm-majority-unrefined", "label_source": "llm_majority", "gold_accessed": False,
        "dataset_id": DATASET_ID, "train_path": "majority_labels/train.jsonl", "train_rows": len(rows),
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "class_counts": dict(sorted(Counter(r["target_text"] for r in rows).items())),
        "consistency_counts": summary["consistency_counts"],
        "passes": {f"pass_{i:02d}": sha256_file(p) for i, p in zip(PASSES, passes)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def plan(data_dir: Path, settings_path: Path) -> dict[str, Any]:
    config = load_settings(settings_path)
    source = inputs_path(data_dir)
    return {
        "dataset_id": DATASET_ID, "rows": EXPECTED_COUNT, "inputs": str(source), "inputs_sha256": sha256_file(source),
        "settings": str(settings_path), "prompt_version": config["prompt_version"],
        "model": config["settings"][SETTING]["model"], "pricing_snapshot": config["pricing_snapshot"],
        "passes": {f"pass_{i:02d}": ("done" if (pass_dir(data_dir, i) / f"{SETTING}.csv").exists() else "pending")
                   for i in PASSES},
        "majority_built": (data_dir / "majority_labels/train.jsonl").exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["plan", "label", "majority"])
    parser.add_argument("--pass-index", type=int, choices=PASSES)
    parser.add_argument("--settings", type=Path, default=SETTINGS)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--confirm-paid-labeling", action="store_true")
    parser.add_argument("--spend-ceiling-usd", type=float)
    args = parser.parse_args()
    if args.action == "majority":
        print(json.dumps(build_majority(args.data_dir), indent=2))
        return
    summary = plan(args.data_dir, args.settings)
    if args.action == "plan":
        print(json.dumps(summary, indent=2))
        return
    if args.pass_index is None:
        parser.error("label requires --pass-index")
    if not args.confirm_paid_labeling:
        print(json.dumps({**summary, "dry_run": True, "pass_index": args.pass_index}, indent=2))
        return
    if args.spend_ceiling_usd is None or not math.isfinite(args.spend_ceiling_usd) or args.spend_ceiling_usd <= 0:
        raise SystemExit("A positive --spend-ceiling-usd is required for confirmed paid labeling")
    from supervision.llm_providers import OpenRouterAnswerOnlyClient, resolve_openrouter_api_key
    api_key = resolve_openrouter_api_key(env_file=ROOT / ".env")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for confirmed paid labeling")
    config = load_settings(args.settings)
    client = OpenRouterAnswerOnlyClient(model=config["settings"][SETTING]["model"], api_key=api_key,
                                        base_url=config["api_url"], timeout=int(config["request_timeout_seconds"]),
                                        max_tokens=int(config["max_output_tokens"]))
    result = label_pass(args.data_dir, args.pass_index, config, client, args.spend_ceiling_usd, args.settings)
    print(json.dumps({"pass_index": args.pass_index, "complete_result": str(result)}, indent=2))


if __name__ == "__main__":
    main()
