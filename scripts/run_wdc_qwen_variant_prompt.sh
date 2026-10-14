#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"
PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uncertainty_refinement/wdc-qwen-quick-refined-variant-identity-v1}"
STUDENT_CONFIG="configs/students/qwen3_reranker_0_6b_variant_identity_v1.json"
TRAIN_TARGETS="data/cache/wdc_products/refinement/quick-tier-v1/train.jsonl"
VALIDATION="data/cache/wdc_products/serialized/validation.jsonl"
RUN_DIR="${OUTPUT_ROOT}/run"
PREDICTIONS="${OUTPUT_ROOT}/validation-predictions.jsonl"
METRICS="${OUTPUT_ROOT}/validation-metrics.json"
PREFLIGHT="${OUTPUT_ROOT}/preflight"
LOGS="${OUTPUT_ROOT}/logs"
CONTRACT="${PREFLIGHT}/artifact-contract.json"
contract_args=(
  --field stage=wdc_variant_identity_v1 --field train_rows=2493 --field validation_rows=2500
  --file "student_config=${STUDENT_CONFIG}" --file "train=${TRAIN_TARGETS}"
  --file "validation=${VALIDATION}" --file "runtime=${PREFLIGHT}/runtime.json"
  --file "audit=${PREFLIGHT}/input-length-audit.json"
  --file runner=scripts/run_wdc_qwen_variant_prompt.sh
  --file trainer=experiments/train_student.py --file evaluator=experiments/evaluate_student.py
  --file backend=models/generative_reranker_student.py
  --file config_loader=models/student_config.py)
check_preflight() {
  require_file "${CONTRACT}"
  "${PYTHON}" -m utils.artifact_contract check --path "${CONTRACT}" "${contract_args[@]}"
}
verify_checkpoint() {
  require_file "${RUN_DIR}/training_summary.json"
  "${PYTHON}" - "${STUDENT_CONFIG}" "${RUN_DIR}" <<'CHECKPOINT'
import json, sys
from pathlib import Path
from utils.checkpoint_manifest import validate_checkpoint_manifest
config_path, run_dir = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text())
summary = json.loads((run_dir / "training_summary.json").read_text())
if summary.get("reranker_instruction") != config["reranker_instruction"]:
    raise SystemExit("Checkpoint instruction mismatch: train with the variant config first")
if summary.get("model_name") != config["model_name"] or summary.get("train_rows") != 2493:
    raise SystemExit("Checkpoint does not describe the 2493-row refined Qwen experiment")
manifest = validate_checkpoint_manifest(run_dir)
if summary.get("checkpoint_manifest") != manifest:
    raise SystemExit("Training summary does not match the verified checkpoint manifest")
CHECKPOINT
}

train_cmd=("${PYTHON}" -m experiments.train_student
  --student-config "${STUDENT_CONFIG}"
  --train-targets "${TRAIN_TARGETS}" --validation-targets "${VALIDATION}"
  --output-dir "${RUN_DIR}" --batch-size 1 --validation-batch-size 1
  --gradient-accumulation-steps 16 --num-epochs 10 --learning-rate 0.0002
  --weight-decay 0.01 --warmup-steps 0 --warmup-ratio 0.1
  --max-input-length 4096 --seed 42 --early-stopping-patience 3
  --precision fp16 --device cuda)
evaluate_cmd=("${PYTHON}" -m experiments.evaluate_student
  --student-config "${STUDENT_CONFIG}" --checkpoint "${RUN_DIR}/best_model"
  --input "${VALIDATION}" --predictions "${PREDICTIONS}" --metrics "${METRICS}"
  --variant quick_refined_variant_identity_v1 --split validation
  --batch-size 1 --max-input-length 4096 --precision fp16 --device cuda)

show_cmd() { printf '%q ' "$@"; printf '\n'; }
require_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}
require_confirmation() {
  [[ "$#" == 1 && "$1" == "$confirmation" ]] || {
    echo "This action requires ${confirmation}; use plan for an offline preview." >&2
    exit 1
  }
}

case "${1:-}" in
  setup)
    [[ "$#" == 1 ]] || exit 1
    mkdir -p "${LOGS}"
    "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "The GPU image must provide CUDA PyTorch"; print(torch.__version__, torch.cuda.get_device_name(0))'
    "${PYTHON}" -m pip install -r requirements-colab.txt 2>&1 | tee "${LOGS}/setup.log"
    "${PYTHON}" -m utils.peft_runtime sanitize
    "${PYTHON}" -m utils.peft_runtime check
    "${PYTHON}" -m pip freeze > "${LOGS}/pip-freeze.txt"
    ;;
  preflight)
    [[ "$#" == 1 ]] || exit 1
    require_file "${STUDENT_CONFIG}"; require_file "${TRAIN_TARGETS}"; require_file "${VALIDATION}"
    mkdir -p "${PREFLIGHT}" "${LOGS}"
    "${PYTHON}" - "${STUDENT_CONFIG}" "${TRAIN_TARGETS}" "${VALIDATION}" "${PREFLIGHT}" <<'PREFLIGHT_PY'
import json, subprocess, sys
from collections import Counter
from pathlib import Path
import torch
from importlib import metadata
config_path, train_path, val_path, out = map(Path, sys.argv[1:])
assert torch.cuda.is_available(), "CUDA is unavailable"
config = json.loads(config_path.read_text())
baseline = json.loads(Path("configs/students/qwen3_reranker_0_6b.json").read_text())
assert [k for k in baseline if baseline[k] != config[k]] == ["reranker_instruction"]
assert "same specific product variant" in config["reranker_instruction"]
train = [json.loads(l) for l in train_path.read_text().splitlines()]
val = [json.loads(l) for l in val_path.read_text().splitlines()]
assert len(train) == 2493 and Counter(r["label"] for r in train) == {0: 2004, 1: 489}
assert len(val) == 2500 and Counter(r["label"] for r in val) == {0: 2000, 1: 500}
a, b = {r["pair_id"] for r in train}, {r["pair_id"] for r in val}
assert len(a) == 2493 and len(b) == 2500 and not a & b
assert all("\n\nRecord A:\n" in r["input_text"] and "\n\nRecord B:\n" in r["input_text"] for r in train + val)
identity = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
            "train_rows": len(train), "validation_rows": len(val),
            "packages": {k: metadata.version(k) for k in ("transformers", "peft", "accelerate")}}
(out / "runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
(out / "student-config.json").write_bytes(config_path.read_bytes())
print(json.dumps(identity, indent=2))
PREFLIGHT_PY
    "${PYTHON}" -m models.generative_reranker_student \
      --student-config "${STUDENT_CONFIG}" --input "${TRAIN_TARGETS}" --input "${VALIDATION}" \
      --max-input-length 4096 --output "${PREFLIGHT}/input-length-audit.json" \
      2>&1 | tee "${LOGS}/input-audit.log"
    if [[ -f "${CONTRACT}" ]]; then
      check_preflight
    else
      "${PYTHON}" -m utils.artifact_contract write --path "${CONTRACT}" "${contract_args[@]}"
    fi
    echo "Variant-prompt GPU preflight passed."
    ;;
  plan)
    [[ "$#" == 1 ]] || { echo "plan takes no arguments" >&2; exit 1; }
    show_cmd "${train_cmd[@]}"
    show_cmd "${evaluate_cmd[@]}"
    ;;
  train)
    shift; confirmation=--confirm-full-training; require_confirmation "$@"
    if [[ -e "${RUN_DIR}" ]]; then
      echo "Refusing to replace an existing run directory: ${RUN_DIR}" >&2; exit 1
    fi
    require_file "${STUDENT_CONFIG}"; require_file "${TRAIN_TARGETS}"; require_file "${VALIDATION}"
    check_preflight
    mkdir -p "${LOGS}"
    show_cmd "${train_cmd[@]}"
    "${train_cmd[@]}" 2>&1 | tee "${LOGS}/train.log"
    ;;
  evaluate)
    shift; confirmation=--confirm-validation; require_confirmation "$@"
    if [[ -e "${PREDICTIONS}" || -e "${METRICS}" ]]; then
      echo "Refusing to overwrite existing validation results" >&2; exit 1
    fi
    require_file "${STUDENT_CONFIG}"; require_file "${VALIDATION}"
    verify_checkpoint
    check_preflight
    mkdir -p "${LOGS}"
    show_cmd "${evaluate_cmd[@]}"
    "${evaluate_cmd[@]}" 2>&1 | tee "${LOGS}/validation.log"
    ;;
  package-results)
    [[ "$#" == 1 ]] || exit 1
    require_file "${PREDICTIONS}"; require_file "${METRICS}"
    check_preflight
    verify_checkpoint
    "${PYTHON}" - "${OUTPUT_ROOT}" "${VALIDATION}" <<'PACKAGE_PY'
import json, math, sys, zipfile
from pathlib import Path
from utils.artifact_contract import sha256_file
out, val_path = map(Path, sys.argv[1:])
expected = {r["pair_id"]: r for r in map(json.loads, val_path.read_text().splitlines())}
rows = list(map(json.loads, (out / "validation-predictions.jsonl").read_text().splitlines()))
metrics = json.loads((out / "validation-metrics.json").read_text())
summary = json.loads((out / "run/training_summary.json").read_text())
assert len(rows) == len({r["pair_id"] for r in rows}) == len(expected) == 2500
assert set(expected) == {r["pair_id"] for r in rows}
for r in rows:
    assert r["is_valid"] and r["label"] == expected[r["pair_id"]]["label"]
    score = r["match_probability"]
    assert math.isfinite(score) and 0 <= score <= 1
    assert r["prediction"] == int(score >= summary["decision_threshold"])
tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in rows)
fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in rows)
fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in rows)
tn = len(rows) - tp - fp - fn
assert all(metrics[k] == v for k, v in {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": 2500, "valid": 2500, "invalid": 0}.items())
assert math.isclose(metrics["same_f1"], 2 * tp / (2 * tp + fp + fn), abs_tol=1e-12)
assert metrics["split"] == "validation" and metrics["variant"] == "quick_refined_variant_identity_v1"
assert metrics["decision_threshold"] == summary["decision_threshold"]
archive = Path(str(out) + ".zip")
temporary = Path(str(archive) + ".tmp")
if archive.exists() or temporary.exists():
    raise SystemExit("Refusing to replace an existing results archive or partial archive")
files = sorted(p for p in out.rglob("*") if p.is_file())
assert not any(p.is_symlink() for p in files)
with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for p in files:
        z.write(p, p.relative_to(out.parent))
with zipfile.ZipFile(temporary) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(files)
temporary.replace(archive)
Path(str(archive) + ".sha256").write_text(sha256_file(archive) + "  " + archive.name + "\n")
print("Download:", archive.resolve())
print("Checksum:", Path(str(archive) + ".sha256").resolve())
PACKAGE_PY
    ;;
  *)
    echo "Usage: bash $0 setup | preflight | plan | train --confirm-full-training | evaluate --confirm-validation | package-results" >&2
    echo "PYTHON selects the runtime; OUTPUT_ROOT selects a fresh experiment directory." >&2
    exit 1
    ;;
esac
