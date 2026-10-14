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
    show_cmd "${train_cmd[@]}"; "${train_cmd[@]}"
    ;;
  evaluate)
    shift; confirmation=--confirm-validation; require_confirmation "$@"
    if [[ -e "${PREDICTIONS}" || -e "${METRICS}" ]]; then
      echo "Refusing to overwrite existing validation results" >&2; exit 1
    fi
    require_file "${STUDENT_CONFIG}"; require_file "${VALIDATION}"
    require_file "${RUN_DIR}/training_summary.json"
    "${PYTHON}" - "${STUDENT_CONFIG}" "${RUN_DIR}" <<'PY'
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
PY
    show_cmd "${evaluate_cmd[@]}"; "${evaluate_cmd[@]}"
    ;;
  *)
    echo "Usage: bash $0 plan | train --confirm-full-training | evaluate --confirm-validation" >&2
    echo "PYTHON selects the runtime; OUTPUT_ROOT selects a fresh experiment directory." >&2
    exit 1
    ;;
esac
