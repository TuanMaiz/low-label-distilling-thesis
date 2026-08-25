#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare the rented RTX 3090 runtime for the WDC–Qwen vertical slice.

Usage:
  bash scripts/run_wdc_qwen_vertical_slice.sh setup
  bash scripts/run_wdc_qwen_vertical_slice.sh preflight
  bash scripts/run_wdc_qwen_vertical_slice.sh smoke

This workflow makes no LLM calls and never reads the WDC test split.
The rented image must already provide a CUDA-compatible PyTorch build.

Environment overrides:
  PYTHON=python
  OUTPUT_ROOT=outputs/full_label/wdc-qwen-vertical-slice/wdc_products_80cc_small_100un/qwen3-reranker-0-6b
  EXPECTED_GPU_SUBSTRING=3090
  ALLOW_GPU_NAME_MISMATCH=0
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/full_label/wdc-qwen-vertical-slice/wdc_products_80cc_small_100un/qwen3-reranker-0-6b}"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-3090}"
ALLOW_GPU_NAME_MISMATCH="${ALLOW_GPU_NAME_MISMATCH:-0}"

STUDENT_CONFIG="configs/students/qwen3_reranker_0_6b.json"
TARGET_DIR="data/cache/wdc_products/full_label_targets"
GOLD_TARGET="${TARGET_DIR}/gold.jsonl"
LLM_TARGET="${TARGET_DIR}/llm_hard.jsonl"
GOLD_MANIFEST="${TARGET_DIR}/gold.manifest.json"
LLM_MANIFEST="${TARGET_DIR}/llm_hard.manifest.json"
VALIDATION="data/cache/wdc_products/serialized/validation.jsonl"
TRAINING_CONTRACT="plans/260820-1507-full-label-er-migration/research/wdc-qwen-training-vertical-slice-contract.md"
PREFLIGHT_DIR="${OUTPUT_ROOT}/preflight"
INPUT_AUDIT="${PREFLIGHT_DIR}/input-length-audit.json"
RUNTIME_IDENTITY="${PREFLIGHT_DIR}/runtime-identity.json"
PREFLIGHT_CONTRACT="${PREFLIGHT_DIR}/artifact-contract.json"
SMOKE_FIXTURES="${OUTPUT_ROOT}/smoke/fixtures"
SMOKE_RUN="${OUTPUT_ROOT}/smoke/run"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

run_cmd() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

setup_runtime() {
  run_cmd "${PYTHON}" -c \
    'import torch; assert torch.cuda.is_available(), "The rental image must provide CUDA-compatible PyTorch"; print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})'
  run_cmd "${PYTHON}" -m pip install --quiet -r requirements-colab.txt
  run_cmd "${PYTHON}" -m utils.peft_runtime sanitize
  run_cmd "${PYTHON}" -m utils.peft_runtime check
  run_cmd "${PYTHON}" -c \
    'import accelerate, peft, torch, transformers; print({"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__, "cuda": torch.cuda.is_available()})'
}

record_runtime_identity() {
  local runtime_args=(
    --output "${RUNTIME_IDENTITY}"
    --expected-gpu-substring "${EXPECTED_GPU_SUBSTRING}"
  )
  if [[ "${ALLOW_GPU_NAME_MISMATCH}" == "1" ]]; then
    runtime_args+=(--allow-gpu-name-mismatch)
  fi
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_preflight runtime \
    "${runtime_args[@]}"
}

write_or_check_preflight_contract() {
  local contract_args=(
    --field "stage=wdc_qwen_preflight"
    --field "dataset_id=wdc_products_80cc_small_100un"
    --field "student_id=qwen3-reranker-0-6b"
    --field "training_arms=gold,llm_hard"
    --field "optimizer=AdamW"
    --field "learning_rate=2e-4"
    --field "weight_decay=0.01"
    --field "schedule=linear"
    --field "warmup_ratio=0.10"
    --field "batch_size=1"
    --field "gradient_accumulation_steps=16"
    --field "num_epochs=10"
    --field "early_stopping_patience=3"
    --field "max_input_length=4096"
    --field "input_truncation=false"
    --field "validation_batch_size=1"
    --field "evaluation_batch_size=1"
    --field "precision=auto"
    --field "test_scope=locked"
    --file "training_contract=${TRAINING_CONTRACT}"
    --file "student_config=${STUDENT_CONFIG}"
    --file "gold_target=${GOLD_TARGET}"
    --file "llm_hard_target=${LLM_TARGET}"
    --file "gold_manifest=${GOLD_MANIFEST}"
    --file "llm_hard_manifest=${LLM_MANIFEST}"
    --file "validation=${VALIDATION}"
    --file "input_length_audit=${INPUT_AUDIT}"
    --file "runtime_identity=${RUNTIME_IDENTITY}"
    --file "runner=scripts/run_wdc_qwen_vertical_slice.sh"
    --file "preflight=experiments/wdc_qwen_preflight.py"
    --file "trainer=experiments/train_student.py"
    --file "trainer_core=experiments/trainer.py"
    --file "evaluator=experiments/evaluate_student.py"
    --file "generative_backend=models/generative_reranker_student.py"
    --file "classification_backend=models/classification_student.py"
    --file "target_row_loader=models/seq2seq_student.py"
    --file "student_config_loader=models/student_config.py"
    --file "torch_runtime=utils/torch_runtime.py"
    --file "peft_runtime=utils/peft_runtime.py"
    --file "checkpoint_manifest=utils/checkpoint_manifest.py"
    --file "classification_threshold=utils/classification_threshold.py"
    --file "metrics=utils/metrics.py"
    --file "artifact_contract_impl=utils/artifact_contract.py"
    --file "target_validator=supervision/build_full_label_targets.py"
  )
  mkdir -p "${PREFLIGHT_DIR}"
  if [[ -f "${PREFLIGHT_CONTRACT}" ]]; then
    run_cmd "${PYTHON}" -m utils.artifact_contract check \
      --path "${PREFLIGHT_CONTRACT}" "${contract_args[@]}"
  else
    run_cmd "${PYTHON}" -m utils.artifact_contract write \
      --path "${PREFLIGHT_CONTRACT}" "${contract_args[@]}"
  fi
}

preflight() {
  require_file "${STUDENT_CONFIG}"
  require_file "${GOLD_TARGET}"
  require_file "${LLM_TARGET}"
  require_file "${GOLD_MANIFEST}"
  require_file "${LLM_MANIFEST}"
  require_file "${VALIDATION}"
  require_file "${TRAINING_CONTRACT}"

  run_cmd "${PYTHON}" -m experiments.wdc_qwen_preflight validate \
    --target-dir "${TARGET_DIR}" \
    --validation "${VALIDATION}" \
    --student-config "${STUDENT_CONFIG}"
  run_cmd "${PYTHON}" -m utils.peft_runtime sanitize
  run_cmd "${PYTHON}" -m utils.peft_runtime check
  record_runtime_identity
  mkdir -p "${PREFLIGHT_DIR}"
  local audit_candidate="${INPUT_AUDIT}.candidate"
  if [[ -e "${audit_candidate}" ]]; then
    echo "Unresolved input-audit candidate exists: ${audit_candidate}" >&2
    exit 1
  fi
  run_cmd "${PYTHON}" -m models.generative_reranker_student \
    --student-config "${STUDENT_CONFIG}" \
    --input "${GOLD_TARGET}" \
    --input "${LLM_TARGET}" \
    --input "${VALIDATION}" \
    --output "${audit_candidate}" \
    --max-input-length 4096
  if [[ -f "${INPUT_AUDIT}" ]]; then
    if ! cmp -s "${audit_candidate}" "${INPUT_AUDIT}"; then
      echo "Input-length audit differs from the frozen audit; candidate retained at ${audit_candidate}." >&2
      exit 1
    fi
    rm "${audit_candidate}"
  else
    mv "${audit_candidate}" "${INPUT_AUDIT}"
  fi
  write_or_check_preflight_contract
  echo "WDC–Qwen RTX 3090 preflight passed."
}

smoke() {
  preflight
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_preflight prepare-smoke \
    --gold-target "${GOLD_TARGET}" \
    --validation "${VALIDATION}" \
    --output-dir "${SMOKE_FIXTURES}" \
    --per-class 8

  if [[ -d "${SMOKE_RUN}" ]]; then
    if [[ -f "${SMOKE_RUN}/checkpoint_manifest.json" \
        && -f "${SMOKE_RUN}/validation.predictions.jsonl" \
        && -f "${SMOKE_RUN}/validation.metrics.json" ]]; then
      run_cmd "${PYTHON}" -m utils.checkpoint_manifest check \
        --output-dir "${SMOKE_RUN}"
      echo "Existing completed WDC–Qwen smoke output verified; nothing to rerun."
      return
    fi
    echo "Incomplete smoke output requires inspection before retry: ${SMOKE_RUN}" >&2
    exit 1
  fi
  run_cmd "${PYTHON}" -m experiments.train_student \
    --student-config "${STUDENT_CONFIG}" \
    --train-targets "${SMOKE_FIXTURES}/train.gold.smoke.jsonl" \
    --validation-targets "${SMOKE_FIXTURES}/validation.smoke.jsonl" \
    --output-dir "${SMOKE_RUN}" \
    --batch-size 1 \
    --validation-batch-size 1 \
    --num-epochs 1 \
    --learning-rate 2e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.0 \
    --max-input-length 4096 \
    --early-stopping-patience 1 \
    --gradient-accumulation-steps 16 \
    --precision auto \
    --device cuda
  run_cmd "${PYTHON}" -m utils.checkpoint_manifest check \
    --output-dir "${SMOKE_RUN}"
  run_cmd "${PYTHON}" -m experiments.evaluate_student \
    --student-config "${STUDENT_CONFIG}" \
    --checkpoint "${SMOKE_RUN}/best_model" \
    --input "${SMOKE_FIXTURES}/validation.smoke.jsonl" \
    --predictions "${SMOKE_RUN}/validation.predictions.jsonl" \
    --metrics "${SMOKE_RUN}/validation.metrics.json" \
    --variant smoke \
    --budget full \
    --split validation_smoke \
    --batch-size 1 \
    --max-input-length 4096 \
    --precision auto \
    --device cuda
  echo "WDC–Qwen LoRA smoke train, checkpoint reload, and evaluation passed."
}

case "${1:-}" in
  setup)
    setup_runtime
    ;;
  preflight)
    preflight
    ;;
  smoke)
    smoke
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
