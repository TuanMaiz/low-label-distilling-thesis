#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare the rented RTX 3090 runtime for the WDC–Qwen vertical slice.

Usage:
  bash scripts/run_wdc_qwen_vertical_slice.sh setup
  bash scripts/run_wdc_qwen_vertical_slice.sh preflight
  bash scripts/run_wdc_qwen_vertical_slice.sh smoke
  bash scripts/run_wdc_qwen_vertical_slice.sh train-gold --confirm-full-training
  bash scripts/run_wdc_qwen_vertical_slice.sh package-arm gold
  bash scripts/run_wdc_qwen_vertical_slice.sh train-llm-hard --confirm-full-training
  bash scripts/run_wdc_qwen_vertical_slice.sh verify-results
  bash scripts/run_wdc_qwen_vertical_slice.sh package-results

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
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-outputs/full_label/wdc-qwen-vertical-slice/wdc_products_80cc_small_100un/qwen3-reranker-0-6b}")"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-3090}"
ALLOW_GPU_NAME_MISMATCH="${ALLOW_GPU_NAME_MISMATCH:-0}"

STUDENT_CONFIG="configs/students/qwen3_reranker_0_6b.json"
TARGET_DIR="data/cache/wdc_products/full_label_targets"
GOLD_TARGET="${TARGET_DIR}/gold.jsonl"
LLM_TARGET="${TARGET_DIR}/llm_hard.jsonl"
VALIDATION="data/cache/wdc_products/serialized/validation.jsonl"
TRAINING_CONTRACT="plans/260820-1507-full-label-er-migration/research/wdc-qwen-training-vertical-slice-contract.md"
PREFLIGHT_DIR="${OUTPUT_ROOT}/preflight"
INPUT_AUDIT="${PREFLIGHT_DIR}/input-length-audit.json"
RUNTIME_IDENTITY="${PREFLIGHT_DIR}/runtime-identity.json"
PREFLIGHT_CONTRACT="${PREFLIGHT_DIR}/artifact-contract.json"
SMOKE_FIXTURES="${OUTPUT_ROOT}/smoke/fixtures"
SMOKE_RUN="${OUTPUT_ROOT}/smoke/run"
FULL_EXPERIMENT_MANIFEST="${OUTPUT_ROOT}/full-experiment-manifest.json"

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

write_checksum() {
  local path="$1"
  local directory
  local filename
  directory="$(dirname "${path}")"
  filename="$(basename "${path}")"
  (cd "${directory}" && sha256sum "${filename}" > "${filename}.sha256")
}

verify_checksum() {
  local checksum="$1"
  local directory
  local filename
  directory="$(dirname "${checksum}")"
  filename="$(basename "${checksum}")"
  (cd "${directory}" && run_cmd sha256sum -c "${filename}")
}

verify_archive_member() {
  local archive="$1"
  local member="$2"
  local current="$3"
  if ! tar -xOf "${archive}" "${member}" | cmp -s - "${current}"; then
    echo "Archive member ${member} does not match current verified results: ${archive}" >&2
    exit 1
  fi
}

require_clean_committed_inputs() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked files differ from the checked-out commit; commit or restore them before full training." >&2
    exit 1
  fi
  run_cmd git ls-files --error-unmatch \
    "${GOLD_TARGET}" "${LLM_TARGET}" "${VALIDATION}" "${STUDENT_CONFIG}" \
    "${TRAINING_CONTRACT}" "scripts/run_wdc_qwen_vertical_slice.sh"
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
    --field "warmup_steps=157"
    --field "planned_optimizer_steps=1570"
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

arm_target() {
  case "$1" in
    gold) printf '%s\n' "${GOLD_TARGET}" ;;
    llm_hard) printf '%s\n' "${LLM_TARGET}" ;;
    *) echo "Unsupported training arm: $1" >&2; exit 2 ;;
  esac
}

arm_root() {
  printf '%s/%s\n' "${OUTPUT_ROOT}" "$1"
}

verify_arm() {
  local arm="$1"
  local write_completion="${2:-}"
  local target
  local root
  local verify_args=()
  target="$(arm_target "${arm}")"
  root="$(arm_root "${arm}")"
  write_or_check_arm_contract "${arm}" "${target}" "${root}" check
  if [[ "${write_completion}" == "--write-completion" ]]; then
    verify_args+=(--write-completion)
  fi
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_preflight verify-arm \
    --arm "${arm}" \
    --target "${target}" \
    --validation "${VALIDATION}" \
    --run-dir "${root}/run" \
    --contract "${root}/artifact-contract.json" \
    --completion "${root}/completion.json" \
    "${verify_args[@]}"
}

write_or_check_arm_contract() {
  local arm="$1"
  local target="$2"
  local root="$3"
  local mode="${4:-check}"
  local commit
  commit="$(git rev-parse HEAD)"
  local contract_args=(
    --field "stage=wdc_qwen_full_validation"
    --field "dataset_id=wdc_products_80cc_small_100un"
    --field "student_id=qwen3-reranker-0-6b"
    --field "arm=${arm}"
    --field "git_commit=${commit}"
    --field "optimizer=AdamW"
    --field "learning_rate=2e-4"
    --field "weight_decay=0.01"
    --field "schedule=linear"
    --field "warmup_ratio=0.10"
    --field "warmup_steps=157"
    --field "planned_optimizer_steps=1570"
    --field "batch_size=1"
    --field "gradient_accumulation_steps=16"
    --field "num_epochs=10"
    --field "early_stopping_patience=3"
    --field "max_input_length=4096"
    --field "input_truncation=false"
    --field "validation_batch_size=1"
    --field "evaluation_batch_size=1"
    --field "precision=auto"
    --field "checkpoint_metric=validation_macro_f1"
    --field "test_scope=locked"
    --file "training_contract=${TRAINING_CONTRACT}"
    --file "student_config=${STUDENT_CONFIG}"
    --file "train_target=${target}"
    --file "validation=${VALIDATION}"
    --file "preflight_contract=${PREFLIGHT_CONTRACT}"
    --file "runtime_identity=${RUNTIME_IDENTITY}"
    --file "input_length_audit=${INPUT_AUDIT}"
    --file "runner=scripts/run_wdc_qwen_vertical_slice.sh"
    --file "preflight=experiments/wdc_qwen_preflight.py"
    --file "trainer=experiments/train_student.py"
    --file "trainer_core=experiments/trainer.py"
    --file "evaluator=experiments/evaluate_student.py"
    --file "checkpoint_manifest=utils/checkpoint_manifest.py"
    --file "classification_threshold=utils/classification_threshold.py"
    --file "metrics=utils/metrics.py"
    --file "artifact_contract_impl=utils/artifact_contract.py"
  )
  mkdir -p "${root}"
  if [[ -f "${root}/artifact-contract.json" ]]; then
    run_cmd "${PYTHON}" -m utils.artifact_contract check \
      --path "${root}/artifact-contract.json" "${contract_args[@]}"
  elif [[ "${mode}" == "write" ]]; then
    run_cmd "${PYTHON}" -m utils.artifact_contract write \
      --path "${root}/artifact-contract.json" "${contract_args[@]}"
  else
    echo "Arm artifact contract is missing: ${root}/artifact-contract.json" >&2
    exit 1
  fi
}

verify_training_arm() {
  local arm="$1"
  local target
  local root
  target="$(arm_target "${arm}")"
  root="$(arm_root "${arm}")"
  write_or_check_arm_contract "${arm}" "${target}" "${root}" check
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_preflight verify-training \
    --arm "${arm}" \
    --target "${target}" \
    --validation "${VALIDATION}" \
    --run-dir "${root}/run" \
    --contract "${root}/artifact-contract.json"
}

arm_state() {
  local root="$1"
  if [[ -f "${root}/completion.json" ]]; then
    printf '%s\n' complete
  elif [[ ! -e "${root}/run" ]]; then
    printf '%s\n' empty
  elif [[ -f "${root}/run/training_summary.json" \
      && -f "${root}/run/checkpoint_manifest.json" ]]; then
    printf '%s\n' trained
  else
    printf '%s\n' partial
  fi
}

evaluation_state() {
  local run_dir="$1"
  local predictions="${run_dir}/validation.predictions.jsonl"
  local metrics="${run_dir}/validation.metrics.json"
  if [[ -e "${predictions}.tmp" || -e "${metrics}.tmp" ]]; then
    printf '%s\n' partial
  elif [[ -f "${predictions}" && -f "${metrics}" ]]; then
    printf '%s\n' complete
  elif [[ ! -e "${predictions}" && ! -e "${metrics}" ]]; then
    printf '%s\n' empty
  else
    printf '%s\n' partial
  fi
}

evaluate_arm() {
  local arm="$1"
  local root
  local run_dir
  root="$(arm_root "${arm}")"
  run_dir="${root}/run"
  run_cmd "${PYTHON}" -m experiments.evaluate_student \
    --student-config "${STUDENT_CONFIG}" \
    --checkpoint "${run_dir}/best_model" \
    --input "${VALIDATION}" \
    --predictions "${run_dir}/validation.predictions.jsonl" \
    --metrics "${run_dir}/validation.metrics.json" \
    --variant "${arm}" \
    --budget full \
    --split validation \
    --batch-size 1 \
    --max-input-length 4096 \
    --precision auto \
    --device cuda
  verify_arm "${arm}" --write-completion
}

train_arm() {
  local arm="$1"
  local confirmation="${2:-}"
  if [[ "${confirmation}" != "--confirm-full-training" ]]; then
    echo "${arm} full training requires --confirm-full-training." >&2
    exit 2
  fi
  require_clean_committed_inputs
  preflight
  run_cmd "${PYTHON}" scripts/check_wdc_target_alignment.py

  local target
  local root
  local run_dir
  local state
  target="$(arm_target "${arm}")"
  root="$(arm_root "${arm}")"
  run_dir="${root}/run"
  state="$(arm_state "${root}")"
  if [[ "${arm}" == "llm_hard" ]]; then
    verify_arm gold
    require_file "${OUTPUT_ROOT}/gold.tar.gz"
    require_file "${OUTPUT_ROOT}/gold.tar.gz.sha256"
    verify_checksum "${OUTPUT_ROOT}/gold.tar.gz.sha256"
    verify_archive_member \
      "${OUTPUT_ROOT}/gold.tar.gz" completion.json \
      "${OUTPUT_ROOT}/gold/completion.json"
  fi

  if [[ "${state}" == "complete" ]]; then
    verify_arm "${arm}"
    echo "Existing completed ${arm} arm verified; nothing to rerun."
    return
  fi
  if [[ "${state}" == "partial" ]]; then
    echo "Incomplete ${arm} output requires inspection before restart: ${run_dir}" >&2
    exit 1
  fi
  if [[ "${state}" == "trained" ]]; then
    verify_training_arm "${arm}"
    case "$(evaluation_state "${run_dir}")" in
      complete)
        verify_arm "${arm}" --write-completion
        ;;
      empty)
        evaluate_arm "${arm}"
        ;;
      partial)
        echo "Partial ${arm} evaluation output requires inspection: ${run_dir}" >&2
        exit 1
        ;;
    esac
    echo "Recovered and verified completed ${arm} training without retraining."
    return
  fi

  write_or_check_arm_contract "${arm}" "${target}" "${root}" write
  run_cmd "${PYTHON}" -m experiments.train_student \
    --student-config "${STUDENT_CONFIG}" \
    --train-targets "${target}" \
    --validation-targets "${VALIDATION}" \
    --output-dir "${run_dir}" \
    --batch-size 1 \
    --validation-batch-size 1 \
    --num-epochs 10 \
    --learning-rate 2e-4 \
    --weight-decay 0.01 \
    --warmup-ratio 0.10 \
    --max-input-length 4096 \
    --early-stopping-patience 3 \
    --gradient-accumulation-steps 16 \
    --precision auto \
    --device cuda
  verify_training_arm "${arm}"
  evaluate_arm "${arm}"
  echo "WDC–Qwen ${arm} full-validation arm passed."
}

package_arm() {
  local arm="$1"
  local root
  local archive="${OUTPUT_ROOT}/${arm}.tar.gz"
  local checksum="${archive}.sha256"
  root="$(arm_root "${arm}")"
  verify_arm "${arm}"
  if [[ -f "${archive}" || -f "${checksum}" ]]; then
    require_file "${archive}"
    require_file "${checksum}"
    verify_checksum "${checksum}"
    verify_archive_member "${archive}" completion.json "${root}/completion.json"
    echo "Existing ${arm} package verified; nothing to rebuild."
    return
  fi
  run_cmd tar -C "${root}" -czf "${archive}" \
    artifact-contract.json completion.json run
  write_checksum "${archive}"
  verify_checksum "${checksum}"
  verify_archive_member "${archive}" completion.json "${root}/completion.json"
  echo "Packaged ${arm} results at ${archive}."
}

verify_results() {
  verify_arm gold
  verify_arm llm_hard
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_preflight verify-experiment \
    --gold-completion "${OUTPUT_ROOT}/gold/completion.json" \
    --llm-hard-completion "${OUTPUT_ROOT}/llm_hard/completion.json" \
    --gold-summary "${OUTPUT_ROOT}/gold/run/training_summary.json" \
    --llm-hard-summary "${OUTPUT_ROOT}/llm_hard/run/training_summary.json" \
    --manifest "${FULL_EXPERIMENT_MANIFEST}"
  echo "Both WDC–Qwen full-validation arms verified."
}

package_results() {
  verify_results
  package_arm gold
  package_arm llm_hard
  local archive="${OUTPUT_ROOT}/wdc-qwen-gold-vs-llm-hard.tar.gz"
  local checksum="${archive}.sha256"
  if [[ -f "${archive}" || -f "${checksum}" ]]; then
    require_file "${archive}"
    require_file "${checksum}"
    verify_checksum "${checksum}"
    verify_archive_member \
      "${archive}" full-experiment-manifest.json "${FULL_EXPERIMENT_MANIFEST}"
    verify_archive_member \
      "${archive}" gold/completion.json "${OUTPUT_ROOT}/gold/completion.json"
    verify_archive_member \
      "${archive}" llm_hard/completion.json "${OUTPUT_ROOT}/llm_hard/completion.json"
    echo "Existing full result package verified; nothing to rebuild."
    return
  fi
  run_cmd tar -C "${OUTPUT_ROOT}" -czf "${archive}" \
    full-experiment-manifest.json gold llm_hard gold.tar.gz \
    gold.tar.gz.sha256 llm_hard.tar.gz llm_hard.tar.gz.sha256
  write_checksum "${archive}"
  verify_checksum "${checksum}"
  verify_archive_member \
    "${archive}" full-experiment-manifest.json "${FULL_EXPERIMENT_MANIFEST}"
  verify_archive_member \
    "${archive}" gold/completion.json "${OUTPUT_ROOT}/gold/completion.json"
  verify_archive_member \
    "${archive}" llm_hard/completion.json "${OUTPUT_ROOT}/llm_hard/completion.json"
  echo "Packaged complete WDC–Qwen comparison at ${archive}."
}

main() {
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
  train-gold)
    train_arm gold "${2:-}"
    ;;
  train-llm-hard)
    train_arm llm_hard "${2:-}"
    ;;
  package-arm)
    package_arm "${2:-}"
    ;;
  verify-results)
    verify_results
    ;;
  package-results)
    package_results
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
