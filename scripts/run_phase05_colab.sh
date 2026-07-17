#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a config-selected Phase 5 compact-student pilot on a Google Colab GPU.

Usage:
  bash scripts/run_phase05_colab.sh setup
  bash scripts/run_phase05_colab.sh preflight
  bash scripts/run_phase05_colab.sh run [all|gold_random|llm_random|llm_active_bucketed_v1]
  bash scripts/run_phase05_colab.sh aggregate [--allow-partial]
  bash scripts/run_phase05_colab.sh package-results
  bash scripts/run_phase05_colab.sh package-checkpoints
  bash scripts/run_phase05_colab.sh all

Recommended Colab flow after cloning the repository branch:
  bash scripts/run_phase05_colab.sh setup
  STUDENT_CONFIG=configs/students/modernbert_base.json \
    STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
    bash scripts/run_phase05_colab.sh all

The all command resumes at completed stage boundaries. Completed
training/evaluation artifacts are skipped unless FORCE=1 is set; interrupted
training restarts that variant. It never calls a teacher LLM and never reads
the test target.

Environment overrides:
  PYTHON=python
  STUDENT_CONFIG=configs/students/flan_t5_base.json
  STUDENT_OUTPUT_ROOT=outputs/students
  BUDGET=128
  BATCH_SIZE=auto             # classifier: 16; seq2seq: 4
  VALIDATION_BATCH_SIZE=auto  # A100/BF16: 32; other CUDA: 16
  EVAL_BATCH_SIZE=8
  NUM_EPOCHS=8
  LEARNING_RATE=5e-5
  WEIGHT_DECAY=0.01
  WARMUP_STEPS=0
  WARMUP_RATIO=auto           # classifier: 0.10; seq2seq: 0
  CLASSIFIER_HEAD_EPOCHS=2
  CLASSIFIER_HEAD_LEARNING_RATE=1e-3
  CLASSIFIER_ENCODER_LEARNING_RATE=1e-5
  CLASSIFIER_UNFREEZE_LAST_N_LAYERS=4
  MAX_INPUT_LENGTH=512
  MAX_TARGET_LENGTH=8
  MAX_NEW_TOKENS=8
  EARLY_STOPPING_PATIENCE=3
  SEED=42
  DEVICE=cuda
  PRECISION=auto             # A100/BF16: bf16; other CUDA: fp16
  COST_ASSUMPTIONS=configs/phase05_cost_assumptions.json
  USE_WANDB=0
  FORCE=0
  EXPECTED_BRANCH=codex/distiller-wdc-implementation
  ALLOW_BRANCH_MISMATCH=0
  ALLOW_CPU=0
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-python}"
STUDENT_CONFIG="${STUDENT_CONFIG:-configs/students/flan_t5_base.json}"
STUDENT_OUTPUT_ROOT="${STUDENT_OUTPUT_ROOT:-${OUTPUT_ROOT:-outputs/students}}"
BUDGET="${BUDGET:-128}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-auto}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
WARMUP_RATIO="${WARMUP_RATIO:-auto}"
CLASSIFIER_HEAD_EPOCHS="${CLASSIFIER_HEAD_EPOCHS:-2}"
CLASSIFIER_HEAD_LEARNING_RATE="${CLASSIFIER_HEAD_LEARNING_RATE:-1e-3}"
CLASSIFIER_ENCODER_LEARNING_RATE="${CLASSIFIER_ENCODER_LEARNING_RATE:-1e-5}"
CLASSIFIER_UNFREEZE_LAST_N_LAYERS="${CLASSIFIER_UNFREEZE_LAST_N_LAYERS:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-auto}"
COST_ASSUMPTIONS="${COST_ASSUMPTIONS:-configs/phase05_cost_assumptions.json}"
USE_WANDB="${USE_WANDB:-0}"
FORCE="${FORCE:-0}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-codex/distiller-wdc-implementation}"
ALLOW_BRANCH_MISMATCH="${ALLOW_BRANCH_MISMATCH:-0}"
ALLOW_CPU="${ALLOW_CPU:-0}"

read_student_config_field() {
  "${PYTHON}" -m models.student_config --config "${STUDENT_CONFIG}" --field "$1"
}

STUDENT_ID="$(read_student_config_field student_id)"
LEGACY_MODEL_NAME_OVERRIDE="${MODEL_NAME:-}"
MODEL_NAME="$(read_student_config_field model_name)"
STUDENT_ARCHITECTURE="$(read_student_config_field architecture)"
if [[ "${BATCH_SIZE}" == "auto" ]]; then
  BATCH_SIZE=4
  if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" ]]; then
    BATCH_SIZE=16
  fi
fi
if [[ "${WARMUP_RATIO}" == "auto" ]]; then
  WARMUP_RATIO=0
  if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" ]]; then
    WARMUP_RATIO=0.10
  fi
fi
if [[ -n "${LEGACY_MODEL_NAME_OVERRIDE}" && "${LEGACY_MODEL_NAME_OVERRIDE}" != "${MODEL_NAME}" ]]; then
  echo "MODEL_NAME no longer selects a student independently." >&2
  echo "Create or choose a STUDENT_CONFIG with its own model_name and student_id." >&2
  exit 2
fi

TARGETS_ROOT="data/cache/wdc_products/targets"
VALIDATION_TARGETS="${TARGETS_ROOT}/validation.label_only.targets.jsonl"
DIRECT_COST="outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.cost.json"
DIRECT_PREDICTIONS="outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.predictions.jsonl"
RUN_ROOT="${STUDENT_OUTPUT_ROOT}/${STUDENT_ID}/train_${BUDGET}"
RUNTIME_CONTRACT="${RUN_ROOT}/runtime_contract.json"
SNAPSHOT_CONFIG="${RUN_ROOT}/student_config.json"
SUMMARY_ROOT="${STUDENT_OUTPUT_ROOT}/${STUDENT_ID}/summary"
ARTIFACT_ROOT="${STUDENT_OUTPUT_ROOT}/${STUDENT_ID}/artifacts"

variants=(gold_random llm_random llm_active_bucketed_v1)

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

archive_if_exists() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    local archived="${path}.stale.$(date -u +%Y%m%dT%H%M%S%N)"
    mv "${path}" "${archived}"
    echo "archived stale artifact: ${archived}"
  fi
}

ensure_student_config_snapshot() {
  mkdir -p "${RUN_ROOT}"
  if [[ -f "${SNAPSHOT_CONFIG}" ]]; then
    if cmp -s "${STUDENT_CONFIG}" "${SNAPSHOT_CONFIG}"; then
      return
    fi
    if [[ "${FORCE}" != "1" ]]; then
      echo "Existing run uses a different student configuration: ${SNAPSHOT_CONFIG}" >&2
      echo "Use a new student_id/config, or set FORCE=1 to replace the run intentionally." >&2
      exit 1
    fi
    archive_if_exists "${SNAPSHOT_CONFIG}"
  fi
  cp "${STUDENT_CONFIG}" "${SNAPSHOT_CONFIG}.tmp"
  mv "${SNAPSHOT_CONFIG}.tmp" "${SNAPSHOT_CONFIG}"
}

CONTRACT_ARGS=()
RUNTIME_IDENTITY_INITIALIZED=0
RESOLVED_PRECISION=""
RESOLVED_VALIDATION_BATCH_SIZE=""
RUNTIME_DEVICE_NAME=""

initialize_runtime_identity() {
  if [[ "${RUNTIME_IDENTITY_INITIALIZED}" == "1" ]]; then
    return
  fi
  local identity=()
  mapfile -t identity < <(
    "${PYTHON}" -m utils.torch_runtime \
      --device "${DEVICE}" \
      --precision "${PRECISION}" \
      --train-batch-size "${BATCH_SIZE}" \
      --validation-batch-size "${VALIDATION_BATCH_SIZE}"
  )
  if [[ "${#identity[@]}" -ne 3 ]]; then
    echo "Could not resolve the Phase 5 runtime identity." >&2
    exit 1
  fi
  RESOLVED_PRECISION="${identity[0]}"
  RESOLVED_VALIDATION_BATCH_SIZE="${identity[1]}"
  RUNTIME_DEVICE_NAME="${identity[2]}"
  RUNTIME_IDENTITY_INITIALIZED=1
}

load_recorded_runtime_identity() {
  local identity=()
  mapfile -t identity < <(
    "${PYTHON}" -m utils.artifact_contract read-fields \
      --path "${RUNTIME_CONTRACT}" \
      --name resolved_precision \
      --name resolved_validation_batch_size \
      --name runtime_device_name
  )
  if [[ "${#identity[@]}" -ne 3 ]]; then
    echo "Could not read the recorded Phase 5 runtime identity." >&2
    exit 1
  fi
  RESOLVED_PRECISION="${identity[0]}"
  RESOLVED_VALIDATION_BATCH_SIZE="${identity[1]}"
  RUNTIME_DEVICE_NAME="${identity[2]}"
  RUNTIME_IDENTITY_INITIALIZED=1
}

build_runtime_contract_args() {
  initialize_runtime_identity
  CONTRACT_ARGS=(
    --field "stage=runtime"
    --field "student_id=${STUDENT_ID}"
    --field "model_name=${MODEL_NAME}"
    --field "student_architecture=${STUDENT_ARCHITECTURE}"
    --field "device=${DEVICE}"
    --field "precision=${PRECISION}"
    --field "resolved_precision=${RESOLVED_PRECISION}"
    --field "validation_batch_size=${VALIDATION_BATCH_SIZE}"
    --field "resolved_validation_batch_size=${RESOLVED_VALIDATION_BATCH_SIZE}"
    --field "runtime_device_name=${RUNTIME_DEVICE_NAME}"
    --file "runner=scripts/run_phase05_colab.sh"
    --file "student_config=${SNAPSHOT_CONFIG}"
    --file "student_config_schema=models/student_config.py"
    --file "runtime=utils/torch_runtime.py"
  )
}

ensure_run_runtime_contract() {
  build_runtime_contract_args
  if [[ -f "${RUNTIME_CONTRACT}" ]]; then
    if "${PYTHON}" -m utils.artifact_contract check --path "${RUNTIME_CONTRACT}" "${CONTRACT_ARGS[@]}"; then
      return
    fi
    if [[ "${FORCE}" != "1" ]]; then
      echo "Refusing to mix Phase 5 runtime identities in one output root." >&2
      echo "Use a new STUDENT_OUTPUT_ROOT, or set FORCE=1 and rerun every affected variant." >&2
      exit 1
    fi
    archive_if_exists "${RUNTIME_CONTRACT}"
    build_runtime_contract_args
    write_current_contract "${RUNTIME_CONTRACT}"
    return
  fi
  if [[ -d "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 2 -maxdepth 2 \
      \( -name training_summary.json -o -name validation.metrics.json \) \
      -print -quit | grep -q . && [[ "${FORCE}" != "1" ]]; then
    echo "Existing Phase 5 artifacts have no run-level runtime contract." >&2
    echo "Use a new STUDENT_OUTPUT_ROOT, or set FORCE=1 and rerun every affected variant." >&2
    exit 1
  fi
  write_current_contract "${RUNTIME_CONTRACT}"
}

build_training_contract_args() {
  local variant="$1"
  local train_targets="$2"
  initialize_runtime_identity
  ensure_run_runtime_contract
  CONTRACT_ARGS=(
    --field "stage=training"
    --field "git_commit=$(git rev-parse HEAD)"
    --field "variant=${variant}"
    --field "student_id=${STUDENT_ID}"
    --field "model_name=${MODEL_NAME}"
    --field "student_architecture=${STUDENT_ARCHITECTURE}"
    --field "budget=${BUDGET}"
    --field "batch_size=${BATCH_SIZE}"
    --field "validation_batch_size=${VALIDATION_BATCH_SIZE}"
    --field "num_epochs=${NUM_EPOCHS}"
    --field "learning_rate=${LEARNING_RATE}"
    --field "weight_decay=${WEIGHT_DECAY}"
    --field "warmup_steps=${WARMUP_STEPS}"
    --field "warmup_ratio=${WARMUP_RATIO}"
    --field "max_input_length=${MAX_INPUT_LENGTH}"
    --field "early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
    --field "seed=${SEED}"
    --field "device=${DEVICE}"
    --field "precision=${PRECISION}"
    --field "resolved_precision=${RESOLVED_PRECISION}"
    --field "resolved_validation_batch_size=${RESOLVED_VALIDATION_BATCH_SIZE}"
    --field "runtime_device_name=${RUNTIME_DEVICE_NAME}"
    --file "train_targets=${train_targets}"
    --file "validation_targets=${VALIDATION_TARGETS}"
    --file "runner=scripts/run_phase05_colab.sh"
    --file "student_config=${SNAPSHOT_CONFIG}"
    --file "student_config_schema=models/student_config.py"
    --file "train_entrypoint=experiments/train_student.py"
    --file "trainer=experiments/trainer.py"
    --file "runtime=utils/torch_runtime.py"
  )
  if [[ "${STUDENT_ARCHITECTURE}" == "seq2seq" ]]; then
    CONTRACT_ARGS+=(
      --field "max_target_length=${MAX_TARGET_LENGTH}"
      --file "student_backend=models/seq2seq_student.py"
    )
  else
    CONTRACT_ARGS+=(
      --field "pair_truncation=longest_first"
      --field "checkpoint_metric=validation_macro_f1"
      --field "decision_threshold_selection=validation_macro_f1"
      --field "classifier_head_epochs=${CLASSIFIER_HEAD_EPOCHS}"
      --field "classifier_head_learning_rate=${CLASSIFIER_HEAD_LEARNING_RATE}"
      --field "classifier_encoder_learning_rate=${CLASSIFIER_ENCODER_LEARNING_RATE}"
      --field "classifier_unfreeze_last_n_layers=${CLASSIFIER_UNFREEZE_LAST_N_LAYERS}"
      --file "student_backend=models/classification_student.py"
      --file "threshold_selection=utils/classification_threshold.py"
      --file "metrics=utils/metrics.py"
    )
  fi
}

build_evaluation_contract_args() {
  local variant="$1"
  local training_contract="$2"
  initialize_runtime_identity
  CONTRACT_ARGS=(
    --field "stage=evaluation"
    --field "git_commit=$(git rev-parse HEAD)"
    --field "variant=${variant}"
    --field "student_id=${STUDENT_ID}"
    --field "model_name=${MODEL_NAME}"
    --field "student_architecture=${STUDENT_ARCHITECTURE}"
    --field "budget=${BUDGET}"
    --field "eval_batch_size=${EVAL_BATCH_SIZE}"
    --field "max_input_length=${MAX_INPUT_LENGTH}"
    --field "device=${DEVICE}"
    --field "precision=${PRECISION}"
    --field "resolved_precision=${RESOLVED_PRECISION}"
    --field "runtime_device_name=${RUNTIME_DEVICE_NAME}"
    --file "training_contract=${training_contract}"
    --file "validation_targets=${VALIDATION_TARGETS}"
    --file "runner=scripts/run_phase05_colab.sh"
    --file "student_config=${SNAPSHOT_CONFIG}"
    --file "evaluation_entrypoint=experiments/evaluate_student.py"
    --file "metrics=utils/metrics.py"
    --file "runtime=utils/torch_runtime.py"
  )
  if [[ "${STUDENT_ARCHITECTURE}" == "seq2seq" ]]; then
    CONTRACT_ARGS+=(--field "max_new_tokens=${MAX_NEW_TOKENS}")
  else
    CONTRACT_ARGS+=(
      --file "decision_threshold=${RUN_ROOT}/${variant}/best_model/decision_threshold.json"
      --file "threshold_selection=utils/classification_threshold.py"
    )
  fi
}

require_matching_contract() {
  local path="$1"
  local stage="$2"
  if ! "${PYTHON}" -m utils.artifact_contract check --path "${path}" "${CONTRACT_ARGS[@]}"; then
    echo "Refusing to reuse ${stage} artifacts with a missing or mismatched contract." >&2
    echo "Use a different STUDENT_OUTPUT_ROOT, or set FORCE=1 to replace this stage intentionally." >&2
    exit 1
  fi
}

write_current_contract() {
  local path="$1"
  run_cmd "${PYTHON}" -m utils.artifact_contract write --path "${path}" "${CONTRACT_ARGS[@]}"
}

target_for_variant() {
  case "$1" in
    gold_random)
      printf '%s\n' "${TARGETS_ROOT}/train_${BUDGET}.gold_random.targets.jsonl"
      ;;
    llm_random)
      printf '%s\n' "${TARGETS_ROOT}/train_${BUDGET}.llm_random.openai-gpt-5-4-mini.targets.jsonl"
      ;;
    llm_active_bucketed_v1)
      printf '%s\n' "${TARGETS_ROOT}/train_${BUDGET}.llm_active_bucketed_v1.openai-gpt-5-4-mini.targets.jsonl"
      ;;
    *)
      echo "Unknown Phase 5 variant: $1" >&2
      return 2
      ;;
  esac
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_rows() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(wc -l < "${path}")"
  if [[ "${actual}" -ne "${expected}" ]]; then
    echo "Unexpected row count for ${path}: ${actual}; expected ${expected}" >&2
    exit 1
  fi
}

setup_colab() {
  run_cmd "${PYTHON}" -m pip install --quiet -r requirements-colab.txt
  run_cmd "${PYTHON}" -c 'import torch, transformers; print(f"torch={torch.__version__} transformers={transformers.__version__} cuda={torch.cuda.is_available()}")'
}

preflight() {
  if ! command -v git >/dev/null 2>&1; then
    echo "git is required" >&2
    exit 1
  fi
  if ! command -v "${PYTHON}" >/dev/null 2>&1 && [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found: ${PYTHON}" >&2
    exit 1
  fi

  local branch
  branch="$(git branch --show-current)"
  if [[ "${branch}" != "${EXPECTED_BRANCH}" && "${ALLOW_BRANCH_MISMATCH}" != "1" ]]; then
    echo "Expected branch ${EXPECTED_BRANCH}, found ${branch:-detached HEAD}." >&2
    echo "Set ALLOW_BRANCH_MISMATCH=1 only if this checkout intentionally contains the same experiment contract." >&2
    exit 1
  fi

  local variant target
  require_file "${STUDENT_CONFIG}"
  for variant in "${variants[@]}"; do
    target="$(target_for_variant "${variant}")"
    require_file "${target}"
    require_rows "${target}" "${BUDGET}"
  done
  require_file "${VALIDATION_TARGETS}"
  require_rows "${VALIDATION_TARGETS}" 2500
  require_file "${DIRECT_COST}"
  require_file "${DIRECT_PREDICTIONS}"
  require_file "${COST_ASSUMPTIONS}"

  if [[ "${DEVICE}" != cuda* && "${ALLOW_CPU}" != "1" ]]; then
    echo "Phase 5 Colab runs require DEVICE=cuda." >&2
    echo "Set ALLOW_CPU=1 only for an intentional non-Colab smoke check." >&2
    exit 1
  fi
  ensure_student_config_snapshot
  initialize_runtime_identity

  echo "Phase 5 preflight passed: student=${STUDENT_ID} architecture=${STUDENT_ARCHITECTURE} branch=${branch} budget=${BUDGET} device=${RUNTIME_DEVICE_NAME} precision=${RESOLVED_PRECISION}"
}

train_variant() {
  local variant="$1"
  local train_targets output_dir checkpoint summary predictions metrics log_path
  local training_contract evaluation_contract
  train_targets="$(target_for_variant "${variant}")"
  output_dir="${RUN_ROOT}/${variant}"
  checkpoint="${output_dir}/best_model/config.json"
  summary="${output_dir}/training_summary.json"
  predictions="${output_dir}/validation.predictions.jsonl"
  metrics="${output_dir}/validation.metrics.json"
  log_path="${output_dir}/training.log"
  training_contract="${output_dir}/training_contract.json"
  evaluation_contract="${output_dir}/evaluation_contract.json"
  mkdir -p "${output_dir}"
  build_training_contract_args "${variant}" "${train_targets}"

  local threshold_marker="${output_dir}/decision_threshold.json"
  local completion_ready=0
  if [[ -f "${checkpoint}" && -f "${summary}" ]]; then
    completion_ready=1
    if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" && ! -f "${threshold_marker}" ]]; then
      completion_ready=0
    fi
  fi
  if [[ "${FORCE}" != "1" && "${completion_ready}" == "1" ]]; then
    require_matching_contract "${training_contract}" "training"
    echo "skip completed training: ${variant}"
    return
  fi

  archive_if_exists "${training_contract}"
  archive_if_exists "${evaluation_contract}"
  archive_if_exists "${summary}"
  archive_if_exists "${predictions}"
  archive_if_exists "${metrics}"
  archive_if_exists "${threshold_marker}"

  args=(
    "${PYTHON}" -m experiments.train_student
    --student-config "${SNAPSHOT_CONFIG}"
    --train-targets "${train_targets}"
    --validation-targets "${VALIDATION_TARGETS}"
    --output-dir "${output_dir}"
    --batch-size "${BATCH_SIZE}"
    --num-epochs "${NUM_EPOCHS}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-steps "${WARMUP_STEPS}"
    --warmup-ratio "${WARMUP_RATIO}"
    --max-input-length "${MAX_INPUT_LENGTH}"
    --max-target-length "${MAX_TARGET_LENGTH}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --precision "${PRECISION}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  )
  if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" ]]; then
    args+=(
      --classifier-head-epochs "${CLASSIFIER_HEAD_EPOCHS}"
      --classifier-head-learning-rate "${CLASSIFIER_HEAD_LEARNING_RATE}"
      --classifier-encoder-learning-rate "${CLASSIFIER_ENCODER_LEARNING_RATE}"
      --classifier-unfreeze-last-n-layers "${CLASSIFIER_UNFREEZE_LAST_N_LAYERS}"
    )
  fi
  if [[ "${VALIDATION_BATCH_SIZE}" != "auto" ]]; then
    args+=(--validation-batch-size "${VALIDATION_BATCH_SIZE}")
  fi
  if [[ "${USE_WANDB}" == "1" ]]; then
    args+=(--use-wandb)
  fi

  printf '+' | tee "${log_path}"
  printf ' %q' "${args[@]}" | tee -a "${log_path}"
  printf '\n' | tee -a "${log_path}"
  "${args[@]}" 2>&1 | tee -a "${log_path}"
  build_training_contract_args "${variant}" "${train_targets}"
  write_current_contract "${training_contract}"
}

evaluate_variant() {
  local variant="$1"
  local output_dir checkpoint predictions metrics log_path
  local training_contract evaluation_contract
  output_dir="${RUN_ROOT}/${variant}"
  checkpoint="${output_dir}/best_model"
  predictions="${output_dir}/validation.predictions.jsonl"
  metrics="${output_dir}/validation.metrics.json"
  log_path="${output_dir}/evaluation.log"
  training_contract="${output_dir}/training_contract.json"
  evaluation_contract="${output_dir}/evaluation_contract.json"
  require_file "${checkpoint}/config.json"
  require_file "${training_contract}"
  if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" ]]; then
    require_file "${checkpoint}/decision_threshold.json"
    require_file "${output_dir}/decision_threshold.json"
  fi
  build_evaluation_contract_args "${variant}" "${training_contract}"

  if [[ "${FORCE}" != "1" && -f "${predictions}" && -f "${metrics}" ]]; then
    require_matching_contract "${evaluation_contract}" "evaluation"
    echo "skip completed evaluation: ${variant}"
    return
  fi

  archive_if_exists "${evaluation_contract}"
  archive_if_exists "${predictions}"
  archive_if_exists "${metrics}"

  args=(
    "${PYTHON}" -m experiments.evaluate_student
    --student-config "${SNAPSHOT_CONFIG}"
    --checkpoint "${checkpoint}"
    --input "${VALIDATION_TARGETS}"
    --predictions "${predictions}"
    --metrics "${metrics}"
    --variant "${variant}"
    --budget "${BUDGET}"
    --split validation
    --batch-size "${EVAL_BATCH_SIZE}"
    --max-input-length "${MAX_INPUT_LENGTH}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --device "${DEVICE}"
    --precision "${PRECISION}"
  )

  printf '+' | tee "${log_path}"
  printf ' %q' "${args[@]}" | tee -a "${log_path}"
  printf '\n' | tee -a "${log_path}"
  "${args[@]}" 2>&1 | tee -a "${log_path}"
  build_evaluation_contract_args "${variant}" "${training_contract}"
  write_current_contract "${evaluation_contract}"
}

run_variant() {
  train_variant "$1"
  evaluate_variant "$1"
  aggregate --allow-partial
}

run_selected() {
  local selected="${1:-all}"
  local variant
  if [[ "${selected}" == "all" ]]; then
    for variant in "${variants[@]}"; do
      run_variant "${variant}"
    done
  else
    target_for_variant "${selected}" >/dev/null
    run_variant "${selected}"
  fi
}

aggregate() {
  run_cmd "${PYTHON}" -m experiments.aggregate_phase05_results \
    --output-root "${STUDENT_OUTPUT_ROOT}" \
    --student-run-root "${RUN_ROOT}" \
    --targets-root "${TARGETS_ROOT}" \
    --direct-cost "${DIRECT_COST}" \
    --cost-assumptions "${COST_ASSUMPTIONS}" \
    --budget "${BUDGET}" \
    --json "${SUMMARY_ROOT}/phase05_train_${BUDGET}.pilot.json" \
    --csv "${SUMMARY_ROOT}/phase05_train_${BUDGET}.pilot.csv" \
    --cost-csv "${SUMMARY_ROOT}/phase05_train_${BUDGET}.cost_scenarios.csv" \
    "$@"
}

write_manifest() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  {
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch=%s\n' "$(git branch --show-current)"
    printf 'student_id=%s\n' "${STUDENT_ID}"
    printf 'model_name=%s\n' "${MODEL_NAME}"
    printf 'student_architecture=%s\n' "${STUDENT_ARCHITECTURE}"
    printf 'student_config=%s\n' "${STUDENT_CONFIG}"
    printf 'budget=%s\n' "${BUDGET}"
    printf 'seed=%s\n' "${SEED}"
    printf 'batch_size=%s\n' "${BATCH_SIZE}"
    printf 'validation_batch_size=%s\n' "${VALIDATION_BATCH_SIZE}"
    printf 'eval_batch_size=%s\n' "${EVAL_BATCH_SIZE}"
    printf 'num_epochs=%s\n' "${NUM_EPOCHS}"
    printf 'learning_rate=%s\n' "${LEARNING_RATE}"
    printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
    printf 'warmup_steps=%s\n' "${WARMUP_STEPS}"
    printf 'warmup_ratio=%s\n' "${WARMUP_RATIO}"
    if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" ]]; then
      printf 'classifier_head_epochs=%s\n' "${CLASSIFIER_HEAD_EPOCHS}"
      printf 'classifier_head_learning_rate=%s\n' "${CLASSIFIER_HEAD_LEARNING_RATE}"
      printf 'classifier_encoder_learning_rate=%s\n' "${CLASSIFIER_ENCODER_LEARNING_RATE}"
      printf 'classifier_unfreeze_last_n_layers=%s\n' "${CLASSIFIER_UNFREEZE_LAST_N_LAYERS}"
    fi
    printf 'device=%s\n' "${DEVICE}"
    printf 'precision=%s\n' "${PRECISION}"
    printf 'resolved_precision=%s\n' "${RESOLVED_PRECISION}"
    printf 'resolved_validation_batch_size=%s\n' "${RESOLVED_VALIDATION_BATCH_SIZE}"
    printf 'runtime_device_name=%s\n' "${RUNTIME_DEVICE_NAME}"
    if [[ "${STUDENT_ARCHITECTURE}" == "seq2seq" ]]; then
      printf 'max_target_length=%s\n' "${MAX_TARGET_LENGTH}"
      printf 'max_new_tokens=%s\n' "${MAX_NEW_TOKENS}"
    fi
    printf 'cost_assumptions=%s\n' "${COST_ASSUMPTIONS}"
  } > "${path}"
}

validate_training_contracts() {
  local variant train_targets variant_dir training_contract
  require_file "${RUNTIME_CONTRACT}"
  load_recorded_runtime_identity
  build_runtime_contract_args
  require_matching_contract "${RUNTIME_CONTRACT}" "run-level runtime"
  for variant in "${variants[@]}"; do
    train_targets="$(target_for_variant "${variant}")"
    variant_dir="${RUN_ROOT}/${variant}"
    training_contract="${variant_dir}/training_contract.json"
    require_file "${training_contract}"
    build_training_contract_args "${variant}" "${train_targets}"
    require_matching_contract "${training_contract}" "training"
  done
}

validate_packaged_stage_contracts() {
  local variant variant_dir training_contract evaluation_contract
  validate_training_contracts
  for variant in "${variants[@]}"; do
    variant_dir="${RUN_ROOT}/${variant}"
    training_contract="${variant_dir}/training_contract.json"
    evaluation_contract="${variant_dir}/evaluation_contract.json"
    require_file "${evaluation_contract}"
    build_evaluation_contract_args "${variant}" "${training_contract}"
    require_matching_contract "${evaluation_contract}" "evaluation"
  done
}

package_results() {
  validate_packaged_stage_contracts
  aggregate
  mkdir -p "${ARTIFACT_ROOT}"
  local stage archive variant variant_dir
  stage="$(mktemp -d)"
  trap 'rm -rf "${stage}"' EXIT
  archive="${ARTIFACT_ROOT}/phase05_${STUDENT_ID}_train_${BUDGET}_results.tar.gz"
  mkdir -p "${stage}/phase05_results/students" "${stage}/phase05_results/direct_llm" \
    "${stage}/phase05_results/summary" "${stage}/phase05_results/targets"

  for variant in "${variants[@]}"; do
    variant_dir="${RUN_ROOT}/${variant}"
    require_file "${variant_dir}/training_summary.json"
    require_file "${variant_dir}/training_contract.json"
    require_file "${variant_dir}/evaluation_contract.json"
    require_file "${variant_dir}/validation.predictions.jsonl"
    require_file "${variant_dir}/validation.metrics.json"
    if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" ]]; then
      require_file "${variant_dir}/decision_threshold.json"
    fi
    mkdir -p "${stage}/phase05_results/students/${variant}"
    cp "${variant_dir}/training_summary.json" \
      "${variant_dir}/training_contract.json" \
      "${variant_dir}/evaluation_contract.json" \
      "${variant_dir}/training.log" \
      "${variant_dir}/evaluation.log" \
      "${variant_dir}/validation.predictions.jsonl" \
      "${variant_dir}/validation.metrics.json" \
      "${stage}/phase05_results/students/${variant}/"
    if [[ "${STUDENT_ARCHITECTURE}" == "sequence_classification" ]]; then
      cp "${variant_dir}/decision_threshold.json" \
        "${stage}/phase05_results/students/${variant}/"
    fi
    cp "$(target_for_variant "${variant}")" "${stage}/phase05_results/targets/"
  done

  cp "${SUMMARY_ROOT}/phase05_train_${BUDGET}.pilot.json" \
    "${SUMMARY_ROOT}/phase05_train_${BUDGET}.pilot.csv" \
    "${SUMMARY_ROOT}/phase05_train_${BUDGET}.cost_scenarios.csv" \
    "${stage}/phase05_results/summary/"
  cp "${COST_ASSUMPTIONS}" "${stage}/phase05_results/summary/"
  cp "${DIRECT_COST}" "${DIRECT_PREDICTIONS}" "${stage}/phase05_results/direct_llm/"
  cp "${VALIDATION_TARGETS}" "${stage}/phase05_results/targets/"
  cp "${SNAPSHOT_CONFIG}" "${stage}/phase05_results/student_config.json"
  cp "${RUNTIME_CONTRACT}" "${stage}/phase05_results/"
  write_manifest "${stage}/phase05_results/run_manifest.txt"
  tar -czf "${archive}" -C "${stage}" phase05_results
  rm -rf "${stage}"
  trap - EXIT
  echo "Results archive ready: ${archive}"
}

package_checkpoints() {
  validate_training_contracts
  mkdir -p "${ARTIFACT_ROOT}"
  local variant checkpoint archive
  for variant in "${variants[@]}"; do
    checkpoint="${RUN_ROOT}/${variant}/best_model"
    require_file "${checkpoint}/config.json"
    archive="${ARTIFACT_ROOT}/phase05_${STUDENT_ID}_train_${BUDGET}_${variant}_checkpoint.tar.gz"
    tar -czf "${archive}" -C "${RUN_ROOT}" \
      "${variant}/best_model" \
      "${variant}/training_contract.json" \
      student_config.json \
      runtime_contract.json
    echo "Checkpoint archive ready: ${archive}"
  done
}

command="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command}" in
  setup)
    setup_colab
    ;;
  preflight)
    preflight
    ;;
  run)
    preflight
    run_selected "${1:-all}"
    ;;
  aggregate)
    aggregate "$@"
    ;;
  package-results)
    package_results
    ;;
  package-checkpoints)
    package_checkpoints
    ;;
  all)
    preflight
    run_selected all
    aggregate
    package_results
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
