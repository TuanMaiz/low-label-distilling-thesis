#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the three-fold WDC Qwen out-of-fold workflow.

Usage:
  bash scripts/run_wdc_qwen_oof.sh verify-data
  bash scripts/run_wdc_qwen_oof.sh plan
  bash scripts/run_wdc_qwen_oof.sh setup
  bash scripts/run_wdc_qwen_oof.sh preflight
  bash scripts/run_wdc_qwen_oof.sh train-fold 1 --confirm-oof-training
  bash scripts/run_wdc_qwen_oof.sh train-fold 2 --confirm-oof-training
  bash scripts/run_wdc_qwen_oof.sh train-fold 3 --confirm-oof-training
  bash scripts/run_wdc_qwen_oof.sh merge
  bash scripts/run_wdc_qwen_oof.sh package-results
  bash scripts/run_wdc_qwen_oof.sh run-all --confirm-oof-training

The fold inputs are committed. This workflow makes no LLM calls and does not
prepare or regenerate folds on the rented GPU.

Environment overrides:
  PYTHON=python
  OUTPUT_ROOT=outputs/uncertainty_refinement/wdc-qwen-oof
  EXPECTED_GPU_SUBSTRING=3090
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT:-outputs/uncertainty_refinement/wdc-qwen-oof}")"
EXPECTED_GPU_SUBSTRING="${EXPECTED_GPU_SUBSTRING:-3090}"
OOF_DIR="data/cache/wdc_products/oof/majority-3fold"
STUDENT_CONFIG="configs/students/qwen3_reranker_0_6b.json"
EXPECTED_COUNT=2500
RUNTIME_IDENTITY="${OUTPUT_ROOT}/runtime-identity.json"
MERGED_CSV="${OUTPUT_ROOT}/oof_predictions.csv"
MERGED_SUMMARY="${OUTPUT_ROOT}/oof_summary.json"

run_cmd() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

fold_name() {
  case "$1" in
    1) printf '%s\n' fold_01 ;;
    2) printf '%s\n' fold_02 ;;
    3) printf '%s\n' fold_03 ;;
    *) echo "Fold must be 1, 2, or 3: $1" >&2; exit 2 ;;
  esac
}

require_clean_committed_inputs() {
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked files differ from the checked-out commit; commit or restore them first." >&2
    exit 1
  fi
  require_file "${OOF_DIR}/oof_manifest.json"
  require_file "${OOF_DIR}/fold_assignments.csv"
  require_file "${STUDENT_CONFIG}"
  run_cmd git ls-files --error-unmatch \
    "${OOF_DIR}/oof_manifest.json" \
    "${OOF_DIR}/fold_assignments.csv" \
    "${STUDENT_CONFIG}" \
    "scripts/run_wdc_qwen_oof.sh" \
    "supervision/prepare_oof_folds.py" \
    "experiments/wdc_qwen_oof.py"
  local fold
  for fold in fold_01 fold_02 fold_03; do
    run_cmd git ls-files --error-unmatch \
      "${OOF_DIR}/${fold}/train.jsonl" \
      "${OOF_DIR}/${fold}/val.jsonl" \
      "${OOF_DIR}/${fold}/test-heldout.jsonl" \
      "${OOF_DIR}/${fold}/manifest.json"
  done
}

verify_data() {
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_oof verify-data \
    --oof-dir "${OOF_DIR}" \
    --student-config "${STUDENT_CONFIG}" \
    --expected-count "${EXPECTED_COUNT}"
}

render_plan() {
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_oof plan \
    --oof-dir "${OOF_DIR}" \
    --student-config "${STUDENT_CONFIG}" \
    --output-root "${OUTPUT_ROOT}" \
    --python-executable "${PYTHON}" \
    --expected-count "${EXPECTED_COUNT}"
}

setup_runtime() {
  run_cmd "${PYTHON}" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"; print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})'
  run_cmd "${PYTHON}" -m pip install --quiet -r requirements-colab.txt
  run_cmd "${PYTHON}" -m utils.peft_runtime sanitize
  run_cmd "${PYTHON}" -m utils.peft_runtime check
}

preflight() {
  require_clean_committed_inputs
  verify_data
  run_cmd "${PYTHON}" -m utils.peft_runtime sanitize
  run_cmd "${PYTHON}" -m utils.peft_runtime check
  mkdir -p "${OUTPUT_ROOT}"
  local candidate="${RUNTIME_IDENTITY}.candidate"
  if [[ -e "${candidate}" ]]; then
    echo "Unresolved runtime identity candidate exists: ${candidate}" >&2
    exit 1
  fi
  "${PYTHON}" -m experiments.wdc_qwen_oof preflight \
    --oof-dir "${OOF_DIR}" \
    --student-config "${STUDENT_CONFIG}" \
    --expected-count "${EXPECTED_COUNT}" \
    --expected-gpu-substring "${EXPECTED_GPU_SUBSTRING}" > "${candidate}"
  if [[ -f "${RUNTIME_IDENTITY}" ]]; then
    if ! cmp -s "${candidate}" "${RUNTIME_IDENTITY}"; then
      echo "Runtime identity differs; candidate retained at ${candidate}." >&2
      exit 1
    fi
    rm "${candidate}"
  else
    mv "${candidate}" "${RUNTIME_IDENTITY}"
  fi
  echo "WDC Qwen OOF GPU preflight passed."
}

fold_root() {
  printf '%s/%s\n' "${OUTPUT_ROOT}" "$1"
}

fold_state() {
  local root="$1"
  if [[ -f "${root}/test-heldout.predictions.jsonl" \
      && -f "${root}/test-heldout.metrics.json" ]]; then
    printf '%s\n' complete
  elif [[ ! -e "${root}/train" ]]; then
    printf '%s\n' empty
  elif [[ -f "${root}/train/training_summary.json" \
      && -f "${root}/train/checkpoint_manifest.json" ]]; then
    printf '%s\n' trained
  else
    printf '%s\n' partial
  fi
}

write_or_check_fold_contract() {
  local fold="$1"
  local root="$2"
  local mode="${3:-check}"
  local contract="${root}/artifact-contract.json"
  local commit
  commit="$(git rev-parse HEAD)"
  local args=(
    --field "stage=wdc_qwen_oof"
    --field "dataset_id=wdc_products_80cc_small_100un"
    --field "student_id=qwen3-reranker-0-6b"
    --field "fold=${fold}"
    --field "git_commit=${commit}"
    --field "inner_validation_percent=20"
    --field "num_epochs=10"
    --field "batch_size=1"
    --field "gradient_accumulation_steps=16"
    --field "learning_rate=2e-4"
    --field "weight_decay=0.01"
    --field "warmup_ratio=0.10"
    --field "test_scope=locked"
    --file "student_config=${STUDENT_CONFIG}"
    --file "oof_manifest=${OOF_DIR}/oof_manifest.json"
    --file "fold_manifest=${OOF_DIR}/${fold}/manifest.json"
    --file "train=${OOF_DIR}/${fold}/train.jsonl"
    --file "validation=${OOF_DIR}/${fold}/val.jsonl"
    --file "held_out=${OOF_DIR}/${fold}/test-heldout.jsonl"
    --file "runtime_identity=${RUNTIME_IDENTITY}"
    --file "runner=scripts/run_wdc_qwen_oof.sh"
    --file "workflow=experiments/wdc_qwen_oof.py"
    --file "trainer=experiments/train_student.py"
    --file "trainer_core=experiments/trainer.py"
    --file "evaluator=experiments/evaluate_student.py"
  )
  mkdir -p "${root}"
  if [[ -f "${contract}" ]]; then
    run_cmd "${PYTHON}" -m utils.artifact_contract check \
      --path "${contract}" "${args[@]}"
  elif [[ "${mode}" == write ]]; then
    run_cmd "${PYTHON}" -m utils.artifact_contract write \
      --path "${contract}" "${args[@]}"
  else
    echo "Fold artifact contract is missing: ${contract}" >&2
    exit 1
  fi
}

verify_training() {
  local fold="$1"
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_oof verify-training \
    --oof-dir "${OOF_DIR}" \
    --student-config "${STUDENT_CONFIG}" \
    --results-root "${OUTPUT_ROOT}" \
    --fold "${fold}" \
    --expected-count "${EXPECTED_COUNT}"
}

verify_fold() {
  local fold="$1"
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_oof verify-fold \
    --oof-dir "${OOF_DIR}" \
    --results-root "${OUTPUT_ROOT}" \
    --fold "${fold}"
}

evaluate_fold() {
  local fold="$1"
  local root
  root="$(fold_root "${fold}")"
  run_cmd "${PYTHON}" -m experiments.evaluate_student \
    --student-config "${STUDENT_CONFIG}" \
    --checkpoint "${root}/train/best_model" \
    --input "${OOF_DIR}/${fold}/test-heldout.jsonl" \
    --predictions "${root}/test-heldout.predictions.jsonl" \
    --metrics "${root}/test-heldout.metrics.json" \
    --variant "oof_${fold}" \
    --budget full \
    --split oof_test_heldout \
    --batch-size 1 \
    --max-input-length 4096 \
    --precision auto \
    --device cuda
  verify_fold "${fold}"
}

train_fold() {
  local fold_number="$1"
  local confirmation="${2:-}"
  if [[ "${confirmation}" != --confirm-oof-training ]]; then
    echo "OOF training requires --confirm-oof-training." >&2
    exit 2
  fi
  local fold
  local root
  local state
  fold="$(fold_name "${fold_number}")"
  root="$(fold_root "${fold}")"
  preflight
  state="$(fold_state "${root}")"
  if [[ "${state}" == complete ]]; then
    write_or_check_fold_contract "${fold}" "${root}" check
    verify_training "${fold}"
    verify_fold "${fold}"
    echo "Existing completed ${fold} verified; nothing to rerun."
    return
  fi
  if [[ "${state}" == partial ]]; then
    echo "Incomplete ${fold} output requires inspection: ${root}" >&2
    exit 1
  fi
  if [[ "${state}" == trained ]]; then
    write_or_check_fold_contract "${fold}" "${root}" check
    verify_training "${fold}"
    evaluate_fold "${fold}"
    echo "Recovered ${fold} from its verified checkpoint without retraining."
    return
  fi

  write_or_check_fold_contract "${fold}" "${root}" write
  run_cmd "${PYTHON}" -m experiments.train_student \
    --student-config "${STUDENT_CONFIG}" \
    --train-targets "${OOF_DIR}/${fold}/train.jsonl" \
    --validation-targets "${OOF_DIR}/${fold}/val.jsonl" \
    --output-dir "${root}/train" \
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
  verify_training "${fold}"
  evaluate_fold "${fold}"
  echo "WDC Qwen ${fold} OOF training and held-out prediction passed."
}

merge_results() {
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_oof merge \
    --oof-dir "${OOF_DIR}" \
    --results-root "${OUTPUT_ROOT}" \
    --output-csv "${MERGED_CSV}" \
    --output-summary "${MERGED_SUMMARY}" \
    --expected-count "${EXPECTED_COUNT}"
  run_cmd "${PYTHON}" -m experiments.wdc_qwen_oof verify-merged \
    --oof-dir "${OOF_DIR}" \
    --results-root "${OUTPUT_ROOT}" \
    --output-csv "${MERGED_CSV}" \
    --output-summary "${MERGED_SUMMARY}" \
    --expected-count "${EXPECTED_COUNT}"
}

package_results() {
  merge_results
  local archive="${OUTPUT_ROOT}/wdc-qwen-oof-results.tar.gz"
  local checksum="${archive}.sha256"
  if [[ -f "${archive}" || -f "${checksum}" ]]; then
    require_file "${archive}"
    require_file "${checksum}"
    (cd "${OUTPUT_ROOT}" && run_cmd sha256sum -c "$(basename "${checksum}")")
    echo "Existing OOF result package verified; nothing to rebuild."
    return
  fi
  local members=(oof_predictions.csv oof_summary.json runtime-identity.json)
  local fold
  for fold in fold_01 fold_02 fold_03; do
    members+=(
      "${fold}/artifact-contract.json"
      "${fold}/test-heldout.predictions.jsonl"
      "${fold}/test-heldout.metrics.json"
      "${fold}/train/training_summary.json"
      "${fold}/train/checkpoint_manifest.json"
      "${fold}/train/decision_threshold.json"
    )
  done
  run_cmd tar -C "${OUTPUT_ROOT}" -czf "${archive}" "${members[@]}"
  (cd "${OUTPUT_ROOT}" && sha256sum "$(basename "${archive}")" > "$(basename "${checksum}")")
  (cd "${OUTPUT_ROOT}" && run_cmd sha256sum -c "$(basename "${checksum}")")
  echo "Packaged non-weight OOF result evidence at ${archive}."
}

run_all() {
  local confirmation="${1:-}"
  if [[ "${confirmation}" != --confirm-oof-training ]]; then
    echo "OOF training requires --confirm-oof-training." >&2
    exit 2
  fi
  train_fold 1 "${confirmation}"
  train_fold 2 "${confirmation}"
  train_fold 3 "${confirmation}"
  package_results
}

case "${1:-}" in
  verify-data)
    verify_data
    ;;
  plan)
    render_plan
    ;;
  setup)
    setup_runtime
    ;;
  preflight)
    preflight
    ;;
  train-fold)
    train_fold "${2:-}" "${3:-}"
    ;;
  merge)
    merge_results
    ;;
  package-results)
    package_results
    ;;
  run-all)
    run_all "${2:-}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
