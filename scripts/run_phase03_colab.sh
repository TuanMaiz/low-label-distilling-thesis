#!/usr/bin/env bash
set -euo pipefail

# Colab-friendly Phase 03 runner.
#
# Usage:
#   bash scripts/run_phase03_colab.sh label_only 128
#   bash scripts/run_phase03_colab.sh structured_rationale 128
#
# Optional environment variables:
#   OUTPUT_ROOT=/content/drive/MyDrive/luan-van/outputs/phase03_v2
#   MODEL_NAME=google/flan-t5-base
#   MODEL_SLUG=flan-t5-base
#   NUM_EPOCHS=50
#   BATCH_SIZE=4
#   LEARNING_RATE=3e-4
#   EARLY_STOPPING_PATIENCE=8
#   INSTALL_DEPS=1

VARIANT="${1:-label_only}"
NOMINAL_BUDGET="${2:-128}"
EVAL_SPLIT="${3:-validation}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_NAME="${MODEL_NAME:-google/flan-t5-base}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL_NAME##*/}}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-512}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

if [[ "${VARIANT}" == "structured_rationale" || "${VARIANT}" == "free_text" ]]; then
  MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-128}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
else
  MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-16}"
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
fi

if [[ -d "/content/drive/MyDrive" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/luan-van/outputs/phase03_v2}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/phase03_v2}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

nvidia-smi || true
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

TRAIN_TARGETS="data/cache/wdc_products/targets/train_${NOMINAL_BUDGET}.${VARIANT}.targets.jsonl"
LABEL_VALIDATION_TARGETS="data/cache/wdc_products/targets/validation.label_only.targets.jsonl"
EVAL_INPUT="data/cache/wdc_products/serialized/${EVAL_SPLIT}.jsonl"

if [[ ! -f "${TRAIN_TARGETS}" ]]; then
  echo "Missing training targets: ${TRAIN_TARGETS}" >&2
  echo "Pull the branch with Phase 03 artifacts or generate targets before training." >&2
  exit 1
fi
if [[ ! -f "${LABEL_VALIDATION_TARGETS}" ]]; then
  echo "Missing validation targets: ${LABEL_VALIDATION_TARGETS}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_INPUT}" ]]; then
  echo "Missing evaluation input: ${EVAL_INPUT}" >&2
  exit 1
fi

TRAIN_ROWS="$(wc -l < "${TRAIN_TARGETS}" | tr -d ' ')"
if [[ "${VARIANT}" == "label_only" ]]; then
  VALIDATION_TARGETS="${LABEL_VALIDATION_TARGETS}"
else
  # No held-out rationale targets exist yet, so choose checkpoints using the
  # same target format as training and evaluate final decisions separately.
  VALIDATION_TARGETS="${TRAIN_TARGETS}"
fi
RUN_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}/train_${NOMINAL_BUDGET}/${VARIANT}"

mkdir -p "${RUN_DIR}"

echo "Training ${VARIANT} model=${MODEL_NAME} nominal_budget=${NOMINAL_BUDGET} train_rows=${TRAIN_ROWS}"
"${PYTHON_BIN}" -m experiments.train_mt5 \
  --train-targets "${TRAIN_TARGETS}" \
  --validation-targets "${VALIDATION_TARGETS}" \
  --output-dir "${RUN_DIR}" \
  --model-name "${MODEL_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-input-length "${MAX_INPUT_LENGTH}" \
  --max-target-length "${MAX_TARGET_LENGTH}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"

echo "Evaluating ${VARIANT} on ${EVAL_SPLIT}"
"${PYTHON_BIN}" -m experiments.evaluate_student \
  --checkpoint "${RUN_DIR}/best_model" \
  --input "${EVAL_INPUT}" \
  --predictions "${RUN_DIR}/${EVAL_SPLIT}_predictions.jsonl" \
  --metrics "${RUN_DIR}/${EVAL_SPLIT}_metrics.json" \
  --summary-csv "${OUTPUT_ROOT}/${MODEL_SLUG}/${EVAL_SPLIT}_summary.csv" \
  --variant "${VARIANT}" \
  --budget "${TRAIN_ROWS}" \
  --split "${EVAL_SPLIT}" \
  --batch-size "${BATCH_SIZE}" \
  --max-input-length "${MAX_INPUT_LENGTH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

echo "Done. Outputs saved under: ${RUN_DIR}"
