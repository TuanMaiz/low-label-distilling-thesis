#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Reproduce Phase 3 artifacts.

Usage:
  scripts/run_phase03_reproducible.sh <command>

Safe local commands:
  manifests       Regenerate fixed train_128 random and active manifests.
  validate        Validate any Phase 3 caches that already exist.
  test            Run the unit test suite.
  local           Run manifests + test. No LLM calls.

Live OpenRouter commands:
  teacher-random  Generate teacher labels for train_128.random.
  teacher-active  Generate teacher labels for the active train_128 manifest.
  teacher-all     Generate teacher labels for both manifests.
  direct          Run direct LLM matcher on the validation split/sample.
  all-live        Run manifests + teacher-all + direct + validate.

Environment overrides:
  PYTHON=.venv/bin/python
  BUDGET=128
  SEED=42
  MODEL=openai/gpt-4o-mini
  TEMPERATURE=0.0
  PROMPT_VERSION=answer_only_v1
  ACTIVE_STRATEGY=llm_active_bucketed_v1
  EASY_MATCH_RATIO=0.25
  HARD_MATCH_RATIO=0.25
  EASY_NON_MATCH_RATIO=0.25
  HARD_NEGATIVE_RATIO=0.25
  DIRECT_INPUT=data/cache/wdc_products/serialized/validation.jsonl
  DIRECT_LIMIT=100   # optional fixed sample size; omit for full split

Live commands require OPENROUTER_API_KEY in the environment or .env.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-.venv/bin/python}"
BUDGET="${BUDGET:-128}"
SEED="${SEED:-42}"
MODEL="${MODEL:-${OPENROUTER_MODEL:-openai/gpt-4o-mini}}"
TEMPERATURE="${TEMPERATURE:-0.0}"
PROMPT_VERSION="${PROMPT_VERSION:-answer_only_v1}"
ACTIVE_STRATEGY="${ACTIVE_STRATEGY:-llm_active_bucketed_v1}"
EASY_MATCH_RATIO="${EASY_MATCH_RATIO:-0.25}"
HARD_MATCH_RATIO="${HARD_MATCH_RATIO:-0.25}"
EASY_NON_MATCH_RATIO="${EASY_NON_MATCH_RATIO:-0.25}"
HARD_NEGATIVE_RATIO="${HARD_NEGATIVE_RATIO:-0.25}"
DIRECT_INPUT="${DIRECT_INPUT:-data/cache/wdc_products/serialized/validation.jsonl}"
DIRECT_LIMIT="${DIRECT_LIMIT:-}"

LOW_LABEL_INPUT="data/cache/wdc_products/low_label/train_${BUDGET}.jsonl"
TRAIN_INPUT="data/cache/wdc_products/serialized/train.jsonl"
RANDOM_MANIFEST="data/cache/wdc_products/selection_manifests/train_${BUDGET}.random.jsonl"
ACTIVE_MANIFEST="data/cache/wdc_products/selection_manifests/train_${BUDGET}.${ACTIVE_STRATEGY}.jsonl"
RANDOM_LABELS="data/cache/wdc_products/teacher_labels/train_${BUDGET}.random.openrouter.${PROMPT_VERSION}.labels.jsonl"
RANDOM_REJECTS="data/cache/wdc_products/teacher_labels/train_${BUDGET}.random.openrouter.${PROMPT_VERSION}.rejects.jsonl"
ACTIVE_LABELS="data/cache/wdc_products/teacher_labels/train_${BUDGET}.${ACTIVE_STRATEGY}.openrouter.${PROMPT_VERSION}.labels.jsonl"
ACTIVE_REJECTS="data/cache/wdc_products/teacher_labels/train_${BUDGET}.${ACTIVE_STRATEGY}.openrouter.${PROMPT_VERSION}.rejects.jsonl"
direct_split="$(basename "${DIRECT_INPUT}" .jsonl)"
DIRECT_PREDICTIONS="outputs/distiller_wdc/direct_llm/${direct_split}.openrouter.${PROMPT_VERSION}.predictions.jsonl"

require_python() {
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON}" >&2
    echo "Set PYTHON=.venv/bin/python or activate the project environment." >&2
    exit 1
  fi
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

build_manifests() {
  require_python
  run_cmd "${PYTHON}" -m data.select_active_pairs \
    --input "${LOW_LABEL_INPUT}" \
    --budget "${BUDGET}" \
    --strategy random \
    --seed "${SEED}"

  active_args=(
    "${PYTHON}" -m data.select_active_pairs
    --input "${TRAIN_INPUT}" \
    --budget "${BUDGET}" \
    --strategy "${ACTIVE_STRATEGY}" \
    --seed "${SEED}"
  )
  if [[ "${ACTIVE_STRATEGY}" == "llm_active_bucketed_v1" ]]; then
    active_args+=(
      --easy-match-ratio "${EASY_MATCH_RATIO}"
      --hard-match-ratio "${HARD_MATCH_RATIO}"
      --easy-non-match-ratio "${EASY_NON_MATCH_RATIO}"
      --hard-negative-ratio "${HARD_NEGATIVE_RATIO}"
    )
  fi
  run_cmd "${active_args[@]}"
}

teacher_random() {
  require_python
  run_cmd "${PYTHON}" -m supervision.generate_teacher_labels \
    --pairs "${RANDOM_MANIFEST}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --prompt-version "${PROMPT_VERSION}" \
    --seed "${SEED}"
}

teacher_active() {
  require_python
  run_cmd "${PYTHON}" -m supervision.generate_teacher_labels \
    --pairs "${ACTIVE_MANIFEST}" \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --prompt-version "${PROMPT_VERSION}" \
    --seed "${SEED}"
}

direct_matcher() {
  require_python
  args=(
    "${PYTHON}" -m supervision.direct_llm_matcher
    --input "${DIRECT_INPUT}"
    --model "${MODEL}"
    --temperature "${TEMPERATURE}"
    --prompt-version "${PROMPT_VERSION}"
    --sample-seed "${SEED}"
  )
  if [[ -n "${DIRECT_LIMIT}" ]]; then
    args+=(--limit "${DIRECT_LIMIT}")
  fi
  run_cmd "${args[@]}"
}

validate_if_exists() {
  local path="$1"
  local mode="$2"
  if [[ -f "${path}" ]]; then
    run_cmd "${PYTHON}" -m supervision.validate_teacher_labels --cache "${path}" --mode "${mode}"
  else
    echo "skip missing cache: ${path}"
  fi
}

validate_caches() {
  require_python
  validate_if_exists "${RANDOM_LABELS}" teacher_label
  validate_if_exists "${RANDOM_REJECTS}" teacher_label
  validate_if_exists "${ACTIVE_LABELS}" teacher_label
  validate_if_exists "${ACTIVE_REJECTS}" teacher_label
  validate_if_exists "${DIRECT_PREDICTIONS}" direct_prediction
}

run_tests() {
  require_python
  run_cmd "${PYTHON}" -m unittest discover -s tests
}

command="${1:-}"
case "${command}" in
  manifests)
    build_manifests
    ;;
  validate)
    validate_caches
    ;;
  test)
    run_tests
    ;;
  local)
    build_manifests
    run_tests
    ;;
  teacher-random)
    teacher_random
    ;;
  teacher-active)
    teacher_active
    ;;
  teacher-all)
    teacher_random
    teacher_active
    ;;
  direct)
    direct_matcher
    ;;
  all-live)
    build_manifests
    teacher_random
    teacher_active
    direct_matcher
    validate_caches
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
