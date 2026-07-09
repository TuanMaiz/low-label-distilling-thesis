#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build Phase 4 compact-student target files from fixed WDC pairs and teacher labels.

Usage:
  scripts/run_phase04_targets.sh <command>

Commands:
  build      Build gold_random, llm_random, and active LLM-label targets.
  test       Run target-builder related unit tests.
  local      Run build + test.

Environment overrides:
  PYTHON=.venv/bin/python
  BUDGET=128
  PROMPT_VERSION=answer_only_v1
  ACTIVE_STRATEGY=llm_active_bucketed_v1
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

PYTHON="${PYTHON:-.venv/bin/python}"
BUDGET="${BUDGET:-128}"
PROMPT_VERSION="${PROMPT_VERSION:-answer_only_v1}"
ACTIVE_STRATEGY="${ACTIVE_STRATEGY:-llm_active_bucketed_v1}"

LOW_LABEL_INPUT="data/cache/wdc_products/low_label/train_${BUDGET}.jsonl"
RANDOM_MANIFEST="data/cache/wdc_products/selection_manifests/train_${BUDGET}.random.jsonl"
ACTIVE_MANIFEST="data/cache/wdc_products/selection_manifests/train_${BUDGET}.${ACTIVE_STRATEGY}.jsonl"
RANDOM_LABELS="data/cache/wdc_products/teacher_labels/train_${BUDGET}.random.openrouter.${PROMPT_VERSION}.labels.jsonl"
ACTIVE_LABELS="data/cache/wdc_products/teacher_labels/train_${BUDGET}.${ACTIVE_STRATEGY}.openrouter.${PROMPT_VERSION}.labels.jsonl"
GOLD_RANDOM_TARGETS="data/cache/wdc_products/targets/train_${BUDGET}.gold_random.targets.jsonl"
LLM_RANDOM_TARGETS="data/cache/wdc_products/targets/train_${BUDGET}.llm_random.targets.jsonl"
ACTIVE_TARGETS="data/cache/wdc_products/targets/train_${BUDGET}.${ACTIVE_STRATEGY}.targets.jsonl"

require_python() {
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON}" >&2
    exit 1
  fi
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

build_targets() {
  require_python
  run_cmd "${PYTHON}" -m supervision.build_targets \
    --pairs "${LOW_LABEL_INPUT}" \
    --output "${GOLD_RANDOM_TARGETS}" \
    --variant gold_random

  run_cmd "${PYTHON}" -m supervision.build_targets \
    --pairs "${RANDOM_MANIFEST}" \
    --teacher-labels "${RANDOM_LABELS}" \
    --output "${LLM_RANDOM_TARGETS}" \
    --variant llm_random

  run_cmd "${PYTHON}" -m supervision.build_targets \
    --pairs "${ACTIVE_MANIFEST}" \
    --teacher-labels "${ACTIVE_LABELS}" \
    --output "${ACTIVE_TARGETS}" \
    --variant "${ACTIVE_STRATEGY}"
}

run_tests() {
  require_python
  run_cmd "${PYTHON}" -m unittest tests.test_phase03_student
}

command="${1:-}"
case "${command}" in
  build)
    build_targets
    ;;
  test)
    run_tests
    ;;
  local)
    build_targets
    run_tests
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
