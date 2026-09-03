#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPOSITORY_ROOT/.venv/bin/python}"
PROFILE="${DBLP_QWEN_PROFILE:-configs/executions/dblp_acm_qwen_vertical_slice.json}"

phase4_blocked() {
  printf '%s\n' "Phase 4 does not authorize setup, model loading, CUDA, training, evaluation, or official packaging." >&2
  printf '%s\n' "Complete real DBLP labeling/targets and the later execution approval before this action." >&2
  exit 2
}

action="${1:-list}"
if [[ $# -gt 0 ]]; then shift; fi
cd "$REPOSITORY_ROOT"

case "$action" in
  list)
    cat <<'EOF'
CPU-safe Phase 4 actions:
  list
  config
  identity
  plan                      (render future train/evaluate commands; execute nothing)
  preflight                 (official; expected to fail until real targets are approved)
  fixture-preflight         (pass fixture arguments after the action)
  state                     (pass --state-path)
  package-fixture           (also pass --validation and --target)

Declared but locked in Phase 4:
  setup smoke train-gold train-llm-hard verify-results package-arm package-results
EOF
    ;;
  config|identity|plan|preflight|fixture-preflight|state|package-fixture)
    if [[ ! -x "$PYTHON_BIN" ]]; then
      printf 'Repository Python is unavailable: %s\n' "$PYTHON_BIN" >&2
      exit 2
    fi
    exec "$PYTHON_BIN" -m experiments.dblp_acm_qwen_preflight "$action" \
      --profile "$PROFILE" --repository-root "$REPOSITORY_ROOT" "$@"
    ;;
  setup|smoke|train-gold|train-llm-hard|verify-results|package-arm|package-results)
    phase4_blocked
    ;;
  *)
    printf 'Unknown action: %s\n' "$action" >&2
    exit 2
    ;;
esac
