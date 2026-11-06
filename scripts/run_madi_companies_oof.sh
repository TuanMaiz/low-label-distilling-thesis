#!/usr/bin/env bash
# MaDI-Bench Companies out-of-fold (OOF) student predictions, for detection of
# uncertain LLM-labelled training pairs (plan 261106, arms B and C). Three folds
# of the LLM majority labels; each fold trains on two thirds and predicts the held
# out third. Same recipe, GPU and precision as scripts/run_madi_companies_arms.sh.
# No LLM calls, no gold labels. Data is git-ignored: the folds are built locally
# (build-folds) and copied with data/cache/madi_companies/ to the GPU host.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

usage() {
  echo "Usage: bash scripts/run_madi_companies_oof.sh <build-folds|verify-folds|plan|preflight|train-fold N|merge|package-results> [--confirm-oof-training]" >&2
  echo "Env: PYTHON, OUTPUT_ROOT, SEED (default 42), PRECISION (default fp16), EXPECTED_GPU (default 'Tesla V100-SXM2-32GB')." >&2
  exit 2
}
[[ "$#" -ge 1 ]] || usage
ACTION="$1"; shift
PYTHON="${PYTHON:-python}"
SEED="${SEED:-42}"
PRECISION="${PRECISION:-fp16}"
EXPECTED_GPU="${EXPECTED_GPU:-Tesla V100-SXM2-32GB}"
RUNNER="scripts/run_madi_companies_oof.sh"
DATA="data/cache/madi_companies/madi-bench-companies-base-062058d"
INPUTS="${DATA}/teacher_labels/inputs.jsonl"
MAJORITY="${DATA}/majority_labels/llm_unrefined.csv"
OOF_DIR="${DATA}/oof/majority-3fold"
STUDENT_CONFIG="configs/students/qwen3_reranker_0_6b_madi_companies.json"
EXPECTED_COUNT=1968
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uncertainty_refinement/madi-companies-oof}"
RUNTIME="${OUTPUT_ROOT}/runtime.json"

fold_name() {
  case "$1" in 1) echo fold_01 ;; 2) echo fold_02 ;; 3) echo fold_03 ;; *) echo "Fold must be 1, 2 or 3" >&2; exit 2 ;; esac
}
require_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
show_cmd() { printf '%q ' "$@"; printf '\n'; }
fold_args=(--inputs "${INPUTS}" --majority-labels "${MAJORITY}" --output-dir "${OOF_DIR}"
           --dataset-id madi_companies --expected-count "${EXPECTED_COUNT}")

train_cmd() {
  local fold; fold="$(fold_name "$1")"
  echo "${PYTHON}" -m experiments.train_student --student-config "${STUDENT_CONFIG}" \
    --train-targets "${OOF_DIR}/${fold}/train.jsonl" --validation-targets "${OOF_DIR}/${fold}/val.jsonl" \
    --output-dir "${OUTPUT_ROOT}/${fold}/train" --batch-size 1 --validation-batch-size 1 \
    --gradient-accumulation-steps 16 --num-epochs 10 --learning-rate 0.0002 \
    --weight-decay 0.01 --warmup-steps 0 --warmup-ratio 0.1 \
    --max-input-length 4096 --seed "${SEED}" --early-stopping-patience 3 \
    --precision "${PRECISION}" --device cuda
}
evaluate_cmd() {
  local fold; fold="$(fold_name "$1")"
  echo "${PYTHON}" -m experiments.evaluate_student --student-config "${STUDENT_CONFIG}" \
    --checkpoint "${OUTPUT_ROOT}/${fold}/train/best_model" --input "${OOF_DIR}/${fold}/test-heldout.jsonl" \
    --predictions "${OUTPUT_ROOT}/${fold}/test-heldout.predictions.jsonl" \
    --metrics "${OUTPUT_ROOT}/${fold}/test-heldout.metrics.json" \
    --variant "oof_${fold}" --budget full --split oof_test_heldout \
    --batch-size 1 --max-input-length 4096 --precision "${PRECISION}" --device cuda
}

case "${ACTION}" in
  build-folds)
    [[ "$#" == 0 ]] || usage
    "${PYTHON}" -m supervision.prepare_oof_folds build "${fold_args[@]}"
    ;;
  verify-folds)
    [[ "$#" == 0 ]] || usage
    "${PYTHON}" -m supervision.prepare_oof_folds verify "${fold_args[@]}"
    ;;
  plan)
    [[ "$#" == 0 ]] || usage
    for f in 1 2 3; do train_cmd "$f"; evaluate_cmd "$f"; done
    ;;
  preflight)
    [[ "$#" == 0 ]] || usage
    "${PYTHON}" -m supervision.prepare_oof_folds verify "${fold_args[@]}" >/dev/null
    mkdir -p "${OUTPUT_ROOT}"
    "${PYTHON}" - "${STUDENT_CONFIG}" "${PRECISION}" "${EXPECTED_GPU}" "${SEED}" "${RUNTIME}" <<'PREFLIGHT_PY'
import json, subprocess, sys
from importlib import metadata
from pathlib import Path
import torch
config_path, precision, expected_gpu, seed, runtime = sys.argv[1:]
assert torch.cuda.is_available(), "CUDA is unavailable"
gpu = torch.cuda.get_device_name(0)
assert gpu == expected_gpu, f"GPU {gpu!r} differs from the contract GPU {expected_gpu!r}"
config = json.loads(Path(config_path).read_text())
assert config == json.loads(Path("configs/students/qwen3_reranker_0_6b_madi_companies.json").read_text())
identity = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "gpu": gpu, "precision": precision, "seed": seed, "torch": torch.__version__, "cuda": torch.version.cuda,
            "packages": {k: metadata.version(k) for k in ("transformers", "peft", "accelerate")}}
path = Path(runtime)
if path.exists() and json.loads(path.read_text()) != identity:
    raise SystemExit(f"Runtime identity differs from {path}; earlier folds ran in another environment")
path.write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps(identity, indent=2))
PREFLIGHT_PY
    echo "OOF preflight passed."
    ;;
  train-fold)
    [[ "$#" == 2 && "$2" == "--confirm-oof-training" ]] || { echo "train-fold requires a fold number and --confirm-oof-training" >&2; exit 1; }
    fold="$(fold_name "$1")"
    require_file "${RUNTIME}"
    for p in "${OUTPUT_ROOT}/${fold}/train" "${OUTPUT_ROOT}/${fold}/test-heldout.predictions.jsonl"; do
      [[ ! -e "${p}" ]] || { echo "Refusing to replace existing ${p}" >&2; exit 1; }
    done
    mkdir -p "${OUTPUT_ROOT}/${fold}/logs"
    read -r -a tcmd <<< "$(train_cmd "$1")"; read -r -a ecmd <<< "$(evaluate_cmd "$1")"
    show_cmd "${tcmd[@]}"; "${tcmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${fold}/logs/train.log"
    show_cmd "${ecmd[@]}"; "${ecmd[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${fold}/logs/heldout.log"
    ;;
  merge)
    [[ "$#" == 0 ]] || usage
    "${PYTHON}" - "${OOF_DIR}" "${OUTPUT_ROOT}" "${EXPECTED_COUNT}" <<'MERGE_PY'
import json, sys
from pathlib import Path
from experiments.wdc_qwen_oof import merge_oof_predictions
oof_dir, root, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
summary = merge_oof_predictions(oof_dir=oof_dir, results_root=root, output_csv=root / "oof_predictions.csv",
                                output_summary=root / "oof_summary.json", expected_count=count)
assert summary["dataset_id"] == "madi_companies"
print(json.dumps({k: summary[k] for k in ("dataset_id", "row_count", "output")}, indent=2))
MERGE_PY
    ;;
  package-results)
    [[ "$#" == 0 ]] || usage
    require_file "${OUTPUT_ROOT}/oof_predictions.csv"; require_file "${OUTPUT_ROOT}/oof_summary.json"
    "${PYTHON}" - "${OUTPUT_ROOT}" <<'PACKAGE_PY'
import sys, zipfile
from pathlib import Path
from utils.artifact_contract import sha256_file
out = Path(sys.argv[1])
archive, tmp = Path(str(out) + ".zip"), Path(str(out) + ".zip.tmp")
if archive.exists() or tmp.exists():
    raise SystemExit("Refusing to replace an existing archive")
# Predictions, metrics, logs, runtime and training summaries; no weights.
files = sorted(p for p in out.rglob("*") if p.is_file() and "best_model" not in p.parts
               and "best_adapter" not in p.parts and "checkpoint" not in "/".join(p.parts))
with zipfile.ZipFile(tmp, "x", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(out.parent))
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None
tmp.replace(archive)
Path(str(archive) + ".sha256").write_text(sha256_file(archive) + "  " + archive.name + "\n")
print("Download:", archive.resolve())
PACKAGE_PY
    ;;
  *) usage ;;
esac
