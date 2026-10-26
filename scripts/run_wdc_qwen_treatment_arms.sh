#!/usr/bin/env bash
# WDC–Qwen training runner for plan 261023 arms A, B, C (contract.md section 5).
# Defaults: Tesla V100-SXM2-32GB with fp16 (contract amendment A3).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

usage() {
  echo "Usage: bash $0 <setup|preflight|plan|train|evaluate|evaluate-test|package-results> <A|B|C> [--confirm-...]" >&2
  echo "Env: PYTHON, OUTPUT_ROOT, PRECISION (default fp16), EXPECTED_GPU (default 'Tesla V100-SXM2-32GB')," >&2
  echo "     PACKAGE_WEIGHTS=none|adapter|all (default adapter)." >&2
  exit 1
}
ACTION="${1:-}"; ARM="${2:-}"
[[ "${ARM}" =~ ^[ABC]$ ]] || usage
shift 2

PYTHON="${PYTHON:-.venv/bin/python}"
PRECISION="${PRECISION:-fp16}"
EXPECTED_GPU="${EXPECTED_GPU:-Tesla V100-SXM2-32GB}"
PACKAGE_WEIGHTS="${PACKAGE_WEIGHTS:-adapter}"
[[ "${PRECISION}" =~ ^(fp16|bf16)$ ]] || { echo "PRECISION must be fp16 or bf16" >&2; exit 1; }
[[ "${PACKAGE_WEIGHTS}" =~ ^(none|adapter|all)$ ]] || { echo "PACKAGE_WEIGHTS must be none, adapter or all" >&2; exit 1; }
RUNNER="scripts/run_wdc_qwen_treatment_arms.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/uncertainty_refinement/wdc-qwen-treatment/arm-${ARM}}"
STUDENT_CONFIG="configs/students/qwen3_reranker_0_6b.json"
DATA_DIR="data/cache/wdc_products/refinement/category-rule-${ARM}"
TRAIN_TARGETS="${DATA_DIR}/train.jsonl"
MANIFEST="${DATA_DIR}/manifest.json"
VALIDATION="data/cache/wdc_products/serialized/validation.jsonl"
TEST="data/cache/wdc_products/serialized/test.jsonl"
VARIANT="category_rule_arm_${ARM}"
RUN_DIR="${OUTPUT_ROOT}/run"
PREDICTIONS="${OUTPUT_ROOT}/validation-predictions.jsonl"
METRICS="${OUTPUT_ROOT}/validation-metrics.json"
PREFLIGHT="${OUTPUT_ROOT}/preflight"
LOGS="${OUTPUT_ROOT}/logs"
CONTRACT="${PREFLIGHT}/artifact-contract.json"

show_cmd() { printf '%q ' "$@"; printf '\n'; }
require_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
require_confirmation() {
  [[ "$#" == 1 && "$1" == "$confirmation" ]] || {
    echo "This action requires ${confirmation}; use plan for an offline preview." >&2
    exit 1
  }
}
manifest_rows() {
  "${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["train_rows"])' "${MANIFEST}"
}
contract_args() {
  local rows; rows="$(manifest_rows)"
  printf '%s\n' --field "stage=wdc_treatment_arm_${ARM}" --field "train_rows=${rows}" \
    --field validation_rows=2500 --field "precision=${PRECISION}" --field "expected_gpu=${EXPECTED_GPU}" \
    --file "student_config=${STUDENT_CONFIG}" --file "train=${TRAIN_TARGETS}" --file "manifest=${MANIFEST}" \
    --file "validation=${VALIDATION}" --file "runtime=${PREFLIGHT}/runtime.json" \
    --file "audit=${PREFLIGHT}/input-length-audit.json" --file "runner=${RUNNER}" \
    --file trainer=experiments/train_student.py --file evaluator=experiments/evaluate_student.py \
    --file backend=models/generative_reranker_student.py --file config_loader=models/student_config.py
}
check_preflight() {
  require_file "${CONTRACT}"; require_file "${MANIFEST}"
  local args; mapfile -t args < <(contract_args)
  "${PYTHON}" -m utils.artifact_contract check --path "${CONTRACT}" "${args[@]}"
}
verify_checkpoint() {
  require_file "${RUN_DIR}/training_summary.json"; require_file "${MANIFEST}"
  "${PYTHON}" - "${STUDENT_CONFIG}" "${RUN_DIR}" "${MANIFEST}" <<'CHECKPOINT'
import json, sys
from pathlib import Path
from utils.checkpoint_manifest import validate_checkpoint_manifest
config_path, run_dir, manifest_path = map(Path, sys.argv[1:])
config = json.loads(config_path.read_text())
summary = json.loads((run_dir / "training_summary.json").read_text())
rows = json.loads(manifest_path.read_text())["train_rows"]
if summary.get("reranker_instruction") != config["reranker_instruction"]:
    raise SystemExit("Checkpoint instruction mismatch: use a checkpoint trained with the original config")
if summary.get("model_name") != config["model_name"] or summary.get("train_rows") != rows:
    raise SystemExit(f"Checkpoint does not describe this arm's {rows}-row training set")
threshold = json.loads((run_dir / "best_model/decision_threshold.json").read_text())["decision_threshold"]
if not 0 <= threshold <= 1 or threshold != summary.get("decision_threshold"):
    raise SystemExit("Checkpoint threshold differs from the training summary")
manifest = validate_checkpoint_manifest(run_dir)
if summary.get("checkpoint_manifest") != manifest:
    raise SystemExit("Training summary does not match the verified checkpoint manifest")
CHECKPOINT
}

train_cmd=("${PYTHON}" -m experiments.train_student
  --student-config "${STUDENT_CONFIG}"
  --train-targets "${TRAIN_TARGETS}" --validation-targets "${VALIDATION}"
  --output-dir "${RUN_DIR}" --batch-size 1 --validation-batch-size 1
  --gradient-accumulation-steps 16 --num-epochs 10 --learning-rate 0.0002
  --weight-decay 0.01 --warmup-steps 0 --warmup-ratio 0.1
  --max-input-length 4096 --seed 42 --early-stopping-patience 3
  --precision "${PRECISION}" --device cuda)
evaluate_cmd=("${PYTHON}" -m experiments.evaluate_student
  --student-config "${STUDENT_CONFIG}" --checkpoint "${RUN_DIR}/best_model"
  --input "${VALIDATION}" --predictions "${PREDICTIONS}" --metrics "${METRICS}"
  --variant "${VARIANT}" --split validation
  --batch-size 1 --max-input-length 4096 --precision "${PRECISION}" --device cuda)
evaluate_test_cmd=("${PYTHON}" -m experiments.evaluate_student
  --student-config "${STUDENT_CONFIG}" --checkpoint "${RUN_DIR}/best_model"
  --input "${TEST}" --predictions "${OUTPUT_ROOT}/test-predictions.jsonl"
  --metrics "${OUTPUT_ROOT}/test-metrics.json"
  --variant "${VARIANT}" --split test
  --batch-size 1 --max-input-length 4096 --precision "${PRECISION}" --device cuda)

case "${ACTION}" in
  setup)
    [[ "$#" == 0 ]] || usage
    mkdir -p "${LOGS}"
    "${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "The GPU image must provide CUDA PyTorch"; print(torch.__version__, torch.cuda.get_device_name(0))'
    "${PYTHON}" -m pip install -r requirements-colab.txt 2>&1 | tee "${LOGS}/setup.log"
    "${PYTHON}" -m utils.peft_runtime sanitize
    "${PYTHON}" -m utils.peft_runtime check
    "${PYTHON}" -m pip freeze > "${LOGS}/pip-freeze.txt"
    ;;
  preflight)
    [[ "$#" == 0 ]] || usage
    require_file "${STUDENT_CONFIG}"; require_file "${TRAIN_TARGETS}"; require_file "${MANIFEST}"; require_file "${VALIDATION}"
    mkdir -p "${PREFLIGHT}" "${LOGS}"
    "${PYTHON}" - "${STUDENT_CONFIG}" "${TRAIN_TARGETS}" "${MANIFEST}" "${VALIDATION}" "${PREFLIGHT}" \
      "${ARM}" "${PRECISION}" "${EXPECTED_GPU}" <<'PREFLIGHT_PY'
import hashlib, json, subprocess, sys
from collections import Counter
from importlib import metadata
from pathlib import Path
import torch
config_path, train_path, manifest_path, val_path, out = map(Path, sys.argv[1:6])
arm, precision, expected_gpu = sys.argv[6:9]
assert torch.cuda.is_available(), "CUDA is unavailable"
gpu = torch.cuda.get_device_name(0)
assert gpu == expected_gpu, f"GPU {gpu!r} differs from the contract GPU {expected_gpu!r}; set EXPECTED_GPU only with a contract amendment"
if precision == "bf16":
    assert torch.cuda.is_bf16_supported(), "bf16 is not supported on this GPU"
config = json.loads(config_path.read_text())
assert config == json.loads(Path("configs/students/qwen3_reranker_0_6b.json").read_text()), "The original config must be selected"
manifest = json.loads(manifest_path.read_text())
assert manifest["arm"] == arm and manifest["gold_accessed"] is False
assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest["train_sha256"], "train.jsonl differs from its manifest"
train = [json.loads(l) for l in train_path.read_text().splitlines()]
val = [json.loads(l) for l in val_path.read_text().splitlines()]
counts = Counter(r["target_text"] for r in train)
assert len(train) == manifest["train_rows"] and dict(counts) == manifest["class_counts"]
assert len(val) == 2500 and Counter(r["label"] for r in val) == {0: 2000, 1: 500}
a = {r["pair_id"] for r in train}
b = {r["pair_id"] for r in val}
assert len(a) == len(train) and len(b) == 2500
assert not {r.get("duplicate_of", r["pair_id"]) for r in train} & b, "train and validation overlap"
assert all("\n\nRecord A:\n" in r["input_text"] and "\n\nRecord B:\n" in r["input_text"] for r in train + val)
identity = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "arm": arm, "gpu": gpu, "precision": precision, "torch": torch.__version__,
            "cuda": torch.version.cuda, "train_rows": len(train), "validation_rows": len(val),
            "packages": {k: metadata.version(k) for k in ("transformers", "peft", "accelerate")}}
(out / "runtime.json").write_text(json.dumps(identity, indent=2) + "\n")
print(json.dumps(identity, indent=2))
PREFLIGHT_PY
    "${PYTHON}" -m models.generative_reranker_student \
      --student-config "${STUDENT_CONFIG}" --input "${TRAIN_TARGETS}" --input "${VALIDATION}" \
      --max-input-length 4096 --output "${PREFLIGHT}/input-length-audit.json" \
      2>&1 | tee "${LOGS}/input-audit.log"
    if [[ -f "${CONTRACT}" ]]; then
      check_preflight
    else
      mapfile -t args < <(contract_args)
      "${PYTHON}" -m utils.artifact_contract write --path "${CONTRACT}" "${args[@]}"
    fi
    echo "Arm ${ARM} GPU preflight passed."
    ;;
  plan)
    [[ "$#" == 0 ]] || usage
    show_cmd "${train_cmd[@]}"
    show_cmd "${evaluate_cmd[@]}"
    show_cmd "${evaluate_test_cmd[@]}"
    ;;
  train)
    confirmation=--confirm-full-training; require_confirmation "$@"
    if [[ -e "${RUN_DIR}" ]]; then
      echo "Refusing to replace an existing run directory: ${RUN_DIR}" >&2; exit 1
    fi
    check_preflight
    mkdir -p "${LOGS}"
    show_cmd "${train_cmd[@]}"
    "${train_cmd[@]}" 2>&1 | tee "${LOGS}/train.log"
    ;;
  evaluate)
    confirmation=--confirm-validation; require_confirmation "$@"
    if [[ -e "${PREDICTIONS}" || -e "${METRICS}" ]]; then
      echo "Refusing to overwrite existing validation results" >&2; exit 1
    fi
    verify_checkpoint
    check_preflight
    mkdir -p "${LOGS}"
    show_cmd "${evaluate_cmd[@]}"
    "${evaluate_cmd[@]}" 2>&1 | tee "${LOGS}/validation.log"
    ;;
  evaluate-test)
    confirmation=--confirm-test; require_confirmation "$@"
    if [[ -e "${OUTPUT_ROOT}/test-predictions.jsonl" || -e "${OUTPUT_ROOT}/test-metrics.json" || -e "${OUTPUT_ROOT}/test-contract.json" ]]; then
      echo "Refusing to overwrite existing test results or dispatch contract" >&2; exit 1
    fi
    require_file "${TEST}"
    verify_checkpoint
    check_preflight
    "${PYTHON}" - "${TEST}" "${TRAIN_TARGETS}" "${VALIDATION}" <<'TEST_INPUT'
import json, sys
from collections import Counter
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
ids = {r["pair_id"] for r in rows}
assert len(rows) == len(ids) == 4500
assert Counter(r["label"] for r in rows) == {0: 4000, 1: 500}
for path in sys.argv[2:]:
    other = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert not ids & {r.get("duplicate_of", r["pair_id"]) for r in other}
TEST_INPUT
    "${PYTHON}" -m utils.artifact_contract write --path "${OUTPUT_ROOT}/test-contract.json" \
      --field split=test --field rows=4500 --field "arm=${ARM}" --field "precision=${PRECISION}" \
      --file "test=${TEST}" --file "student_config=${STUDENT_CONFIG}" \
      --file "summary=${RUN_DIR}/training_summary.json" \
      --file "threshold=${RUN_DIR}/best_model/decision_threshold.json" \
      --file evaluator=experiments/evaluate_student.py --file "runner=${RUNNER}"
    mkdir -p "${LOGS}"
    show_cmd "${evaluate_test_cmd[@]}"
    "${evaluate_test_cmd[@]}" 2>&1 | tee "${LOGS}/test.log"
    ;;
  package-results)
    [[ "$#" == 0 ]] || usage
    require_file "${PREDICTIONS}"; require_file "${METRICS}"
    check_preflight
    verify_checkpoint
    "${PYTHON}" - "${OUTPUT_ROOT}" "${VALIDATION}" "${TEST}" "${VARIANT}" "${PACKAGE_WEIGHTS}" <<'PACKAGE_PY'
import json, math, sys, zipfile
from pathlib import Path
from utils.artifact_contract import sha256_file
out, val_path, test_path = map(Path, sys.argv[1:4])
variant, weights = sys.argv[4:6]
splits = [("validation", val_path, 2500)]
if any((out / n).exists() for n in ("test-predictions.jsonl", "test-metrics.json", "test-contract.json")):
    from utils.artifact_contract import validate_recorded_contract
    contract = validate_recorded_contract(out / "test-contract.json")
    assert contract["fields"]["split"] == "test" and contract["fields"]["rows"] == "4500"
    assert contract["files"]["test"] == {"path": str(test_path), "sha256": sha256_file(test_path)}
    splits.append(("test", test_path, 4500))
summary = json.loads((out / "run/training_summary.json").read_text())
for split, input_path, count in splits:
    expected = {r["pair_id"]: r for r in map(json.loads, input_path.read_text().splitlines())}
    rows = list(map(json.loads, (out / f"{split}-predictions.jsonl").read_text().splitlines()))
    metrics = json.loads((out / f"{split}-metrics.json").read_text())
    assert len(rows) == len({r["pair_id"] for r in rows}) == len(expected) == count
    assert set(expected) == {r["pair_id"] for r in rows}
    for r in rows:
        assert r["is_valid"] and r["label"] == expected[r["pair_id"]]["label"]
        score = r["match_probability"]
        assert math.isfinite(score) and 0 <= score <= 1
        assert r["prediction"] == int(score >= summary["decision_threshold"])
    tp = sum(r["label"] == 1 and r["prediction"] == 1 for r in rows)
    fp = sum(r["label"] == 0 and r["prediction"] == 1 for r in rows)
    fn = sum(r["label"] == 1 and r["prediction"] == 0 for r in rows)
    tn = len(rows) - tp - fp - fn
    assert all(metrics[k] == v for k, v in {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": count, "valid": count, "invalid": 0}.items())
    assert math.isclose(metrics["same_f1"], 2 * tp / (2 * tp + fp + fn), abs_tol=1e-12)
    assert metrics["split"] == split and metrics["variant"] == variant
    assert metrics["decision_threshold"] == summary["decision_threshold"]
excluded = {"none": ("run/best_model/", "run/best_adapter/"), "adapter": ("run/best_model/",), "all": ()}[weights]
archive = Path(str(out) + ".zip")
temporary = Path(str(archive) + ".tmp")
if archive.exists() or temporary.exists():
    raise SystemExit("Refusing to replace an existing results archive or partial archive")
files = sorted(p for p in out.rglob("*") if p.is_file()
               and not any(p.relative_to(out).as_posix().startswith(e) for e in excluded)
               or (p.is_file() and p.name == "decision_threshold.json"))
assert not any(p.is_symlink() for p in files)
with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for p in files:
        z.write(p, p.relative_to(out.parent))
with zipfile.ZipFile(temporary) as z:
    assert z.testzip() is None and len(z.namelist()) == len(files)
temporary.replace(archive)
Path(str(archive) + ".sha256").write_text(sha256_file(archive) + "  " + archive.name + "\n")
print("Weights packaged:", weights)
print("Download:", archive.resolve())
print("Checksum:", Path(str(archive) + ".sha256").resolve())
PACKAGE_PY
    ;;
  *)
    usage
    ;;
esac
