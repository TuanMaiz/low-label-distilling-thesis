# Runbook — treatment arms A, B, C on a rented Tesla V100

Plan: `plans/261023-2015-wdc-guideline-judge-treatment/` (contract amendment A3:
Tesla V100-SXM2-32GB, fp16). Runner: `scripts/run_wdc_qwen_treatment_arms.sh`.
Written 2026-10-26 after the first setup on host `bronze-echidna`.

## Known-good environment

| Item | Value |
|---|---|
| GPU | `Tesla V100-SXM2-32GB` (preflight requires this exact name) |
| Image | provider "PyTorch" template |
| Python with torch | `/opt/conda/bin/python` (Python 3.10) — **not on PATH** |
| torch | 2.2.1, CUDA 12.1 (same as the 2026-09 OOF runs) |
| Driver | 580.82.07 (reports CUDA 13.0; the wheel's CUDA 12.1 runtime works with it) |

## Pitfalls already hit

1. **`ModuleNotFoundError: No module named 'torch'` in `setup`.** `setup` does
   not install torch; its first line only checks that torch exists. The
   template's torch lives in `/opt/conda`, which is not on PATH, and
   `/usr/bin/python3` has no torch. Fix: `export PYTHON=/opt/conda/bin/python`.
2. **Do not `pip install -r requirements.txt` on the GPU host.** It has an
   unpinned `torch>=2.0.0`; pip would pull a current PyPI build whose CUDA
   (12.8+/13.x) no longer supports Volta. `setup` uses
   `requirements-colab.txt`, which leaves torch out on purpose.
3. **If torch must be installed by hand**, use the Volta-compatible build:
   `$PYTHON -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121`.
   PyTorch states CUDA 13.x does not support GPUs below Turing, and the
   cu128/cu129 builds dropped Volta; cu126 up to torch 2.14 is the last
   prebuilt option.
4. **A new shell or tmux window forgets `PYTHON`.** Put it in `~/.bashrc`.

## 0. Clone and tmux

```bash
git clone -b feat/wdc-guideline-judge https://github.com/TuanMaiz/low-label-distilling-thesis.git code
cd code
tmux new -s wdc          # detach: Ctrl-b d   reattach: tmux attach -t wdc
```

Private repo: username = GitHub user, password = a personal access token.
The arm datasets and the serialized validation/test files are tracked in git,
so nothing else needs copying.

## 1. Python and GPU check

```bash
export PYTHON=/opt/conda/bin/python
echo 'export PYTHON=/opt/conda/bin/python' >> ~/.bashrc
$PYTHON -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.2.1 12.1 True Tesla V100-SXM2-32GB
```

If `/opt/conda` does not exist on a new image, find torch with
`find / -maxdepth 7 -type d -name torch -path "*site-packages*" 2>/dev/null`
and use the `python` in that environment's `bin/`.

## 2. Setup (once)

```bash
bash scripts/run_wdc_qwen_treatment_arms.sh setup A
```

## 3. Arms, one step at a time

Run one command at a time; start the next only after the previous one ends
cleanly.

```bash
# Arm A
bash scripts/run_wdc_qwen_treatment_arms.sh preflight A
bash scripts/run_wdc_qwen_treatment_arms.sh train A --confirm-full-training
bash scripts/run_wdc_qwen_treatment_arms.sh evaluate A --confirm-validation
bash scripts/run_wdc_qwen_treatment_arms.sh evaluate-test A --confirm-test
bash scripts/run_wdc_qwen_treatment_arms.sh package-results A

# Arm B
bash scripts/run_wdc_qwen_treatment_arms.sh preflight B
bash scripts/run_wdc_qwen_treatment_arms.sh train B --confirm-full-training
bash scripts/run_wdc_qwen_treatment_arms.sh evaluate B --confirm-validation
bash scripts/run_wdc_qwen_treatment_arms.sh evaluate-test B --confirm-test
bash scripts/run_wdc_qwen_treatment_arms.sh package-results B

# Arm C
bash scripts/run_wdc_qwen_treatment_arms.sh preflight C
bash scripts/run_wdc_qwen_treatment_arms.sh train C --confirm-full-training
bash scripts/run_wdc_qwen_treatment_arms.sh evaluate C --confirm-validation
bash scripts/run_wdc_qwen_treatment_arms.sh evaluate-test C --confirm-test
bash scripts/run_wdc_qwen_treatment_arms.sh package-results C
```

- Packaging runs once per arm and includes test results only if they exist,
  so it comes after `evaluate-test`.
- Test is run once per frozen arm (contract section 6). Once any test result
  exists, do not change code, settings or data for the remaining arms; a real
  bug found afterwards means rerunning every arm and a contract amendment.
  To avoid that risk, run all three `evaluate-test` lines only after all three
  validations, then the three `package-results` lines.

## 4. Resume after a failure

Results live in `outputs/uncertainty_refinement/wdc-qwen-treatment/arm-<X>/`.
Commands never overwrite, so remove the failed step's leftovers, then rerun
only that step.

| Failed at | Clean up |
|---|---|
| `preflight` | nothing |
| `train` | `rm -rf .../arm-X/run` |
| `evaluate` | `rm -f .../arm-X/validation-predictions.jsonl .../arm-X/validation-metrics.json` |
| `evaluate-test` | `rm -f .../arm-X/test-predictions.jsonl .../arm-X/test-metrics.json .../arm-X/test-contract.json` (only if it crashed before writing results) |
| `package-results` | `rm -f .../arm-X.zip .../arm-X.zip.tmp` |

## 5. Download and check (on your own computer)

```bash
scp 'root@<host>:code/outputs/uncertainty_refinement/wdc-qwen-treatment/arm-*.zip*' .
sha256sum -c arm-A.zip.sha256 arm-B.zip.sha256 arm-C.zip.sha256
```

Archives hold predictions, metrics, logs, preflight records and the LoRA
adapter; the merged 2.4 GB model is left out (`PACKAGE_WEIGHTS=adapter`).
