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

## Dataset 3 — MaDI-Bench Companies (arm G now, arm A after teacher labelling)

Runner: `scripts/run_madi_companies_arms.sh <action> <G|A>`, same recipe,
defaults and safeguards as the WDC runner. Plan:
`plans/261028-2230-madi-companies-dataset3/plan.md`.

**The data is not in git.** MaDI-Bench has no licence and FullContact forbids
redistribution, so `data/raw/madi_companies/` and `data/cache/madi_companies/`
are git-ignored. Copy them to the GPU host privately; never upload them anywhere
public.

On your computer (WSL), after `python -m data.prepare_madi_companies prepare`:

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
python -m data.prepare_madi_companies verify
tar czf madi-companies-data.tar.gz data/raw/madi_companies data/cache/madi_companies
sha256sum madi-companies-data.tar.gz > madi-companies-data.tar.gz.sha256
scp -P "$PORT" madi-companies-data.tar.gz madi-companies-data.tar.gz.sha256 "$HOST:code/"
```

On the GPU host, after the usual clone, PYTHON export, version pins and `setup G`:

```bash
cd ~/code
sha256sum -c madi-companies-data.tar.gz.sha256 && tar xzf madi-companies-data.tar.gz
$PYTHON -m data.prepare_madi_companies verify
bash scripts/run_madi_companies_arms.sh preflight G
bash scripts/run_madi_companies_arms.sh train G --confirm-full-training
bash scripts/run_madi_companies_arms.sh evaluate G --confirm-validation
bash scripts/run_madi_companies_arms.sh evaluate-test G --confirm-test
bash scripts/run_madi_companies_arms.sh package-results G
rm -f madi-companies-data.tar.gz   # do not leave the data on the rented machine
```

- `evaluate-test` writes the locked test split once (`serialized-test/`) if it
  is missing; it needs the raw files, which is why `data/raw/madi_companies/`
  is copied too.
- `package-results` also writes `validation-source-pair-metrics.json` and
  `test-source-pair-metrics.json`: match F1 for Forbes–DBpedia and
  Forbes–FullContact separately plus their mean. Always report these next to
  the pooled F1.
- Results land in `outputs/uncertainty_refinement/madi-companies-qwen/arm-<X>.zip`
  (git-ignored as well).
- Before ending the rental, delete the data and the code copy from the machine.

## Dataset 2 — Geo-ER Pittsburgh OSM–Foursquare (D24)

Runner: `scripts/run_geo_er_arms.sh <action> <G|A>`, same recipe, defaults and
safeguards as the Companies runner. Each pair carries a computed
`distance_m` line; teacher and student see the same input.

**The data is not in git** (no licence; Foursquare terms). Copy
`data/raw/geo_er/` and `data/cache/geo_er/` privately, as for Companies.

Teacher labelling, on your computer (dry run without the two flags):

```bash
python -m supervision.run_full_labeling_geo_er label --pass-index 1 --confirm-paid-labeling --spend-ceiling-usd 5
python -m supervision.run_full_labeling_geo_er label --pass-index 2 --confirm-paid-labeling --spend-ceiling-usd 5
python -m supervision.run_full_labeling_geo_er label --pass-index 3 --confirm-paid-labeling --spend-ceiling-usd 5
python -m supervision.run_full_labeling_geo_er majority
```

Transfer, on your computer:

```bash
python -m data.prepare_geo_er verify
tar czf geo-er-data.tar.gz data/raw/geo_er data/cache/geo_er
sha256sum geo-er-data.tar.gz > geo-er-data.tar.gz.sha256
scp -P "$PORT" geo-er-data.tar.gz geo-er-data.tar.gz.sha256 "$HOST:code/"
```

On the GPU host, after the usual clone, PYTHON export and `setup G`:

```bash
cd ~/code
sha256sum -c geo-er-data.tar.gz.sha256 && tar xzf geo-er-data.tar.gz && rm -f geo-er-data.tar.gz
$PYTHON -m data.prepare_geo_er verify
for ARM in G A; do
  bash scripts/run_geo_er_arms.sh preflight $ARM &&
  bash scripts/run_geo_er_arms.sh train $ARM --confirm-full-training &&
  bash scripts/run_geo_er_arms.sh evaluate $ARM --confirm-validation || break
done
# only after both validations finish cleanly:
for ARM in G A; do
  bash scripts/run_geo_er_arms.sh evaluate-test $ARM --confirm-test &&
  bash scripts/run_geo_er_arms.sh package-results $ARM || break
done
```

Results: `outputs/uncertainty_refinement/geo-er-qwen/arm-<X>.zip` (git-ignored).
Delete `~/code` on the GPU host before ending the rental.

## Companies treatment arms B and C (plan 261106)

Two GPU stages with one paid step on your computer in between. Keep the rental
running during the evidence check (it is cheap), or end it and rent again.

**1. Your computer: refresh the data archive** (it now includes `oof/majority-3fold`).

```bash
uv run python -m data.prepare_madi_companies verify
uv run bash scripts/run_madi_companies_oof.sh verify-folds
rm -f madi-companies-data.tar.gz madi-companies-data.tar.gz.sha256
tar czf madi-companies-data.tar.gz data/raw/madi_companies data/cache/madi_companies
sha256sum madi-companies-data.tar.gz > madi-companies-data.tar.gz.sha256
scp -P "$PORT" madi-companies-data.tar.gz madi-companies-data.tar.gz.sha256 "$HOST:code/"
```

**2. GPU: three OOF folds.**

```bash
cd ~/code && git pull
sha256sum -c madi-companies-data.tar.gz.sha256 && tar xzf madi-companies-data.tar.gz && rm -f madi-companies-data.tar.gz
bash scripts/run_madi_companies_oof.sh verify-folds
bash scripts/run_madi_companies_oof.sh preflight
for f in 1 2 3; do bash scripts/run_madi_companies_oof.sh train-fold $f --confirm-oof-training || break; done
bash scripts/run_madi_companies_oof.sh merge
bash scripts/run_madi_companies_oof.sh package-results
```

**3. Your computer: download OOF, evidence check (paid), build B and C.**

```bash
cd /mnt/d/study/cao-hoc/luan-van/code/outputs/uncertainty_refinement
scp -P "$PORT" "$HOST:code/outputs/uncertainty_refinement/madi-companies-oof.zip*" .
sha256sum -c madi-companies-oof.zip.sha256 && python3 -c "import zipfile; zipfile.ZipFile('madi-companies-oof.zip').extractall('.')"
cd /mnt/d/study/cao-hoc/luan-van/code
uv run python -m supervision.run_evidence_check_v2 prepare --dataset madi_companies   # free: count + estimate
uv run python -m supervision.run_evidence_check_v2 run --dataset madi_companies --confirm-paid-evidence-check
uv run python -m supervision.build_category_treated_dataset_companies build --arm B
uv run python -m supervision.build_category_treated_dataset_companies build --arm C
```

Stop and check if `prepare` selects more than 400 pairs (contract 261106, section 8).

**4. Your computer → GPU: send the bundles, train B and C.**

```bash
tar czf madi-companies-bc.tar.gz data/cache/madi_companies/madi-bench-companies-base-062058d/refinement
sha256sum madi-companies-bc.tar.gz > madi-companies-bc.tar.gz.sha256
scp -P "$PORT" madi-companies-bc.tar.gz madi-companies-bc.tar.gz.sha256 "$HOST:code/"
```

```bash
cd ~/code && sha256sum -c madi-companies-bc.tar.gz.sha256 && tar xzf madi-companies-bc.tar.gz && rm -f madi-companies-bc.tar.gz
for ARM in B C; do
  bash scripts/run_madi_companies_arms.sh preflight $ARM &&
  bash scripts/run_madi_companies_arms.sh train $ARM --confirm-full-training &&
  bash scripts/run_madi_companies_arms.sh evaluate $ARM --confirm-validation || break
done
# only after both validations finish cleanly:
for ARM in B C; do
  bash scripts/run_madi_companies_arms.sh evaluate-test $ARM --confirm-test &&
  bash scripts/run_madi_companies_arms.sh package-results $ARM || break
done
```

Download `arm-B.zip` and `arm-C.zip` as for G and A, then delete `~/code` on the GPU host.

**5. Analysis (after both arms are frozen).** Paired comparisons C vs A, C vs B,
B vs A, and the label audit against train gold:

```bash
uv run python -m analysis.category_rule_gold_audit \
  --decisions data/cache/madi_companies/madi-bench-companies-base-062058d/refinement/category-rule-C/decisions.csv \
  --gold data/cache/madi_companies/madi-bench-companies-base-062058d/full_label_targets/gold.jsonl
```

