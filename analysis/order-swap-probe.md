# Record-order swap probe (WDC–Qwen)

Diagnostic for one question: does the Qwen reranker judge identity
symmetrically, or does its relevance-style framing make it judge one direction?

- Entity identity is symmetric, so `(A, B)` and `(B, A)` should score alike.
- The reranker puts Record A in `<Query>` and Record B in `<Document>` under the
  fixed system prompt "Judge whether the Document meets the requirements based
  on the Query…". Each training pair was seen in one order only.
- A relevance prior predicts higher scores when the vaguer record is the Query
  (a specific Document "meets the requirements" of a vague Query).

The probe is inference only: no training, no LLM calls, and no change to any
existing label, checkpoint or threshold. Gold labels are used only to group
errors and compute metrics.

## Outputs

`score` writes `scores.jsonl` (one row per pair and order: logits, log-odds,
match probability, token count) and `run.json` (model identity and hash, input
hash, device, precision, timing, git commit). Re-running the same command
resumes; a different input, pair subset, limit or instruction must use a new
`--output-dir`.

`summarize` writes `summary.json` and an optional per-pair CSV:

| Field | Meaning |
|---|---|
| `reproduction_check` | Original-order scores vs. saved predictions |
| `groups.*.abs_delta` | Size of the order effect in log-odds, per original outcome (TP/FP/TN/FN) |
| `groups.*.decision_flips` | Pairs whose decision changes when swapped |
| `groups.*.direction_delta` | Log-odds with the vaguer title as Query minus the reverse; positive supports a relevance prior |
| `metrics_at_saved_threshold` | Original, swapped, `mean` and `min` order fusion at the run's saved threshold |
| `selected_thresholds` | Validation-selected thresholds for fused scores (optimistic on the same split) |

"Vaguer" is a proxy: fewer distinct title tokens; ties are excluded. Log-odds
are used because probabilities saturate near 0 and 1.

## Inputs to upload

Not in git (about 50 MB). Create the bundle locally:

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
tar czf outputs/new/order-swap-colab-inputs.tar.gz -C outputs/new/wdc-qwen-gold-vs-llm-hard \
  gold/run/best_adapter gold/run/decision_threshold.json gold/run/validation.predictions.jsonl \
  llm_hard/run/best_adapter llm_hard/run/decision_threshold.json llm_hard/run/validation.predictions.jsonl
```

Upload it to Google Drive, for example `MyDrive/thesis/order-swap-colab-inputs.tar.gz`.

## Colab

Use a GPU runtime (T4 is enough). Keep outputs on Drive so a disconnect only
costs the unfinished batch.

```python
from google.colab import drive, userdata
drive.mount("/content/drive")
token = userdata.get("GITHUB_TOKEN")  # Colab secret with repo read access
!git clone https://{token}@github.com/TuanMaiz/low-label-distilling-thesis.git /content/repo
%cd /content/repo
!git checkout analysis/wdc-qwen-order-swap-probe
!pip install -q -r requirements-colab.txt
!mkdir -p outputs/new/wdc-qwen-gold-vs-llm-hard && tar xzf /content/drive/MyDrive/thesis/order-swap-colab-inputs.tar.gz -C outputs/new/wdc-qwen-gold-vs-llm-hard
```

```bash
RUN=outputs/new/wdc-qwen-gold-vs-llm-hard
OUT=/content/drive/MyDrive/thesis/order-swap
VAL=data/cache/wdc_products/serialized/validation.jsonl
TEST=data/cache/wdc_products/serialized/test.jsonl
```

### 1. Reproduction check (10 pairs, about a minute)

```bash
python -m analysis.order_swap_probe score --model $RUN/gold/run/best_adapter \
  --input $VAL --output-dir $OUT/check-gold --limit 10
python -m analysis.order_swap_probe summarize --scores $OUT/check-gold/scores.jsonl \
  --input $VAL --threshold-json $RUN/gold/run/decision_threshold.json \
  --saved-predictions $RUN/gold/run/validation.predictions.jsonl \
  --allow-partial --output $OUT/check-gold/summary.json
```

Continue only if `reproduction_check` shows small probability differences
(fp16/bf16 and CPU/GPU differ slightly) and no or very few decision
disagreements. Large differences mean the setup does not reproduce the run.

### 2. Validation, all 2,500 pairs

```bash
for ARM in gold llm_hard; do
  python -m analysis.order_swap_probe score --model $RUN/$ARM/run/best_adapter \
    --input $VAL --output-dir $OUT/val-$ARM
  python -m analysis.order_swap_probe summarize --scores $OUT/val-$ARM/scores.jsonl \
    --input $VAL --threshold-json $RUN/$ARM/run/decision_threshold.json \
    --saved-predictions $RUN/$ARM/run/validation.predictions.jsonl \
    --select-fused-thresholds --output $OUT/val-$ARM/summary.json \
    --pairs-csv $OUT/val-$ARM/pairs.csv
done

python -m analysis.order_swap_probe score --model base --input $VAL --output-dir $OUT/val-base
python -m analysis.order_swap_probe summarize --scores $OUT/val-base/scores.jsonl \
  --input $VAL --threshold 0.5 --select-fused-thresholds \
  --output $OUT/val-base/summary.json --pairs-csv $OUT/val-base/pairs.csv
```

For the base model, outcome groups at 0.5 are only indicative; read its
asymmetry and direction over all pairs.

### 3. Test (optional; researcher decision)

The gold-arm checkpoint has not been evaluated on test before. WDC test has
already informed later method ideas, so it is not a pristine confirmation set.
Apply only thresholds chosen on validation:

```bash
python -m analysis.order_swap_probe score --model $RUN/gold/run/best_adapter \
  --input $TEST --output-dir $OUT/test-gold
python -m analysis.order_swap_probe summarize --scores $OUT/test-gold/scores.jsonl \
  --input $TEST --threshold-json $RUN/gold/run/decision_threshold.json \
  --fused-thresholds-json $OUT/val-gold/summary.json \
  --output $OUT/test-gold/summary.json --pairs-csv $OUT/test-gold/pairs.csv
```

## Reading the result

| Result | Reading |
|---|---|
| Little asymmetry in tuned and base model | Order and framing are not the issue; look at learning (minimal-pair supervision) |
| Asymmetry in base, gone after tuning | The prior existed, training removed it |
| Asymmetry in both, positive direction | Relevance framing still shapes decisions; consider an identity system prompt or two-order training |
| Asymmetry without consistent direction | Likely one-order training bias rather than a relevance prior |

Order fusion can only change pairs where one order disagrees; it cannot fix
pairs scored near 1 in both orders. Validation-selected fusion gains are leads
to confirm on another dataset, not WDC results.

## Local checks

```bash
.venv/bin/python -m unittest tests.test_order_swap_probe -v
```
