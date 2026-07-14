# Phase 5 Colab Runtime Optimization

Date: 2026-07-13
Plan: `plans/260704-distiller-wdc-agent-execution/plan.md`
Tags: `wdc-products`, `phase-05`, `colab`, `runtime`, `mixed-precision`

## Context

The fixed 128-row Phase 5 pilot will run on Colab rather than locally. Before
starting the three FLAN-T5-base students, the training path was reviewed for
avoidable work that could be removed without changing the target data,
validation split, optimization schedule, or anti-cherry-pick contract.

## Verified Length Evidence

The FLAN-T5 tokenizer represents `match` in 2 tokens and `non-match` in 4
tokens, including EOS. Therefore, both `MAX_TARGET_LENGTH=8` and
`MAX_NEW_TOKENS=8` leave a safe margin for every Phase 5 binary target.

The input limit remains 512. Reducing it would discard too much product-pair
context: 1,082 of 2,500 validation rows exceed 256 tokens, 597 exceed 384, and
348 exceed 512. The corresponding counts for the 128-row gold/random target
are 63, 36, and 23; for the active target they are 38, 22, and 8. Phase 5
therefore keeps the declared 512-token truncation behavior.

## What Changed

- Set the Colab prediction default to `MAX_NEW_TOKENS=8`, matching the already
  reduced `MAX_TARGET_LENGTH=8`.
- Added a separate validation batch size without changing the training batch
  size. Automatic CUDA defaults are 32 for BF16 and 16 otherwise; CPU keeps
  the training batch size.
- Changed validation loss from an unweighted mean of batch losses to a
  label-token-weighted mean. Early stopping and checkpoint selection therefore
  remain invariant to validation batch grouping.
- Added automatic precision selection: BF16 on CUDA devices that report BF16
  support, FP16 on other CUDA devices, and FP32 on CPU. Training and validation
  use autocast; FP16 training uses gradient scaling; final generation uses
  autocast and inference mode.
- Pre-tokenized training, validation, and prediction rows once when each
  dataset is constructed instead of repeating tokenization on every access and
  epoch. Fixed 512-token padding was retained because the measured dynamic
  padding benefit falls to about 0.1-1.4 percent at validation batch sizes
  16-32.
- Recorded resolved precision, validation batch size, PyTorch and Transformers
  versions, CUDA version, and CUDA device name in the training summary.
- Improved interruption recovery by writing training summaries, predictions,
  and metrics atomically, archiving stale partial result files before a retry,
  and preserving stage-boundary resume behavior.
- Added a non-finite validation-loss guard with an explicit FP32 retry hint.

## Verification

- Focused runtime and recovery checks: 22/22 passed.
- Full suite: 35/35 passed with
  `.venv/bin/python -m unittest discover -s tests -v`.
- Python compilation, Bash syntax, runner help, and CLI wiring checks passed.
- Precision selection was exercised on CPU and mocked CUDA capability paths.
- Tests cover validation-loss invariance, the non-finite guard,
  pre-tokenization, atomic writes, and isolated stale-artifact recovery.
- Static runner review confirmed no teacher invocation and no test-target path.
- Final code-review and debugger passes reported no remaining findings.

## Decision and Limitation

These changes are runtime and recovery optimizations only. They do not alter
the fixed training pairs, validation rows, epoch limit, learning rate, primary
metric, or test-split embargo.

CUDA FP16/BF16 execution and FLAN-T5-base training were intentionally not run
locally. Actual GPU compatibility, memory use, runtime, student metrics, and
the Phase 5 continue/revise/stop decision remain pending until the Colab run
returns its validation artifacts. No Phase 5 research result is claimed here.

## Next Step

Commit and push the complete Phase 5 change set, execute the fixed branch on a
Colab GPU, and return `phase05_train_128_results.tar.gz` for validation review.

## Follow-up: Artifact Reuse and Cost Evidence

Date: 2026-07-14

The pre-Colab review identified three gaps and the Phase 5 workflow was updated
before execution:

- Training and evaluation stages now write atomic artifact contracts containing
  the Git commit, runtime configuration, and SHA-256 hashes of targets and
  relevant upstream contract/source files. Existing completion markers are
  skipped only when the current contract matches exactly. A missing or
  mismatched contract blocks reuse and requires a new `OUTPUT_ROOT` or explicit
  `FORCE=1`; forced reruns archive stale contracts and downstream artifacts.
  A final run-level runtime contract also fixes the actual GPU name, resolved
  precision, and resolved validation batch across variants, preventing T4/A100
  mixing after a Colab reconnect.
- Student cost evidence now comes from the actual local FLAN-T5 validation run,
  which records synchronized generation time, wall time, throughput, seconds
  per pair, device, precision, batch size, and generation limits. An unrelated
  small OpenRouter model was rejected as a price proxy because it would measure
  a different model and serving stack; a declared GPU-price assumption can be
  applied later to the captured student runtime.
- The Phase 5 aggregator now reports signed match F1, macro F1, and accuracy
  deltas for every student versus both `llm_random` and `gold_random`, making
  the same-budget active-versus-random comparison explicit.

Focused tests cover exact contract reuse, rejection after target or
configuration changes, missing contracts, inference timing fields, and signed
aggregation deltas. Shell and Python checks remain part of the Colab preflight.
Final follow-up verification passed 18/18 focused tests and 40/40 tests in the
full regression suite.

Phase 5 is still pending: no CUDA student result or continue/revise/stop decision
is claimed until the Colab validation archive is returned and reviewed.

## Follow-up: Stable Student Cost Accounting

Date: 2026-07-14

The workflow now records synchronized `trainer.train` wall seconds for each
student and combines them with the existing synchronized inference seconds.
The versioned `configs/phase05_cost_assumptions.json` predeclares low, base, and
high analytical GPU-hour rates ($0.25, $1.00, and $4.00). Aggregation reports
all scenarios together,
embeds the assumptions SHA-256, and emits one-time teacher plus training cost,
per-pair inference cost, savings at the direct-baseline validation scale, and
the first whole-query cost-parity-or-better point. Break-even is null if student
inference is not cheaper per pair. The scenario output carries the explicit
trainer-loop timing scope so it does not imply model loading or tokenization was
priced.

The dollar rates are sensitivity assumptions, not observed Colab charges or
current provider quotes. Measured seconds and device metadata remain the stable
evidence; a later dated, sourced assumptions file can be substituted without
rerunning the students. The compact result archive includes both the scenario
CSV and the assumptions file.

Final verification after the cost-accounting follow-up passed 49/49 repository
tests, including invalid-rate/cost guards and exact cost-parity behavior.
