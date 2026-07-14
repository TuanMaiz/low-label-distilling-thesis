# Project Progress: 2026-07-13

| Plan | Status | Progress | Priority | Next action |
|---|---|---:|---|---|
| Cost-Aware Active LLM Labeling For WDC Entity Matching | In progress | 4/8 phases (50%) | P1 | Commit/push the Phase 5 change set; run the fixed Colab GPU pilot |

## Phase 5 Status

| Item | Status |
|---|---|
| Three fixed `train_128` targets | Prepared |
| Fixed gold validation and direct-LLM artifacts | Prepared |
| Binary target/generation limit | `8` tokens |
| Input limit and padding | Fixed at `512`; unchanged |
| Tokenization | Once per training/evaluation process |
| Validation loss | Weighted by non-padding label tokens |
| Automatic CUDA precision | BF16 when supported; FP16 with scaling otherwise |
| Automatic validation batch | 32 on BF16 CUDA; 16 on other CUDA |
| Recovery | Contract-validated stage skip, stale-artifact archive, atomic markers |
| Provenance contracts | Git commit, configuration, and target/upstream hashes |
| Student inference evidence | Local timing, throughput, device, precision, and batch metadata |
| Aggregate comparisons | Signed deltas versus `llm_random` and `gold_random` |
| GPU student validation results | Pending |
| Phase 5 success criteria | 0/7 complete; unchanged |

## Verification Evidence

| Gate | Result |
|---|---|
| Latest contract/runtime/aggregation tests | 18/18 passed |
| Full regression suite | 40/40 passed |
| Python compilation, shell syntax, and CLI checks | Passed |
| Precision resolution and validation-batch tests | Passed |
| Non-finite mixed-precision validation guard | Passed |
| Atomic summary, prediction, and metric writes | Passed |
| Isolated recovery and stale-artifact archive scenario | Passed |
| Teacher-call and test-target audit | Passed |
| Independent review/debug pass | Clean |
| Actual CUDA/model training | Pending Colab run |

## Runtime Contract

- Training batch size remains 4; runtime changes do not alter update batching.
- `MAX_TARGET_LENGTH=8` and `MAX_NEW_TOKENS=8` match the fixed `match` and
  `non-match` outputs.
- `PRECISION=auto` selects BF16 on supporting CUDA hardware, FP16 with gradient
  scaling on other CUDA hardware, and FP32 for CPU smoke checks.
- `VALIDATION_BATCH_SIZE=auto` selects 32 with BF16 CUDA, 16 with other CUDA,
  and the training batch size on CPU.
- Token-weighted validation loss preserves early-stopping/checkpoint semantics
  across validation batch groupings.
- Tokenization happens once per process; fixed 512-token truncation and padding
  remain unchanged.
- Training completes only when a best checkpoint and atomically replaced
  summary both exist. Evaluation completes only when atomically replaced
  predictions and metrics both exist.
- Each stage also requires a matching atomic contract containing the Git
  commit, relevant configuration, and SHA-256 hashes of target or upstream
  contract files. Missing or mismatched contracts block reuse and require a new
  `OUTPUT_ROOT` or explicit `FORCE=1`.
- Interrupted training restarts the affected variant. Before retraining, prior
  contracts, summaries, and downstream evaluation artifacts receive
  `.stale.<timestamp>` names so incompatible stages are not combined.
- Compact packages preserve training and evaluation contracts.
- Student metrics preserve synchronized local FLAN-T5 generation time, total
  evaluation wall time, throughput, seconds per pair, device, precision,
  batch size, and sequence limits; training summaries preserve synchronized
  training seconds. A small OpenRouter model is not used as a student-cost
  proxy. The aggregator applies all predeclared low/base/high analytical
  GPU-hour sensitivity rates, preserves their file hash, and emits training,
  per-pair inference, comparison-scale savings, and break-even costs.
- Aggregation reports signed match F1, macro F1, and accuracy deltas for each
  student versus `llm_random` and `gold_random`.

## Risks

- The optimizations are covered by CPU/unit/integration checks but still need
  one real Colab CUDA/model run.
- GPU dollar costs remain sensitivity estimates rather than observed Colab
  charges or provider quotes; measured runtime remains the primary evidence.
- A Colab disconnect during training restarts the current variant; use a Google
  Drive `OUTPUT_ROOT` to preserve completed stages.
- The fresh-clone workflow still depends on committing and pushing the full
  Phase 5 change set together.
- Test leakage remains controlled: the runner does not read the test target,
  and no teacher LLM call is reachable from the fixed Phase 5 flow.

## Next Steps

1. Commit and push `codex/distiller-wdc-implementation`.
2. From a fresh Colab GPU clone, run `scripts/run_phase05_colab.sh setup` and
   then `scripts/run_phase05_colab.sh all`.
3. Return `phase05_train_128_results.tar.gz` for validation comparison and the
   Phase 5 continue/revise/stop decision.

## Unresolved Questions

- None before the fixed validation pilot; checkpoint archives remain optional.
