# Phase 2: Per-pair score logging

## Context Links

- `experiments/trainer.py`: `Trainer.train_epoch`, `Trainer.evaluate`, `Trainer.train`
- `experiments/train_student.py`: `train_configured_student` (builds loaders; train loader `shuffle=True`)
- `models/generative_reranker_student.py`: `GenerativeRerankerModel.save_pretrained` (saves the PEFT adapter)
- Test fakes: `tests/test_generative_reranker.py` (`_FakeTokenizer`, `_FakeCausalLM`)

## Overview

Priority P1. Pending. Opt-in logging so one run also yields per-pair signals
for RHO-style, cartography and JTT-style selection, plus per-epoch artifacts for
the Phase 9 robustness checks.

## Key Insights

- `Trainer` logs only batch-mean losses; nothing per pair is stored.
- **RNG coupling:** creating a `DataLoader` iterator draws a base seed from the global torch generator. An extra pass over train would shift the shuffle order of later epochs. The pass must save and restore RNG state and use its own generator.
- `evaluate` already computes validation match probabilities but discards them.
- LoRA adapters are ~9 MB; one per epoch is ~90 MB, so keep them all.

## Requirements

- New argument `pair_scores_dir: Path | None = None` on `train_configured_student`, CLI `--pair-scores-dir`; default `None` keeps behaviour unchanged.
- `Trainer.__init__` gains `epoch_end_hook: Callable[[int, float | dict], None] | None = None`, called in `train` after validation and checkpoint saving, **before** the early-stopping `break`, every epoch.
- `Trainer.evaluate` gains `return_probabilities: bool = False`; when true the result dict also holds `match_probabilities` (validation order).
- Hook behaviour (`PairScoreRecorder` in a new module `experiments/pair_scores.py`):
  - Snapshot `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` (if CUDA), `np.random.get_state()`, `random.getstate()`; restore after the pass.
  - Train-eval loader: `shuffle=False`, `generator=torch.Generator().manual_seed(0)`, validation batch size, same collator.
  - `model.eval()`, `torch.inference_mode()`, trainer autocast. Per training pair: `pair_id`, `epoch`, `label_id`, `logit_match`, `logit_non_match`, `log_odds`, `match_probability`, `loss` (cross-entropy on the supplied label, as in `GenerativeRerankerModel.forward`).
  - Append rows to `train_pair_scores.jsonl`; flush and fsync per epoch.
  - Write `val_predictions/epoch-NN.jsonl` (`pair_id`, `label`, `match_probability`) from the evaluate result.
  - `model.save_pretrained(epoch_adapters/epoch-NN)`.
- `training_summary.json` gains `pair_scores`: directory, epochs logged, rows per epoch, seconds per epoch.
- Generative reranker only; raise a clear error if enabled for other architectures.

## Recorded scores (exact definitions)

The student reads one pair and predicts the next token. Only two tokens
matter: `yes` (match) and `no` (non-match). `GenerativeRerankerModel.forward`
already takes the final-position logits of these two tokens; the recorder
stores them and values derived from them. All values are computed in eval mode
(no dropout), after the epoch's last optimizer step.

| Field | Definition | Range | Meaning |
|---|---|---|---|
| `pair_id` | training pair ID | — | which pair |
| `epoch` | epoch number, 1-based | 1–10 | when it was measured |
| `label_id` | supplied training label: 1 = match, 0 = non-match (teacher majority label) | 0 / 1 | what the model is taught |
| `logit_match` | final-position logit of token `yes` | any real | raw evidence for match |
| `logit_non_match` | final-position logit of token `no` | any real | raw evidence for non-match |
| `log_odds` | `logit_match − logit_non_match` | any real | > 0 leans match; same quantity as the swap probe's `log_odds` |
| `match_probability` | softmax over the two logits = `1 / (1 + exp(−log_odds))` | 0–1 | same number `evaluate_student` writes as `match_probability` |
| `loss` | `−log P(label_id)` = cross-entropy of the two logits against `label_id`, as in `forward` | ≥ 0 | how wrong the model is on its own training label; 0.69 = coin flip |

Example row:

```json
{"pair_id": "88031640#6256141", "epoch": 2, "label_id": 0,
 "logit_match": 3.1, "logit_non_match": -1.4, "log_odds": 4.5,
 "match_probability": 0.989, "loss": 4.51}
```

Here the teacher says non-match, the model says match with 98.9%, so the loss
is high (−log 0.011 ≈ 4.51).

Which later phase uses which field:

| Field | Used for |
|---|---|
| `loss` (epochs 1–2, averaged) | Phase 6 `current_loss` (RHO-style priority, teacher-trust ranking) |
| `match_probability` over all epochs | optional cartography view: mean confidence and variability per pair |
| `log_odds` | comparison with swap-probe and OOF scores on the same scale |

Validation files (`val_predictions/epoch-NN.jsonl`) store `pair_id`, `label`
(validation gold, allowed for checkpoint selection) and `match_probability`,
the same definition as above.

## Architecture

```text
Trainer.train
  for epoch:
    train_epoch -> evaluate(val, return_probabilities=hook set)
    -> checkpoint/threshold save (unchanged)
    -> epoch_end_hook(epoch, evaluation)        # new, before early-stop break
         save RNG -> score train (eval mode) -> restore RNG
         -> train_pair_scores.jsonl, val_predictions/epoch-NN.jsonl, epoch_adapters/epoch-NN/
```

## Related Code Files

- Modify: `experiments/trainer.py`, `experiments/train_student.py`
- Create: `experiments/pair_scores.py`, `tests/test_pair_scores_logging.py`

## Implementation Steps

1. Add `return_probabilities` to `evaluate` (default off; existing return shape unchanged).
2. Add `epoch_end_hook` to `Trainer` and call it at the stated point.
3. Implement `PairScoreRecorder` with RNG isolation and the three outputs.
4. Wire `--pair-scores-dir` through `train_configured_student` and `main`; add summary fields.
5. Tests (CPU, fake model):
   - With and without the recorder: identical per-epoch `train_loss` history and identical best-epoch choice.
   - `torch.get_rng_state()` equal before and after one recorder call.
   - Exactly N train rows per epoch; IDs match train rows; validation file order matches validation rows.
   - One adapter directory per epoch; early-stopped epoch is still logged.
6. Run focused tests, then the full suite.

## Todo List

- [ ] `evaluate(return_probabilities)`
- [ ] `epoch_end_hook`
- [ ] `PairScoreRecorder` with RNG isolation
- [ ] CLI and summary fields
- [ ] Tests pass with the full suite

## Success Criteria

- `.venv/bin/python -m unittest tests.test_pair_scores_logging tests.test_generative_reranker -v` passes; the full suite passes.
- Flag-off runs produce the same metrics and checkpoints as today.

## Risk Assessment

- Hidden RNG coupling changes training. Mitigation: equality tests on losses and RNG state.
- Extra time per epoch. Mitigation: batched inference; seconds recorded in the summary.

## Security Considerations

- None beyond existing artifact handling.

## Next Steps

- Phase 4 (runs with logging on), Phase 5 (OOF with logging).
