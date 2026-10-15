# WDC Qwen student: variant-aware instruction experiment

Status: retired by researcher request on 2026-10-15. The current legacy-named
runner selects the original config `configs/students/qwen3_reranker_0_6b.json`,
uses result identity `quick_refined_original_prompt`, and defaults to the separate
output root `outputs/uncertainty_refinement/wdc-qwen-quick-refined-original-prompt`.
The variant config is retained unchanged to reproduce completed variant runs.
Historical commands below describe the variant experiment before rollback; use
Git revision 7c8f923 when reproducing that run, with its recorded runtime.
No training or inference is started by the rollback.

Verified result: original/variant validation match F1 was 0.8113744076/0.8091743119;
test match F1 was 0.7581227437/0.7465986395. Test FP increased 188 to 237 while
FN decreased 80 to 61. These results do not establish a causal explanation or
show that shorter instructions generally improve generalization.

## Hypothesis

An explicit product-variant instruction may reduce false matches between
closely related products without an unacceptable recall loss. This is an
experimental hypothesis, not an established improvement.

## Configs

- Baseline: `configs/students/qwen3_reranker_0_6b.json`.
- Variant v1: `configs/students/qwen3_reranker_0_6b_variant_identity_v1.json`.
- Only `reranker_instruction` changes. Model identity, student ID, LoRA,
  serialization, answer tokens and input-length settings are identical.
- The variant retains the same student ID to isolate the instruction change;
  select a distinct output directory so results cannot overwrite another arm.
- The existing WDC vertical-slice/OOF preflight deliberately pins the baseline
  instruction. Those frozen dispatchers do not select or accept variant v1.
  A separately reviewed execution profile is needed before using a frozen
  dispatcher for this experiment.

## Run entry point

The dedicated dispatcher selects variant v1 for training, validation, and test:

```bash
PYTHON=.venv/bin/python bash scripts/run_wdc_qwen_variant_prompt.sh plan
```

`plan` only prints the exact commands; it starts no model work and needs no
GPU. The train command uses the existing quick-refined targets and the saved
baseline schedule (batch 1, accumulation 16, 10 maximum epochs, learning rate
0.0002, warmup ratio 0.1, seed 42, fp16). Validation uses the same config and
the new run's `best_model`, with saved predictions and metrics in a dedicated
output root. `OUTPUT_ROOT` can select another fresh directory; `PYTHON` selects
the runtime (use the GPU environment's Python on the rental machine).

After separate authorization for GPU execution, the actions are:

```bash
bash scripts/run_wdc_qwen_variant_prompt.sh setup
bash scripts/run_wdc_qwen_variant_prompt.sh preflight
bash scripts/run_wdc_qwen_variant_prompt.sh train --confirm-full-training
bash scripts/run_wdc_qwen_variant_prompt.sh evaluate --confirm-validation
bash scripts/run_wdc_qwen_variant_prompt.sh evaluate-test --confirm-test
bash scripts/run_wdc_qwen_variant_prompt.sh package-results
```

The dispatcher refuses existing training directories and existing evaluation
outputs. Before evaluating, it verifies that the training summary records the
variant instruction, expected model and 2,493 training rows, and matches the
verified checkpoint manifest. The researcher authorized test evaluation for this
experiment. `evaluate-test --confirm-test` reads the tracked 4,500-pair test
input and uses the checkpoint's saved validation-selected threshold without
retuning. It saves test-predictions.jsonl, test-metrics.json, logs/test.log and
a test-contract.json binding the input, config, summary, threshold and code.
Packaging verifies test outputs when present and rejects partial test results.

For an already-trained run created with b16ce0e, pull this branch update and
skip completed stages. Validation, test and packaging accept that exact prior
runner hash while checking every other original preflight binding; the original
contract is preserved and current result code bindings are recorded separately.
Training still checks the current runner hash exactly. Do not rerun preflight
on an existing trained run. Existing archives are never overwritten; package
only after both evaluations finish.

### Fresh rental machine

The image must already supply CUDA-compatible PyTorch. Clone the branch, then
start tmux:

```bash
git clone --branch codex/wdc-student-variant-prompt https://github.com/TuanMaiz/low-label-distilling-thesis.git
cd low-label-distilling-thesis
tmux new -s wdc-prompt-v1
```

Inside tmux, use the template interpreter that already imports CUDA PyTorch:

```bash
# Set this to your verified template interpreter path.
export PYTHON=/path/to/template/python
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
```

Then run the six dispatcher actions above in order, stopping on any failure.
The refined targets, validation input and test input are tracked on this branch. `setup`
checks CUDA, installs requirements-colab.txt, and checks LoRA. `preflight`
checks counts/IDs, audits complete tokenized inputs, and freezes an artifact
contract. Training requires that contract and saves its log automatically.
`package-results` rederives validation and available test counts/F1, checks checkpoint integrity,
and creates OUTPUT_ROOT.zip and OUTPUT_ROOT.zip.sha256, including both weight
formats, predictions, metrics, preflight records, and logs. Zip packaging uses
Python's standard library; no system zip installation is needed.

Detach with Ctrl+B then D; reconnect using `tmux attach -t wdc-prompt-v1`.

## Approved instruction

```text
Determine whether Record A and Record B describe the same specific product variant. Shared brand, product family, or similar wording alone is insufficient. Explicit conflicts in model identity, color, size, capacity, flavor, pack quantity, or bundle contents indicate different variants when they describe the offered products. Equivalent translations, abbreviations, and units are not conflicts. Missing attributes alone are not conflicts. An accessory is different from the main product it is compatible with. Answer yes only when the records support the same specific variant without an explicit identity conflict; otherwise answer no.
```

## Controlled comparison

Select the variant config using `--student-config` in both training and
evaluation. Use the same 2,493-row quick-refined training dataset, validation
split, model, seed, LoRA settings, optimizer schedule, checkpoint metric and
validation-based threshold-selection rule as the corresponding baseline.
Numeric fitted thresholds may differ; the selection rule must be identical.
Preserve per-pair predictions and all checkpoint files after an authorized run.

Evaluate validation match F1, precision and recall together. A blanket shift
toward non-match is not sufficient evidence of better variant discrimination.
Do not substitute a single-pass-label checkpoint for the three-pass/refined
comparison. An inference-only instruction swap on a checkpoint tests prompt
sensitivity; training and evaluating with a new instruction tests the complete
instruction condition.

The proposed instruction was motivated partly by inspection of saved test
errors. The hypothesis is post-hoc; do not tune on test or describe that test
set as untouched confirmation of a method designed from those errors. No
teacher labels, official splits or previous predictions change here.

## Literature basis and limits

- Steiner and Bizer (2026), *Labeling Training Data for Entity Matching Using
  Large Language Models*, Figure 3 / Section 2.4: a teacher review prompt treats
  conflicting identity attributes as evidence against a match. Applying that
  principle to the Qwen student is our adaptation, not a result of their paper.
  https://arxiv.org/html/2606.28823v1#S2.SS4
- Pangakis and Wolken (2024), *Knowledge Distillation in Automated Annotation*,
  Appendix C.1: inspect disagreements and make limited prompt clarifications;
  avoid overfitting instructions to a small subset. Their method concerns GPT-4
  annotation, not this Qwen student experiment.
  https://aclanthology.org/2024.nlpcss-1.9/
- The official Qwen model card supports customized reranking instructions. It
  does not guarantee an improvement on entity resolution.
  https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
