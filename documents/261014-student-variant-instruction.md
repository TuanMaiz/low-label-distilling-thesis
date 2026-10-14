# WDC Qwen student: variant-aware instruction experiment

Status: researcher-approved instruction/config change, 2026-10-14. Commit and
push are authorized. No GPU training, paid labeling or new official evaluation
is started by this change.

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
