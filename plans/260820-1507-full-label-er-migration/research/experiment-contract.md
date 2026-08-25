# Full-Label Entity Resolution Experiment Contract

## Document Status

| Field | Value |
|---|---|
| Status | Draft — not frozen |
| Active plan | `plans/260820-1507-full-label-er-migration/plan.md` |
| Research question | Can LLM-generated hard labels provide a practical alternative to benchmark gold training labels for compact cross-encoder Entity Resolution models while reducing deployment cost compared with direct LLM matching? |
| Contract owner | TBD |
| Reviewers | TBD |
| Freeze commit | TBD |
| Last scientific review | TBD |

This Markdown file is the human-readable scientific source of truth. Code may
hash it for provenance but must not parse it as executable configuration.
Dataset and model JSON files implement decisions recorded here after this
contract is frozen.

## Fixed Scope

- Three benchmark datasets.
- Three compact models that jointly encode both records.
- Two training-label sources: benchmark gold and LLM-generated hard labels.
- One predeclared run per experiment cell.
- Eighteen compact-model train/evaluate cells.
- Three logical direct-LLM baselines, one per dataset.
- Match-class F1 as the primary metric.

Out of scope unless the supervisor explicitly changes the study: low-label
budgets, active selection, rationale generation or distillation, adaptive
cascades, repeated-run experiments, and additional datasets or models.

## Decision Summary

| Decision | Status | Selected value | Evidence |
|---|---|---|---|
| Dataset 1 | Selected; license clarification remains a freeze gate | WDC Products, pair-wise, 80% corner-cases, small development set, 100% unseen-products test | [WDC benchmark page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/); local archive and split hashes below |
| Dataset 2 | TBD | TBD | TBD |
| Dataset 3 | TBD | TBD | TBD |
| Compact model 1 | Selected | `Qwen/Qwen3-Reranker-0.6B` | [Official model repository](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B); prior repository screening artifacts |
| Compact model 2 | TBD | TBD | TBD |
| Compact model 3 | TBD | TBD | TBD |
| LLM provider and model | TBD | TBD | TBD |
| Prompt version | TBD | TBD | TBD |
| Validation threshold metric | TBD | TBD | TBD |
| Direct-LLM evaluation scope | TBD | TBD | TBD |
| Cost ceiling | TBD | TBD | TBD |

No row may remain `TBD` when this contract is frozen.

## Dataset Contract

### Candidate and Exclusion Log

| Candidate | Exact version | Considered because | Decision | Exclusion reason |
|---|---|---|---|---|
| WDC Products | Initial release 2022-12-22; `80pair.zip`; pair-wise 80% corner-cases, small development, 100% unseen test | Existing project stress benchmark; hardest corner-case/unseen/small-data combination | Selected as Dataset 1 | Not excluded |

### Selected Datasets

Complete one subsection for each of exactly three selected datasets.

#### Dataset 1 — WDC Products Hard Variant

| Field | Frozen value |
|---|---|
| Dataset ID | `wdc_products_80cc_small_100un` |
| Display name | WDC Products — pair-wise 80% corner-cases, small development set, 100% unseen-products test |
| Exact version/release | WDC Products initial release dated 2022-12-22; `80pair.zip`; selected files dated 2022-12-20 inside the archive |
| Authoritative source URL | [WDC Products benchmark](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) and [official `80pair.zip`](https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip) |
| License and usage conditions | The associated benchmark-construction code is BSD-3-Clause, but the official download page does not state a separate dataset license. Academic experiment use is intended by the public benchmark page; redistribution terms must be confirmed before contract freeze or artifact sharing. |
| Acquisition method | Download the official `80pair.zip`; verify SHA-256 before reading; never redistribute the raw archive until license terms are confirmed |
| Source files | `wdcproducts80cc20rnd000un_train_small.json.gz`; `wdcproducts80cc20rnd000un_valid_small.json.gz`; `wdcproducts80cc20rnd100un_gs.json.gz` |
| Source checksum algorithm and values | SHA-256 archive: `b2044939cee5ea6f12148a2f3551508de3cb77660dfc91767c44daaf9d8a9c4a`; compressed train member: `1915a92de76ddbf63a6c4d7ff3162df98f505033a8ad12ee7072b78b083d77c6`; validation: `892e8d39cc8230dce5039a4bded16be1237d5c5c37b94a37404226d349ef3df8`; test: `258f9b408e715410f07480bf4ad39788a10d5cee1512e322091de43dafae1297` |
| Official train split | `wdcproducts80cc20rnd000un_train_small.json.gz`: 2,500 pairs; 500 matches; 2,000 non-matches |
| Official validation split | `wdcproducts80cc20rnd000un_valid_small.json.gz`: 2,500 pairs; 500 matches; 2,000 non-matches |
| Official test split | `wdcproducts80cc20rnd100un_gs.json.gz`: 4,500 pairs; 500 matches; 4,000 non-matches; 100% unseen products |
| Validation policy if no official split exists | Not applicable; use the official validation file without resplitting |
| Expected row counts by split | Train 2,500; validation 2,500; test 4,500 |
| Expected match/non-match balance by split | Train 500/2,000; validation 500/2,000; test 500/4,000 |
| Left-record fields | `id_left`, `cluster_id_left`, `title_left`, `description_left`, `brand_left`, `price_left`, `priceCurrency_left` |
| Right-record fields | `id_right`, `cluster_id_right`, `title_right`, `description_right`, `brand_right`, `price_right`, `priceCurrency_right` |
| Raw pair and record identifiers | Official `pair_id = id_left#id_right`; offer IDs are `id_left`/`id_right`; product truth is represented by `cluster_id_left`/`cluster_id_right` and must not enter LLM prompts |
| Missing-field policy | Preserve missing/null source attributes; deterministic serialization renders them explicitly as `<missing>`; never impute product truth |
| Within-split duplicate-pair policy | Fail on duplicate official pair IDs or duplicate side-order-invariant canonical pair fingerprints |
| Cross-split pair-overlap policy | Fail on any repeated canonical offer pair, including reversed sides |
| Cross-split entity-overlap policy | Preserve official splits. Allow and report seen-product overlap between train and the 0%-unseen validation split; require and verify no train/validation product overlap with the selected 100%-unseen test split. Record overlap counts in manifests. |
| Known limitations | Labels originate from identifier-based clustering; WDC reports about 4% estimated label noise in its manually checked sample. Splits are class-imbalanced. Product data was extracted in 2020. Separate dataset redistribution terms are not explicit on the download page. |

Evidence and rationale: WDC describes 80% corner-cases as its highest
corner-case difficulty, 100% unseen as its strongest unseen-entity setting,
and the 2,500-pair development set as the small setting. The combination
therefore exercises all three hard directions while preserving official files
and splits. The official page documents five offer attributes, pair IDs,
labels, hard-negative metadata, split isolation at the offer level, and the
published row counts. Local archive inspection reproduced those filenames,
counts, and hashes. See
[`260821-1540-wdc-qwen-selection-evidence.md`](./260821-1540-wdc-qwen-selection-evidence.md).

#### Dataset 2 — TBD

Use the same field table as Dataset 1.

Evidence and rationale: TBD.

#### Dataset 3 — TBD

Use the same field table as Dataset 1.

Evidence and rationale: TBD.

### Dataset Identity and Serialization Rules

| Rule | Frozen value |
|---|---|
| Dataset ID grammar | TBD |
| Version grammar | TBD |
| Namespaced pair ID grammar | TBD |
| Canonical record identity | TBD |
| Side-order-invariant pair fingerprint | TBD |
| Row ordering | TBD |
| JSON key ordering and encoding | TBD |
| Newline and null-value policy | TBD |
| Source and normalized manifest schema version | TBD |

## Compact Cross-Encoder Contract

### Eligibility Rules

- Each selected model must jointly encode both records before producing a match
  score or answer.
- The same config and hyperparameters must be used for gold and LLM-hard-label
  arms; only target provenance may differ.
- Final-test performance must not influence model selection.

Additional eligibility rules: TBD.

### Candidate and Exclusion Log

| Candidate | Architecture/backend | Parameter scale | Decision | Exclusion reason |
|---|---|---:|---|---|
| `Qwen/Qwen3-Reranker-0.6B` | Instruction-aware causal-LM text reranker; repository backend `generative_reranker` | 0.6B | Selected as Model 1 | Not excluded |

### Selected Models

Complete one row for each of exactly three selected models.

| Field | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Model ID | `qwen3-reranker-0-6b` | TBD | TBD |
| Repository | [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | TBD | TBD |
| Architecture/backend | Qwen3 causal LM adapted as an instruction-aware pointwise generative reranker; repository backend `generative_reranker` | TBD | TBD |
| Joint-encoding evidence | The official reranker scores one prompt containing the instruction, Query, and Document in the same causal-LM forward pass. The ER adapter maps Record A to Query and Record B to Document and scores the final `no`/`yes` logits. | TBD | TBD |
| Parameter count | 0.6B | TBD | TBD |
| License | Apache-2.0 | TBD | TBD |
| Context length | 32K advertised by the model card; model config has `max_position_embeddings = 40960` | TBD | TBD |
| Tokenizer source | Same repository as the model | TBD | TBD |
| Pair separator/prompt | Official Qwen reranker `Instruct`/`Query`/`Document` interface. ER instruction: `Determine whether Record A and Record B describe the same real-world product. Answer yes only when they refer to the same product.` | TBD | TBD |
| Maximum input length | 4,096 experiment tokens, retained from the verified repository implementation; this is a conservative experiment cap, not the native context limit | TBD | TBD |
| Pair-aware truncation policy | No truncation. Tokenize the complete prompt and fail preflight if it exceeds 4,096 tokens. Any revised cap requires contract review before results. | TBD | TBD |
| Match-label/logit mapping | `no` / non-match = 0; `yes` / match = 1; softmax only over the final single-token `no` and `yes` logits | TBD | TBD |
| Tuning method | LoRA: rank 8, alpha 16, dropout 0.05; target `q_proj`, `k_proj`, `v_proj`, `o_proj`; gradient checkpointing enabled | TBD | TBD |
| Precision and hardware requirements | Base repository weights are BF16. Prior local screening resolved FP16 on a Tesla T4 with microbatch 1; final full-label precision and GPU ceiling remain to be frozen after all-dataset length/resource audits. | TBD | TBD |
| Resource stop condition | Fail on over-limit input, invalid answer-token mapping, non-LoRA trainable base parameters, OOM, or inability to reproduce saved probabilities. Do not silently truncate or shrink the selected model after results. | TBD | TBD |
| Eligibility evidence | Official model card classifies it as a 0.6B text reranker and documents the joint Query/Document interface. Repository screening successfully trained, saved, reloaded, and evaluated this model without single-class collapse. | TBD | TBD |
| Known limitations | Generative `yes`/`no` scoring needs exact prompt/token compatibility. Prior screening used only 128 labels and does not determine full-label hyperparameters. WDC contamination risk is unknown because the model training disclosure does not enumerate all datasets. | TBD | TBD |

### Frozen Training Settings

| Setting | Frozen value |
|---|---|
| Optimizer | TBD |
| Learning-rate schedule | TBD |
| Batch size | TBD |
| Gradient accumulation | TBD |
| Epoch limit | TBD |
| Early-stopping policy | TBD |
| Checkpoint-selection metric | TBD |
| Threshold-selection metric and tie-break | TBD |
| Determinism policy | TBD |
| Save/reload numeric tolerance | TBD |

Model-specific exceptions are prohibited unless documented and approved here:
TBD.

## LLM Machine-Labeling Contract

| Field | Frozen value |
|---|---|
| Provider | TBD |
| Model ID | TBD |
| Model revision/snapshot semantics | TBD |
| Prompt ID and version | TBD |
| Temperature | TBD |
| Maximum output tokens | TBD |
| Other decoding parameters | TBD |
| Allowed answer grammar | TBD |
| Parser version and normalization | TBD |
| Maximum attempts | TBD |
| Retryable failures | TBD |
| Terminal reject behavior | TBD |
| Automatic fallback | None |
| Pricing source and snapshot date | TBD |
| Input-token price | TBD |
| Output-token price | TBD |
| Cost-estimation method | TBD |
| Paid-labeling cost ceiling | TBD |
| Mid-run stopping rule | TBD |

### Frozen Prompt

Prompt ID/version: TBD.

```text
TBD
```

The prompt payload may include only the dataset/version identity, pair ID, and
serialized left/right records required for matching. It must exclude train gold
labels, validation/test labels, entity or cluster truth, selection metadata,
and any answer-derived field.

### LLM Cache and Request Identity

The identity must cover at least the following fields. Final field names and
hash construction are TBD.

- Dataset ID and version.
- Normalized training-split hash.
- Pair ID and serialized-input hash.
- Provider and model ID/revision.
- Prompt ID/version and prompt-content hash.
- Temperature, token limit, and other decoding settings.
- Parser/schema version.
- Request-attempt identity.

Raw responses, parsed labels, validity, token usage, price, timestamps, and
reject state are retained without secrets or provider headers.

## Experiment Matrix

| Dimension | Frozen value |
|---|---|
| Datasets | Exactly 3; IDs TBD |
| Compact models | Exactly 3; IDs TBD |
| Training-label sources | `gold`, `llm_hard` |
| Runs per cell | 1 |
| Compact-model cells | 18 |
| Logical direct-LLM baselines | 3 |

Gold training targets copy benchmark train labels. LLM-hard-label targets must
have exactly one valid label for every frozen training pair. Missing, duplicate,
extra, invalid, or identity-mismatched labels block target publication.

## Evaluation Contract

| Field | Frozen value |
|---|---|
| Primary metric | Match-class F1 |
| Supporting accuracy metrics | Match precision/recall, macro F1, accuracy, TP/FP/TN/FN |
| Invalid-output metric | TBD |
| Validation checkpoint metric | TBD |
| Validation threshold metric and tie-break | TBD |
| Compact-model test ID manifest | TBD |
| Direct-LLM cost-only sample scope | TBD |
| Direct-LLM final accuracy scope | Exact compact-model test IDs; details TBD |
| Cross-dataset summary | Macro average across datasets; never pooled rows |
| Statistical claims | Descriptive only; no repeated-run variance or significance claim |

Validation selects checkpoints and thresholds. Final-test evaluation is a
separate gated stage. A smaller direct-LLM cost sample must be labeled
non-comparable and must not be reported as the accuracy baseline.

## Leakage and Data-Integrity Rules

- Gold train labels feed only the gold-supervision arm.
- Validation and test gold labels are evaluation-only.
- Machine labeling operates on frozen training pairs only.
- Compact-model training and inference never call the LLM labeler.
- Gold/entity/cluster truth never reaches an LLM labeling prompt.
- Gold and LLM targets within a dataset use byte-identical pair order and input
  text; only target and provenance fields may differ.
- Direct-LLM and compact-model accuracy use the same frozen test IDs.
- Artifact, target, cache, and run identities are namespaced by all relevant
  scientific inputs and fail closed on mismatch.

Additional rules and enforcement evidence: TBD.

## Contamination Audit

Rate each selected dataset/model pairing or relevant component as `low`,
`unknown`, or `high` risk. Unknown risk must remain explicit and limit claims.

| Component | Dataset | Release/training evidence | Risk | Claim limitation |
|---|---|---|---|---|
| LLM labeler | TBD | TBD | TBD | TBD |
| Qwen3-Reranker-0.6B | WDC Products | WDC data was extracted in 2020 and the benchmark was released in 2022; Qwen3-Reranker was released later. Its public training disclosure does not establish whether WDC Products or derivative data was included. | Unknown | Treat results as task performance, not proof of generalization from a contamination-free pretrained model; disclose the unknown risk. |
| Model 2 | TBD | TBD | TBD | TBD |
| Model 3 | TBD | TBD | TBD | TBD |

## Timing, Cost, and Break-Even Contract

| Field | Frozen value |
|---|---|
| LLM-labeling token and cost fields | TBD |
| Compact-model training timer boundaries | TBD |
| Compact-model inference timer boundaries | TBD |
| Direct-LLM timer/cost boundaries | TBD |
| Throughput definition | TBD |
| GPU-hour sensitivity scenarios | TBD |
| Gold-label acquisition cost representation | `benchmark-provided` / `unavailable`; never assumed zero |
| Per-model labeling-cost allocation | Charge full labeling cost |
| Optional shared-label portfolio view | TBD |

For each compact-model deployment comparison, use:

```text
N* = (C_label + C_train) / (c_direct - c_model)
```

Report no finite break-even when `c_direct - c_model <= 0`. Whether outputs
include both the raw ratio and the rounded-up whole-query count is TBD.

## Provenance and Artifact Contract

Each run manifest must record or hash:

- Git commit and dirty-worktree policy: TBD.
- This contract file and its frozen commit/hash.
- Used dataset configs and normalized split manifests.
- Used target files and target manifests.
- Used model configs and model identities.
- Prompt ID/version and content hash where applicable.
- Runtime, hardware, dependency, and precision identity.
- Validation, threshold, checkpoint, and final-test scope manifests.
- Cost-assumption file and hash.

Canonical artifact schemas and path identities: TBD.

## Manual Confirmation Gates

### Before Paid Machine Labeling

- [ ] Exactly three datasets and their evidence are approved.
- [ ] Exactly three models and their eligibility evidence are approved.
- [ ] LLM provider, model, prompt, parser, and retry policy are approved.
- [ ] Dry-run token and cost estimate is reviewed.
- [ ] Projected cost is within the frozen ceiling.
- [ ] Prompt leakage review passes.
- [ ] Researcher explicitly approves paid labeling.

Paid generation must additionally require the explicit CLI flag
`--confirm-paid-labeling` after this checklist is complete.

### Before Final-Test Evaluation

- [ ] All six training-target manifests validate.
- [ ] All three model configurations pass smoke verification.
- [ ] All 18 validation cells are complete and provenance-valid.
- [ ] Checkpoints and thresholds were selected using validation only.
- [ ] Compact-model and direct-LLM test ID scopes are identical.
- [ ] Final-test cost estimate and provider requirements are reviewed.
- [ ] Researcher explicitly approves final-test evaluation.

Final testing must additionally require the explicit CLI flag
`--allow-final-test`. Any separate paid-direct-call confirmation requirement is
TBD.

## Freeze Checklist

- [ ] Every `TBD` is resolved or explicitly marked not applicable with rationale.
- [ ] Exactly three datasets and three models are selected.
- [ ] All factual claims have authoritative evidence links.
- [ ] Dataset versions, model identities, licenses, checksums, and pricing dates are explicit.
- [ ] Leakage, overlap, retry, reject, and stopping policies are unambiguous.
- [ ] Experiment matrix, metrics, threshold policy, and scopes are frozen.
- [ ] Contamination risks and claim limitations are recorded.
- [ ] Cost ceiling and break-even interpretation are frozen.
- [ ] Researcher and supervisor review requirements are satisfied.
- [ ] The final document is frozen in a dedicated Git commit.

## Decision Log

| Date | Decision | Rationale and evidence | Approved by | Commit |
|---|---|---|---|---|
| 2026-08-21 | Initial contract skeleton created | Phase 1 starting point; no scientific choices frozen | Researcher | Uncommitted |
| 2026-08-21 | Select WDC Products hard variant as Dataset 1 | Highest corner-case ratio, small development set, and 100% unseen test provide the intended stress benchmark; official files and local hashes recorded | Researcher | Uncommitted |
| 2026-08-21 | Select Qwen3-Reranker-0.6B as Model 1 | The repository implementation passed reranker screening and satisfies the joint-record scoring requirement | Researcher | Uncommitted |
| 2026-08-24 | Freeze a WDC/Sol-high vertical-slice exception for full train labeling | Researcher requested completing one WDC/Qwen experiment before the remaining contract; the narrow paid-labeling checklist, exact prompt, reuse set, and USD 5 ceiling are frozen in `wdc-sol-high-vertical-slice-contract.md` | Researcher | Uncommitted |

## References

- Active migration plan: `plans/260820-1507-full-label-er-migration/plan.md`
- Phase 1: `plans/260820-1507-full-label-er-migration/phase-01-freeze-experiment-contract.md`
- [WDC Products official benchmark page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/)
- [WDC Products construction repository](https://github.com/wbsg-uni-mannheim/wdcproducts)
- [Qwen3-Reranker-0.6B official model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- Selection evidence report: `plans/260820-1507-full-label-er-migration/research/260821-1540-wdc-qwen-selection-evidence.md`
- Additional authoritative dataset, model, and provider references: TBD.

## Next Steps

1. Research and select the exact three datasets.
2. Research and select the exact three eligible compact cross-encoders.
3. Freeze the LLM labeler, prompt, evaluation, cost, and provenance decisions.
4. Review all evidence and manual gates with the researcher.
5. Remove all unresolved placeholders and freeze the contract by Git commit.
