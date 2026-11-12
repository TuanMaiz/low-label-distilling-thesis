---
title: Treating flagged training labels — deep research report
date: 2026-11-12
status: research note (literature); no decision taken
provenance: Claude deep-research report, 2026-11-12, written from outside sources. Numbers from 2026 preprints are provisional and should be checked against the cited versions before use in the thesis. The error-budget figures (≈80 errors on WDC, ≈28 flagged) are derived from our own rates and are approximate.
related: docs/research/literature-uncertain-label-treatment.md (2026-11-11)
---

# Treating Flagged Training Labels: What Exists, When It Helps, and What It Means for LLM-Labelled Entity Matching

Across the label-noise literature, treatment after detection reliably helps only when the label noise is substantial and the downstream model has real room to gain; in our setup (a ~3% teacher error rate, mostly systematic, unanimous errors, and a gold-trained reference G that does not beat the LLM-trained student A), the expected benefit of any treatment is close to zero. The arm C/B result is therefore the expected outcome, not a failed experiment. The best remaining options are cheap "soft" treatments (vote-based soft targets, down-weighting instead of removal, blending in out-of-fold student probabilities), plus a stricter independent-consensus rule for flips. Even these should be framed as testing whether a ceiling of roughly 0–1 F1 can be approached, not as a way to rescue a large gain.

## TL;DR

- **Treatment families exist and work, but under conditions we don't have.** Filtering, relabeling, reweighting, soft labels, robust losses, loss correction, semi-supervised/co-teaching methods, and human/LLM adjudication all have positive evidence, but mostly at noise rates of 20–60% in synthetic or crowdsourced settings. With pretrained transformers at low noise, studies repeatedly find that noise-handling "does not always improve" and can hurt. For systematic (feature-dependent) noise, the popular methods "add little to no robustness."
- **Our own numbers set the ceiling.** WDC has ~80 wrong labels out of 2,500, and only ~28 of them sit in the 158 flagged pairs. Gold-trained G does not beat A (Δ +0.7 F1, CI [−1.2, +2.7]), and benchmark gold labels themselves carry an error rate comparable to strong LLM teachers. Even perfect cleaning of the flagged set has almost nothing to gain, while removal and flipping pay real costs by discarding hard-but-correct pairs and adding correlated errors.
- **What to try (cheap, ordered):** (1) state the ceiling/power result as a finding; (2) train on soft targets (vote fraction and/or OOF-blended targets) instead of hard labels for flagged pairs; (3) down-weight rather than remove; (4) flip only under independent consensus (a different model family *and* the OOF student must both disagree with the teacher); (5) optionally, hand-relabel the 158 flagged WDC pairs as a bounded human-in-the-loop arm. Skip DivideMix, co-teaching, and meta-reweighting: they need large data, a clean meta set, or the small-loss assumption that systematic noise breaks.

## Key Findings

### 1. Our setup in numbers: why the ceiling is near zero

- **Error budget on WDC.** 96.8% agreement means about 80 of 2,500 labels are wrong. The detector flags 158 pairs with a 17.7% error rate, so about 28 errors are flagged. The remaining 2,342 pairs at 2.2% hold about 52 errors. Roughly two-thirds of the errors are therefore invisible to the detector, consistent with most errors being unanimous (3/3) teacher votes.
- **The gold reference does not beat the LLM labels.** A 76.5 vs G 75.9 on WDC and A 88.8 vs G 89.7 on Companies, with both CIs spanning zero, means that "perfect cleaning" is not a reliably better target on WDC. Companies may have a ~1 F1 ceiling.
- **The "gold" labels are themselves noisy at a similar rate.** The Mannheim EM study (Steiner & Bizer, arXiv 2606.28823) audited WDC Products test pairs by hand: error rates of 1.91% for the benchmark gold labels, 1.82% for GPT-5.2, 2.36% for Qwen 3.6 Plus and 1.60% for Kimi K2.6 ("the teacher error rates differ from the benchmark by 0.5 points on average and by at most 2 points"). G is not a noise-free ceiling, and test-set label errors also limit how much any training-side treatment can show.
- **Our treatment failure has a clear mechanism.** All 10 flips were wrong, 73 of 93 removed pairs were correct, and 8 wrong labels were upweighted: the "replacing noise with noise" failure, compounded by filtering away hard-but-correct pairs.

### 2. Evidence from entity matching with LLM-generated labels

- **Mannheim study (arXiv 2606.28823), closest to this thesis.**
  - Setup: an LLM teacher (GPT-5.2) labels the pairs; students include Ditto/RoBERTa and LoRA-tuned Qwen3-0.6B/1.7B/8B. On WDC Products (80% corner cases, unseen test entities), Ditto on active-learning-selected LLM labels scored 72.17 ± 1.01 F1 vs 71.94 ± 0.97 on benchmark labels.
  - Qwen3-0.6B: 57.14 on benchmark labels and between 53.94 and 60.19 on LLM-labelled sets. Average gain from the best machine-labelled set over benchmark labels: "+0.11 F1 for Ditto, +0.60 for Qwen3-0.6B, −1.08 for Qwen3-1.7B, and −1.61 for Qwen3-8B."
  - Cleaning on WDC: a second conservative LLM review that replaces labels ("Relabel") gave 73.35 (+1.18). Dropping relabeled pairs gave 71.40, a graph "closure drop" filter gave 68.66 (−3.51), and dropping when either filter flagged gave 69.27.
  - Authors' conclusion: "no variant improves performance consistently across benchmarks," only one gain (1.56 F1 on Abt-Buy) "exceeds run-to-run variance," and practitioners should "treat label cleaning as optional." The teacher's self-reported confidence "proved too noisy to be a reliable filter."
  - Version caveat: from v1 of the preprint; a later version reportedly covers eight benchmarks.
- **DistillER (Zeakis et al., arXiv 2602.05452).** Llama-3.1-8B student on Qwen-2.5-32B teacher labels averaged 0.85 F1 vs 0.82 on ground truth (attributed to annotations on "harder or borderline cases"); RoBERTa 0.69 vs 0.66. Majority vote over four LLM teachers gave 0.83 label F1, the same as the best single teacher: "the multi-teacher setting does not justify the effectiveness–efficiency tradeoff." Single runs, no variance reported; per-dataset results mixed.
- **Steiner et al. (arXiv 2409.08185)** fine-tuned LLMs on WDC Products: fine-tuning "significantly improves the performance of the smaller models while the results for the larger models are mixed"; explored error-based example selection rather than label cleaning.
- **Takeaway for EM:** two independent studies find LLM labels about as good as benchmark labels for training students, and cleaning gains inconsistent and usually within run-to-run variance. Our G ≈ A replicates this.

### 3. When treatment helps and when it hurts

- **Noise level.**
  - Brodley & Friedl: filtering "significantly improves classification accuracy for noise levels up to 30%."
  - Zhu et al., "Is BERT Robust to Label Noise?" (ACL Insights 2022): existing noise-handling methods "do not always improve its performance, and may even deteriorate it"; where they help at high noise, "the effect is limited (less than 4%)."
  - Companion study (arXiv 2206.01476): BERT dropped "less than 5% under 60% uniform noise in AG-News"; "noise-handling methods rarely outperform the baseline" at low-to-mild noise.
  - Our ~3% noise is far below where these methods pay off.
- **Systematic vs random noise.**
  - Agro & Aldarmaki, "Handling Realistic Label Noise in BERT Text Classification": feature-dependent noise "degrades BERT performance and these noise handling techniques add little to no robustness at all"; boosting "maintains baseline performance at least."
  - CILN benchmark (Islam, Kristiadi & Milani, arXiv 2606.14965): corruption-mediated instance-dependent noise exposes "failure modes of popular noisy-label learning methods, including Co-Teaching and DivideMix," because it "weakens the small-loss assumption."
  - Our unanimous, systematic teacher errors fit this pattern: plausible, easy to fit, missed by loss- and agreement-based detectors.
- **Hard-but-correct examples.**
  - Dataset Cartography (Swayamdipta et al., EMNLP 2020): "hard-to-learn" instances often correspond to labeling errors, but "ambiguous" instances "contribute the most towards out-of-distribution generalization."
  - Late Stopping and "Mislabeled Easy Examples": small-loss selection excludes clean hard examples; the most harmful mislabeled examples are the easy-looking ones (mean influence 4.96 vs 3.01 for mislabeled hard examples).
  - Our 73-of-93 correct removals is the textbook version of this failure; in EM, corner cases are exactly the informative pairs.
- **Data paucity.** Brodley & Friedl: "for situations in which there is a paucity of data, consensus filters are preferable." Our OR-rule detector behaves like an aggressive (majority-style) filter.
- **Dependent corrector ("replacing noise with noise").**
  - TRACE (arXiv 2608.03432): "a pseudo target learned from corrupted supervision may reproduce the noise it is meant to correct." A public review notes 6.16% high-confidence pseudo-target errors remain and suggests "an ensemble of diverse teachers."
  - CANOLA (arXiv 2606.11695): one correction method (SIDYP) raised the AGNews LLM-label error rate from 19% to 29.9%; CANOLA cut it to 16.3%.
  - Aggregating LLM judges (arXiv 2601.22336): independence is "often violated by LLM judges due to shared data, architectures, prompts, and failure modes." Our same-family evidence check is a dependent source by construction.
- **Evaluation pitfalls.** Test-set label errors can change model rankings (Northcutt et al., "Pervasive Label Errors in Test Sets"); ~1-point effects with CIs of ~±2 F1 cannot be resolved by single-seed comparisons; using gold to design treatments invalidates confirmation (WDC as development set, Companies/Geo-ER as confirmation is the right protocol).

## Treatment families

### A. Removal / filtering
- Brodley & Friedl (JAIR 1999): single, majority and consensus ensemble filters. Confident Learning (Northcutt, Jiang & Chuang, JAIR 2021): prune a number equal to the estimated error count, ranked by margin.
- How much: consensus rather than OR rules; cap at the estimated error count (WDC ≈ 28 flagged errors, not 158); rank by margin.
- Evidence: positive at ≥10–30% noise; neutral to negative for low-noise text with pretrained models (Zhu et al. 2022; Mannheim closure drop −3.51 F1).
- Cost: one K-fold OOF pass (we have it). Fit: poor as primary; if used, consensus-flagged subset capped at ~1% of data.

### B. Relabeling / correction
- TRACE; CANOLA; Mannheim "Relabel" (+1.18 F1 on WDC, within variance).
- Risk: correlated errors when the corrector depends on the original labeler (our 10/10 wrong flips).
- Cost: extra LLM calls per flagged item. Fit: only with an independent corrector (different model family, different evidence, or a human), and only when several independent signals agree.

### C. Reweighting
- Heuristic down-weighting (e.g. weight 0.3–0.5) keeps hard-but-correct pairs while limiting damage. Error-type-aware loss reweighting for NER on LLM labels (arXiv 2608.30827) works in "a single fine-tuning run without additional resources," but robust losses work best "when aligning them with the error distribution," which "can not be known in advance since measuring it requires a small set of clean labels."
- Meta-reweighting (Ren et al., ICML 2018; Meta-Weight-Net, NeurIPS 2019; LLMaAA) needs a clean validation/meta set.
- Fit: heuristic down-weighting cheap and good; meta-reweighting conflicts with the gold-boundary rule. Note: our trainer currently has no per-sample weights (upweighting is by duplication).

### D. Soft labels, label smoothing, distillation
- Lukasik et al. (ICML 2020): label smoothing "is competitive with loss-correction under label noise"; "label smoothing of the teacher is beneficial" when distilling from noisy data.
- Sarfraz et al.: vanilla KD improved generalization at all corruption levels (50% corrupted: 68.54% vs 56.61%).
- SERSAL (medical tabular): hard LLM labels "usually suboptimal" vs soft confidences; RED-CT: soft labels improved edge classifiers on LLM labels.
- Counter-evidence: EffiARA, "hard-label training performs better than soft-label training" for BERT/Llama baselines (soft improved calibration).
- Fit: vote fractions are free (3/3 → 1.0, 2/3 → ~0.67) but reach only 2/3 cases; extend with OOF blending for flagged pairs, y′ = β·y_teacher + (1−β)·p_OOF. Needs soft-target support in our trainer.

### E. Loss correction / transition matrices
- Zhu et al. (2022): no BERT peak gain. Our noise is directional (Companies strict → false negatives; Geo-ER lenient → false positives); a 2×2 class-conditional correction amounts to a threshold/prior shift. Low cost, small expected gain.

### F. Robust losses and regularization
- Burns et al. (OpenAI, weak-to-strong): auxiliary confidence loss raised median performance-gap recovery "from about 25% to nearly 80%", but in the inverse regime (strong student, weak teacher) and "some settings ... degrades performance."
- LNMBench (Ma et al., arXiv 2512.09315; Pattern Recognition 2026): robust regularization (SCE, CDR) did best under real-world noise in medical imaging.
- Fit: low cost, worth one run (GCE/SCE or stronger early stopping), not a priority.

### G. Semi-supervised, co-teaching, co-training
- DivideMix ~94–96% on CIFAR-10 at 20–50% symmetric noise; fails under instance-dependent noise. SERSAL adapted DivideMix to LLM soft labels.
- Cost high; fit poor (2,500 pairs, LoRA reranker, systematic noise).

### H. Curriculum and training dynamics
- Data maps (confidence/variability) separate ambiguous from hard-to-learn items. Cost: one logged run. Fit: moderate, as a diagnostic within the 158 flagged pairs.

### I. Abstention / selective prediction
- Changes the evaluation target (coverage–F1); future work.

### J. Human-in-the-loop relabeling of flagged items
- Wang et al. (Findings EMNLP 2021): humans relabel the lowest-confidence GPT-3 labels; mixed labeling beat all-human and all-GPT-3 at equal budget, 50–96% cost savings. CoAnnotating (EMNLP 2023): "up to 21% performance improvement over random baseline." ActiveAED.
- Cost: our time (158 WDC pairs ≈ 1–2 hours). Fit: good as a bounded arm; its ceiling is still small (~28 of ~80 errors flagged).

### K. LLM jury / different adjudicator
- DistillER 4-LLM vote = best single teacher (0.83). Same-family judges share errors. CROWDLAB-style methods combine votes with OOF probabilities.
- Fit: a different-family adjudicator on the ~158 flagged pairs only, as one vote in a consensus rule, never a sole flipper.

## Summary table

| Method | Key paper(s) | Evidence of benefit | Conditions where it works | Cost | Fit to this thesis |
|---|---|---|---|---|---|
| Removal (OR-rule) | Brodley & Friedl 1999; Northcutt et al. 2021 | + at 10–30%+ noise; neutral/negative for low-noise text | Random noise, large data | Low | Poor (73/93 correct removed) |
| Consensus / count-capped removal | Brodley & Friedl; CL prune-by-noise-rate | Conservative; keeps good data | Small data | Low | Optional, tiny scope |
| Relabel by dependent LLM | TRACE 2026; CANOLA | "Noise with noise" risk | Only with independent corrector | Medium (API) | Poor (10/10 flips wrong) |
| Relabel by independent LLM review | Mannheim 2606.28823 | +1.18 F1 WDC, within variance | Conservative prompt | Medium (API) | Moderate, if consensus-gated |
| Down-weighting flagged | Error-type-aware NER reweighting | Gains in single run with LLM labels | Error profile known | ~Zero (needs weight support) | Good |
| Meta-reweighting | Ren 2018; Shu 2019; LLMaAA | Strong with clean meta set | Clean validation set | Medium–high | Poor (gold constraint) |
| Soft targets (votes / OOF blend) | Lukasik 2020; Sarfraz 2019; SERSAL; RED-CT | + under noise; mixed in EffiARA | Calibrated teacher signal | ~Zero (needs soft-target support) | Best first try |
| Label smoothing | Lukasik 2020 | Competitive with loss correction | Any | ~Zero | Good |
| Transition-matrix correction | Patrini 2017; Zhu 2022 | No BERT peak gain | Class-conditional noise | Low | Moderate (directional noise) |
| Robust losses, early stopping | LNMBench 2512.09315 | SCE/CDR best on real noise there | Moderate noise | Low | Moderate |
| Auxiliary confidence loss | Burns et al. 2023 | PGR 25%→~80% (weak→strong) | Student stronger than teacher | Low | Poor (inverse regime) |
| DivideMix / co-teaching | Li 2020; Han 2018; CILN 2606.14965 | Strong on CIFAR; fails under IDN | Large data, small-loss separable | High | Poor |
| Data maps / curriculum | Swayamdipta 2020 | Hard-to-learn ≈ mislabeled; ambiguous = valuable | One logged run | Low | Good as diagnostic |
| Human relabel of flagged | Wang 2021; CoAnnotating 2023 | Mixed human/LLM > either alone | Uncertainty-ranked | Human hours | Good (bounded arm) |
| Multi-LLM jury | DistillER 2026 | 4-LLM vote = best single (0.83) | Diverse models | Medium–high | Only as one consensus vote |

## Recommendations from the report (not decided)

1. Make the ceiling result a contribution: error budget (~80 errors, ~28 flagged), G ≈ A, gold-label error rates from the Mannheim audit, and a power analysis of what F1 difference the test sets can detect.
2. Soft-target arm: vote-fraction targets for all pairs and an OOF-blended target y′ = 0.5·y + 0.5·p_OOF for flagged pairs, optionally mild label smoothing (ε ≈ 0.05–0.1); fix β and ε on WDC, confirm on Companies/Geo-ER.
3. Down-weight (0.3–0.5) instead of remove; no duplication-upweighting.
4. Independent-consensus flips only: a different model family (or differently-evidenced prompt) disagrees, the OOF student disagrees above a threshold, and the teacher vote was not unanimous or the evidence check found a concrete factual contradiction.
5. Directional bias correction for Companies/Geo-ER (class-conditional correction or class-weight/threshold shift estimated from confident-learning counts on teacher labels only, not from gold).
6. Optional bounded human arm: relabel the 158 flagged WDC pairs blind to gold.
7. Run multiple seeds (≥ 3–5) and report mean ± SD alongside the paired bootstrap.
8. Skip DivideMix, co-teaching, meta-reweighting, auxiliary-confidence loss and multi-LLM juries as primary treatments.

## Caveats (from the report)

- Several key sources are recent 2026 preprints (TRACE, CANOLA, CILN, error-type-aware NER, the Mannheim study); exact numbers are provisional.
- DistillER's comparisons are single runs without variance.
- Much of the treatment literature uses synthetic noise on image benchmarks; transfer to a 0.6B LoRA reranker with ~3% systematic noise is an inference.
- The error-budget arithmetic is approximate.
- Soft labels are not universally better (EffiARA).

## Sources

- https://aclanthology.org/2022.insights-1.8/
- https://ar5iv.labs.arxiv.org/html/2204.09371
- https://aclanthology.org/2023.icnlsp-1.2.pdf
- https://arxiv.org/abs/2606.28823
- https://github.com/wbsg-uni-mannheim/Automatic-data-labeling
- https://www.arxiv.org/pdf/2602.05452
- https://arxiv.org/abs/2409.08185
- https://arxiv.org/pdf/1106.0219
- https://arxiv.org/html/2305.16337
- https://www.aclweb.org/anthology/2020.emnlp-main.746/
- https://arxiv.org/html/2502.08227
- https://arxiv.org/abs/2608.03432
- https://pith.science/paper/2608.03432
- https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf
- https://arxiv.org/pdf/2103.16324
- https://arxiv.org/pdf/2405.09602
- https://arxiv.org/pdf/2606.11695
- https://arxiv.org/pdf/2608.30827
- https://arxiv.org/pdf/2310.19596
- http://papers.neurips.cc/paper/8467-meta-weight-net-learning-an-explicit-mapping-for-sample-weighting.pdf
- https://arxiv.org/html/1803.09050v1
- https://proceedings.mlr.press/v119/lukasik20a.html
- https://arxiv.org/pdf/1910.05057
- https://arxiv.org/pdf/2403.01570
- https://arxiv.org/pdf/2408.08217
- https://arxiv.org/pdf/2410.14515
- https://arxiv.org/pdf/2312.09390
- https://arxiv.org/html/2512.09315v1
- https://arxiv.org/pdf/2606.14965
- https://dl.acm.org/doi/10.1145/3723009
- https://ar5iv.labs.arxiv.org/html/2108.13487
- https://www.semanticscholar.org/paper/Want-To-Reduce-Labeling-Cost-GPT-3-Can-Help-Wang-Liu/4e263b4cd6998bff2501dd143e685f413179b12d
- https://arxiv.org/pdf/2310.15638
- https://arxiv.org/pdf/2305.20045
- https://www.alphaxiv.org/abs/2601.22336
- https://cleanlab.ai/blog/learn/multiannotator/
