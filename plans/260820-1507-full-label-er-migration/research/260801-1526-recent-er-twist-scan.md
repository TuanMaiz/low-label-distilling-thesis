# Research Report: Recent ER Twists for Cost-Aware LLM Labeling

---
date: 2026-08-01 15:26 +07:00
status: discussion
scope: recent ER and EM papers from 2024-2026
---

## Summary

Keep the original thesis: use an LLM once to label training data, train compact
students, then compare quality and cost with repeated direct LLM matching.

Recommended twist: replace independent pairwise teacher labeling with
**candidate-set-aware (listwise) teacher labeling**. Present one anchor and
several candidates in a single prompt, ask the teacher to select all matching
candidates, convert the response back into pairwise hard labels, and train the
same students. This combines two recent lines of work without replacing the
original cost claim:

1. LLM-generated training labels for compact EM students.
2. Record-interaction/selecting prompts that outperform independent binary
   matching in direct LLM inference.

The unresolved research gap is whether listwise record interaction also
produces better or cheaper **training supervision for compact students**.

## Research Method

- Sources: primary paper pages from ACL Anthology, arXiv, and OpenReview.
- Recency: 2024-2026.
- Search focus: LLM labeling, global consistency, candidate interaction,
  serialization, calibration, positive-unlabeled ER, hard negatives, and cost.
- Selection criterion: an idea must reuse the existing teacher/student pipeline,
  preserve cost as a central outcome, and avoid a combinatorial experiment grid.

## Relevant Recent Work

| Paper | Main idea | Implication for this thesis |
|---|---|---|
| Steiner and Bizer (2026), *Labeling Training Data for Entity Matching Using Large Language Models* | LLM labels selected pairs; compact students learn from hard labels; evaluates selection, teacher, post-processing, student, and cost | Nearest work and baseline design; independent training-pair labeling leaves room for a different labeling interface |
| Wang et al. (2025), *Match, Compare, or Select?* | Binary pairwise matching ignores record interactions; selecting over candidates improves effectiveness; ComEM combines strategies | Direct inspiration for listwise teacher annotation, but their contribution targets LLM matching rather than downstream student training-data generation |
| Yin et al. (2025), *How to Talk to Language Models* | EM performance depends strongly on structured-record serialization | Supports a smaller robustness ablation: field order and record-order consistency |
| Kamsteeg et al. (2025), *Confidence Calibration in LLM-Based Entity Matching* | Temperature scaling reduces overconfidence in a RoBERTa EM model | Useful secondary metric, but calibration alone would be a weaker thesis twist |
| Wang et al. (2025), *PUER* | Positive-unlabeled ER; stresses scarce positives and difficult negative construction | Supports analyzing candidate-set class balance and hard-negative quality |
| Guo et al. (2026), *CaRL-EM* | RL controller dynamically chooses LLM operators/model capacities under a quality-cost objective | Makes adaptive inference routing crowded; avoid making it the main twist |
| Wadhwa et al. (2024), *Learning from Natural Language Explanations* | Explanation distillation improves out-of-domain generalization in their setup | Rationale is unsuitable as our main twist because our existing rationale result is negative |

## Candidate Twists

### 1. Candidate-Set-Aware Teacher Labeling — Recommended

Pairwise arm:

```text
(anchor, candidate_1) -> match/non-match
(anchor, candidate_2) -> match/non-match
...
```

Listwise arm:

```text
(anchor, [candidate_1 ... candidate_k]) -> matching candidate IDs
                                      -> pairwise hard labels
                                      -> compact student
```

Research question:

> Under the same LLM-labeling cost, does candidate-set-aware teacher labeling
> produce higher-quality training labels and compact EM students than
> independent pairwise teacher labeling?

Minimum comparison:

| Arm | Teacher interface | Student training format |
|---|---|---|
| Gold context | benchmark labels | pairwise |
| Pairwise LLM labeling | one pair per decision | pairwise |
| Listwise LLM labeling | one anchor plus multiple candidates | pairwise after conversion |
| Direct LLM matcher | fixed evaluation pairs | none |

Compare at equal **actual teacher cost/token budget**, not merely equal calls.
Reuse each generated training set across all three students.

Primary outcomes:

- downstream match F1 and macro F1;
- teacher-label precision/recall against hidden benchmark labels;
- cost per valid training label;
- positive/negative and hard-negative composition;
- student inference cost and break-even query count.

Main risks:

- candidate groups must be reconstructable from record IDs;
- listwise prompts can become long, especially on WDC;
- position bias requires candidate-order randomization or a small audit;
- candidate sets with multiple true matches need an explicit multi-select output.

### 2. Serialization-Consistency Labeling — Feasible Backup

Label the same pair after swapping left/right records or permuting field order.
Use agreement as a reliability signal and compare breadth versus verification
under the same teacher-token budget. This is easy to integrate but is a smaller
contribution and adds extra teacher calls.

### 3. Hard-Negative-Focused Labeling — Weakest Novelty

Allocate more teacher calls to similar non-matches. This is practically useful,
but overlaps active selection in Steiner and Bizer and the current project
contract. Keep it as analysis or a controlled sampling rule, not the headline.

## Recommendation

Adopt candidate-set-aware teacher labeling as the sole new method. Keep three
datasets, three students, one teacher, hard labels, direct LLM baseline, gold
evaluation, and cost accounting unchanged. Use one pairwise-label arm and one
listwise-label arm. Avoid adding rationale, multiple teachers, RL routing, or
several active-selection variants.

Provisional contribution statement:

> This thesis studies whether record interactions should be exposed not only
> during direct LLM matching, but also while an LLM constructs reusable
> pairwise training data for compact entity-matching students.

## References

- https://arxiv.org/abs/2606.28823
- https://aclanthology.org/2025.coling-main.8/
- https://aclanthology.org/2025.findings-naacl.437/
- https://aclanthology.org/2025.uncertainlp-main.12/
- https://aclanthology.org/2025.findings-emnlp.1336/
- https://aclanthology.org/2026.acl-long.1258/
- https://arxiv.org/abs/2406.09330

## Unresolved Questions

- Can all three datasets reconstruct anchor-centered candidate groups without
  changing their official evaluation semantics?
- What maximum candidate-set size fits the fixed teacher context window and
  preserves a fair cost comparison?
- Should equal cost be enforced by input tokens, estimated dollars, or both?
