# Progress Log - Multilingual Name Entity Resolution (Generative Approach)

**Date:** 2025-01-17
**Status:** Phase 1 & 2 Complete, Ready for Phase 3

---

## Project Overview

**Research Question:** Can a generative seq2seq approach (mGENRE-inspired) improve multilingual name entity resolution compared to traditional methods (transliteration + phonetic matching + PSL)?

**Target:** Beat F1 ≥ 0.79 (Paper 1's best result on Russian-English name matching)

**Approach:**
1. Given Record A (Russian), generate what Record B's name should be in English
2. Compare generated name vs actual name using string similarity (Jaro-Winkler, etc.)
3. Apply threshold for binary classification (MATCH / NO MATCH)

---

## Completed Work

### Phase 1: Setup & Data ✅

**Directory Structure:**
```
code/
├── .venv/                   # uv venv with dependencies
├── data/
│   ├── raw/
│   │   ├── fake_dataset.csv # 22 records (11 people × 2 languages)
│   │   └── fake_pairs.csv   # 37 pairs (12 positive, 25 negative)
│   ├── processed/
│   ├── schema.py            # Pydantic models
│   └── __init__.py
├── models/
│   ├── base.py              # BaseModel abstract class
│   └── architectures/
│       └── mbart.py         # mBART implementation
├── utils/
│   ├── data_loader.py       # CSV loading, formatting
│   └── metrics.py           # Similarity + classification metrics
├── experiments/
│   ├── trainer.py           # Training loop + W&B logging
│   └── evaluate.py          # Evaluation (Paper 1 format)
├── main.py                  # CLI entry point
└── requirements.txt
```

**Data Format (CSV):**
- `fake_dataset.csv`: One person per row with `record_id, person_id, family_id, name, language, age, gender`
- `fake_pairs.csv`: Pairs with `record_a_id, record_b_id, label, split`

**Classes:**
- `PersonRecord`: Single person in one language
- `RecordPair`: Pair for matching (with label: True=match, False=different)
- `Dataset`: Container with records + pairs

### Phase 2: Base Model ✅

**Model Architecture:**
- `BaseModel`: Abstract interface for all seq2seq models
- `MBartModel`: First implementation (facebook/mbart-large-50-many-to-many-mmt)
- Framework: HuggingFace Transformers + PyTorch

**Input Format:**
```
[TRANSLATE] [RU→EN] [AGE:68] [GENDER:M] Владимир Путин
```

**Target Format:**
```
[EN] Vladimir Vladimirovich Putin
```

**Training:**
- Standard seq2seq cross-entropy
- Teacher forcing
- AdamW optimizer
- W&B logging integrated

**Evaluation:**
- Generate target name from source
- String similarity: Jaro-Winkler, Levenshtein, Token F1, Char N-gram, Combined
- Threshold → MATCH/NO MATCH
- Metrics: Precision, Recall, F1 (Paper 1 format)

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Evaluation approach | Generation quality + string similarity | Natural task, easy to explain, Paper 1 compatible |
| Similarity metric | Combined (Jaro-Winkler 40%, etc.) | Robust to variations |
| Framework | HuggingFace + PyTorch | Best ecosystem, Colab-friendly |
| First model | mBART | Closest to mGENRE paper |
| Tracking | W&B | Experiment management for thesis |

---

## Installed Dependencies

```bash
# Core
torch, transformers, pydantic, numpy

# Training
wandb, tqdm

# Data
pandas (if needed)
```

Install with: `uv pip install -r requirements.txt`

---

## How to Use

```bash
# Activate venv
source .venv/bin/activate

# Test setup
python main.py --mode test

# Train (needs GPU/Colab)
python main.py --mode train --model mbart --epochs 10

# Evaluate
python main.py --mode evaluate --checkpoint checkpoints/best_model
```

---

## Remaining Work

### Phase 3: First Model Implementation (Week 4)
- [ ] Run actual training on mBART
- [ ] Compare against Paper 1 baseline (F1: 0.79)
- [ ] Debug and tune

### Phase 4: Multi-Model Comparison (Weeks 5-6)
- [ ] Implement NLLB model
- [ ] Implement mT5 model
- [ ] (Optional) Implement Aya model
- [ ] Benchmark all models

### Phase 5: Analysis & Extension (Weeks 7-8)
- [ ] Ablation studies
- [ ] Qualitative error analysis
- [ ] (Optional) Approach C (joint embedding)

### Phase 6: Thesis Writing (Weeks 9-12)
- [ ] Document results
- [ ] Write thesis chapters

---

## Important Notes

1. **Dataset:** Currently using fake data (22 records, 37 pairs). Need to:
   - Contact Paper 1 authors for their dataset OR
   - Reconstruct from Wikidata OR
   - Start with larger fake dataset

2. **Training:** Requires GPU. Workflow:
   - Code locally (WSL2)
   - Push to GitHub
   - Pull & train in Colab

3. **Model checkpoint location:** `checkpoints/best_model` (created during training)

---

## Quick Commands

```bash
# Navigate to project
cd /mnt/d/Study/Cao-hoc/luan-van/code

# Activate venv
source .venv/bin/activate

# Test everything works
python main.py --mode test

# Install missing packages
uv pip install <package_name>
```

---

## Files to Reference

| File | Purpose |
|------|---------|
| `docs/2025-01-14-generative-ner-design.md` | Original design document |
| `models/base.py` | Model interface (add NLLB/mT5 here) |
| `experiments/evaluate.py` | Evaluation functions |
| `main.py` | Entry point for all operations |
