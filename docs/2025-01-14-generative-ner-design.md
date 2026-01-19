# Design Report: Generative Approach for Multilingual Name Entity Resolution

## 1. Project Overview

**Research Question:** Can a generative sequence-to-sequence approach (inspired by mGENRE) improve multilingual name entity resolution compared to traditional methods (transliteration + phonetic matching + PSL)?

**Key Hypotheses:**
- **Context-aware matching:** Generative models can leverage surrounding information better than string similarity algorithms
- **End-to-end learning:** No need for hand-crafted features like Double Metaphone, alignment heuristics, or PSL rules

---

## 2. Background & Baseline

### Paper 1 (B paper): "Multilingual Entity Matching"

Uses transliteration (Cyrillic → Latin), alignment, and Double Metaphone phonetic encoding with Probabilistic Soft Logic (PSL) for collective entity resolution.

**Best result:** F1 = 0.79 (Translit + Align + Phonetic)

**Dataset:** 470 people from 64 families, Russian ↔ English names

### Paper 2 (Q1): "mGENRE - Multilingual Autoregressive Entity Linking"

Sequence-to-sequence model (mBART) generating entity names token-by-token with constrained beam search. Uses multilingual entity representations with marginalization.

**Results:** State-of-the-art on Mewsli-9, TR2016hard, TAC-KBP2015

---

## 3. Proposed Approach

### 3.1 Task Formulation (Primary - Approach B)

**Canonical Name Generation:**

Given Record A (with name, age, gender in language L1), generate what Record B's name *should* be in language L2. Then compare the generated name with the actual Record B name to determine if they match.

```
Input:  "[TASK] Translate to English: [AGE:45] [GENDER:M] Владимир Владимирович Путин"
Output: "Vladimir Vladimirovich Putin"
Score:  similarity(generated, actual)
```

### 3.2 Secondary Formulation (If time permits - Approach C)

**Joint Embedding + Generation:**

Generate a shared "entity representation" from each record independently, then compare representations.

### 3.3 Why This Works

| Traditional Approach | Generative Approach |
|---------------------|---------------------|
| Hand-crafted transliteration rules | Learned transliteration patterns |
| Fixed phonetic encoding (Double Metaphone) | Context-sensitive encoding |
| No use of personal attributes | Attributes as conditioning signal |
| Brittle to unseen names | Can generalize via language model |

---

## 4. Model Architecture

### 4.1 Model-Agnostic Design

The implementation will support multiple base architectures:

| Model | Parameters | Languages | Notes |
|-------|-----------|-----------|-------|
| **mBART** | 406M | 125 | Original mGENRE base |
| **NLLB** | 3.3B | 200 | Newer, translation-focused |
| **mT5** | Various | 101 | Different architecture |
| **Aya** | - | 101 | Newest option |

### 4.2 Input Format

```
[特殊TOKEN_任务] [特殊TOKEN_语言] [特殊TOKEN_年龄:XX] [特殊TOKEN_性别:X] <姓名>
```

Example:
```
[TRANSLATE] [RU→EN] [AGE:68] [GENDER:M] Владимир Путин
```

### 4.3 Training Objective

Standard seq2seq cross-entropy with:
- Teacher forcing
- Label smoothing (0.1)
- Length normalization for scoring

---

## 5. Dataset

### 5.1 Primary Dataset: Paper 1's Wikidata Dataset

| Statistic | Value |
|-----------|-------|
| People | 470 |
| Families | 64 |
| Test split | 134 people |
| Matching pairs | 134 |
| Non-matching pairs | 62,846 |
| Class imbalance | ~0.2% positive |

### 5.2 Data Sources (Priority Order)

1. **Paper 1's exact dataset** - if authors made it available
2. **Reconstruct from Paper 1's description** - extract from Wikidata using same methodology
3. **Fresh extraction** - query Wikidata API for people with multilingual names

---

## 6. Evaluation

### 6.1 Metrics (Same as Paper 1)

- **Precision, Recall, F1** for both classes (Same/Different)
- **Macro-average** and **Micro-average** F1
- Direct comparability with baseline results

### 6.2 Baseline Comparison

| Method | F1 (Names only) |
|--------|-----------------|
| Baseline (string similarity) | 0.33 |
| Translit | 0.61 |
| Translit + Align | 0.60 |
| **Translit + Align + Phonetic** | **0.79** |

**Goal:** Surpass 0.79 F1 with generative approach.

### 6.3 Ablation Studies

1. Names only vs. Names + attributes
2. Different model architectures
3. With/without task prompting
4. Zero-shot transfer to unseen languages

---

## 7. Implementation Plan

### 7.1 Project Structure

```
code/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Preprocessed data
│   └── Wikipedia/              # Wikidata extraction scripts
├── models/
│   ├── base.py                 # Model-agnostic base class
│   ├── mgenre_adapter.py       # mGENRE-style adapter
│   └── architectures/          # Specific model implementations
│       ├── mbart.py
│       ├── nllb.py
│       ├── mt5.py
│       └── aya.py
├── experiments/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── config/                 # Experiment configs
├── utils/
│   ├── metrics.py              # Evaluation metrics
│   ├── data_loader.py          # Data loading utilities
│   └── wandb_logger.py         # Weights & Biases integration
├── notebooks/
│   └── colab_driver.ipynb      # Colab driver notebook
├── main.py                     # Main entry point
└── requirements.txt
```

### 7.2 Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | HuggingFace Transformers | Best model support, Colab-friendly |
| Deep Learning | PyTorch | Native to HF, flexible |
| Experiment Tracking | Weights & Biases | Easy integration, good visualization |
| Development | Hybrid (modular + notebook) | Code organization + interactive experimentation |
| Environment | Google Colab Pro | GPU access, no local resource constraints |

### 7.3 Development Workflow

```
1. Write modular code in .py files
2. Use Colab notebook as "driver" with !python commands
3. Track experiments with W&B
4. Refactor reusable components into modules
```

---

## 8. Implementation Phases

### Phase 1: Setup & Data (Week 1)
- [ ] Set up project structure
- [ ] Install dependencies (HuggingFace, W&B, etc.)
- [ ] Obtain/reconstruct Paper 1's dataset
- [ ] Implement data loading and preprocessing

### Phase 2: Base Model (Weeks 2-3)
- [ ] Implement model-agnostic base class
- [ ] Implement canonical name generation formulation
- [ ] Set up training loop with W&B logging
- [ ] Implement evaluation metrics (matching Paper 1)

### Phase 3: First Model Implementation (Week 4)
- [ ] Integrate mBART (closest to mGENRE)
- [ ] Run initial experiments
- [ ] Compare against Paper 1 baseline (F1: 0.79)

### Phase 4: Multi-Model Comparison (Weeks 5-6)
- [ ] Integrate NLLB model
- [ ] Integrate mT5 model
- [ ] (Optional) Integrate Aya model
- [ ] Benchmark all models against each other and baseline

### Phase 5: Analysis & Extension (Weeks 7-8)
- [ ] Ablation studies
- [ ] Qualitative error analysis
- [ ] (Optional) Implement Approach C (joint embedding)
- [ ] (Optional) Expand to more languages

### Phase 6: Thesis Writing (Weeks 9-12)
- [ ] Document results
- [ ] Write thesis chapters
- [ ] Prepare figures and tables

---

## 9. Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary hypothesis | Context-aware + End-to-end | Focus on learned vs hand-crafted |
| Dataset | Paper 1's Russian-English | Established benchmark, direct comparison |
| Task formulation | Canonical name generation (B) | Closest to mGENRE, directly comparable |
| Extension | Joint embedding (C) | If time permits, adds novelty |
| Models | Multiple newer models | Model-agnostic + benchmark contribution |
| Features | Names + task prompt + basic attributes | Lean start, enough context for generation |
| Evaluation | Same as Paper 1 | Direct comparability |
| Framework | HuggingFace + PyTorch | Best ecosystem, Colab-friendly |
| Tracking | Weights & Biases | Experiment management for thesis |
| Workflow | Hybrid (modular + notebook) | Organization + flexibility |

---

## 10. Success Criteria

The project will be considered successful if:

1. **Primary goal:** Generative approach achieves F1 ≥ 0.79 (matches or exceeds Paper 1's best)
2. **Secondary goal:** Demonstrates clear advantage in context-aware matching scenarios
3. **Contribution:** Model-agnostic framework + multi-model comparison
4. **Thesis output:** Complete analysis with ablation studies and discussion

---

## 11. References

1. Mustafin, I. et al. "Multilingual Entity Matching." (B paper)
2. De Cao, N. et al. "Multilingual Autoregressive Entity Linking (mGENRE)." TACL 2022 (Q1 paper)
