---
name: benchmark-paper-template
description: "Structures Benchmark and Evaluation papers using five-pillar framework (Research Gap, Construction Pipeline, Evaluation Framework, Empirical Findings, Companion Method). Returns completeness audit, Introduction logic chain, and section skeleton."
user-invocable: true
when_to_use: "When writing or structuring a benchmark/evaluation paper."
category: research
keywords: [benchmark, evaluation, paper, template, research]
argument-hint: "[benchmark-description]"
metadata:
  author: hkustdial
  version: "1.0.0"
  license: CC-BY-4.0
---

# Benchmark Paper Template

## Overview

A Benchmark paper wins by defining a new evaluation dimension and shipping a construction pipeline that makes measurement high-quality, scalable, and reproducible. This skill scaffolds the five pillars reviewers check, then provides a six-part Introduction chain, Section 2-7 skeleton, and pre-submission checklist.

## Input

Benchmark paper description:
<user-input>$ARGUMENTS</user-input>

## Core Capabilities

1. **Five-pillar completeness audit**: Research Gap, Construction Pipeline, Evaluation Framework, Empirical Findings, Companion Method
2. **Introduction six-part logic chain**: Background + Running Example, Existing-Benchmark Limitations, Research Questions, Design Considerations, Our Proposal, Contributions
3. **Section skeleton for §2-7**: Task and Design Goals, Construction Pipeline, Optional Companion Method, Experiments, Discussion, Related Work, Conclusion
4. **Pre-submission self-check**: four-category reviewer checklist with severity

## The Five Pillars

1. **Research Gap**: What evaluation dimension does existing work miss? Ground gap in concrete failure case, cite 3+ prior benchmarks.
2. **Construction Pipeline**: How to build high-quality, scalable, reproducible data? Paradigms: Reverse Synthesis, Controlled Injection, Adaptive Generation.
3. **Evaluation Framework**: Beyond single score - difficulty tiers, error taxonomy, per-dimension rubrics.
4. **Empirical Findings**: Multi-angle comparisons condensed into bolded *Finding X:* sentences.
5. **Companion Method (optional)**: Specialized model tuned for this benchmark.

## Introduction Six-Part Flowchart

1. Research Background + Running Example (Figure 1)
2. Existing-Benchmark Limitations (at most 3)
3. Research Questions (2-3 RQs)
4. Design Considerations
5. Our Proposal
6. Contributions (typically 4 items)

## Core Procedure

### Step 1: Five-pillar completeness audit

Check each pillar is covered. Report gaps with severity.

### Step 2: Introduction six-part logic chain

Fill the flowchart with user's content.

### Step 3: Section skeleton §2-7

For each section, produce one-paragraph sketch naming the figure/table that carries its weight.

### Step 4: Pre-submission self-check

Walk four-category checklist. Report CRITICAL/MAJOR unresolved items.

### Step 5: Integrity gate

Before emitting:
1. All five pillars addressed
2. Research gap grounded in concrete failure case
3. Construction pipeline specifies sources, generation, QC
4. Evaluation framework explains taxonomy rationale
5. Empirical findings are bolded and actionable
6. Benchmark comparison table included

## Output Format

### 1. Five-pillar completeness
| Pillar | Covered? | Content | Improvement |
|---|---|---|---|
| Research Gap | Y/N | ... | ... |
| Construction Pipeline | Y/N | ... | ... |
| Evaluation Framework | Y/N | ... | ... |
| Empirical Findings | Y/N | ... | ... |
| Companion Method | Y/N/NA | ... | ... |

### 2. Introduction six-part chain
| Part | Content |
|---|---|
| 1. Background + Running Example | ... |
| 2. Existing-benchmark limitations | ... |
| 3. Research Questions | ... |
| 4. Design Considerations | ... |
| 5. Our Proposal | ... |
| 6. Contributions | ... |

### 3. Section outline §2-7
For each section, one-paragraph sketch.

### 4. Pre-submission self-check
Report CRITICAL/MAJOR items.

### 5. Integrity gate result
- Gate 1-5: <pass or fail>
