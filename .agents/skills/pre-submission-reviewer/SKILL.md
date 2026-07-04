---
name: pre-submission-reviewer
description: "Runs pre-submission review across five dimensions: macro logic, writing details, English grammar, LaTeX formatting, and figure quality. Uses reviewer-style severity taxonomy."
user-invocable: true
when_to_use: "3-5 days before submission deadline or after major rewrite."
category: research
keywords: [review, proofread, paper, submission, grammar, latex]
argument-hint: "[paper-content-or-section]"
metadata:
  author: hkustdial
  version: "1.0.0"
  license: CC-BY-4.0
---

# Pre-Submission Reviewer

## Overview

Three to five days before a submission deadline is the window where a careful external review pays off most. This skill produces a structured review across five dimensions with severity-tagged findings and concrete rewrite suggestions. It enforces mechanical rules (no em-dashes, no banned AI-tone vocabulary, leading text per paragraph, topic-sentence discipline) and surfaces common patterns for non-native English speakers.

## Input

Paper or section to review:
<user-input>$ARGUMENTS</user-input>

## Severity Taxonomy

- **CRITICAL**: blocks submission. Example: contributions do not map to sections; introduction flowchart broken; no real-world running example; raster figure; missing key baseline; page-limit violation.
- **MAJOR**: reviewers will flag. Example: topic-sentence absent from 3+ paragraphs; em-dash in 5+ places; banned AI-tone word in 3+ places; Table 1 comparison missing; chart type mismatched.
- **MINOR**: polish. Example: two long sentences; default Matplotlib styling; single article error.

## Core Procedure

### Step 1: Dimension 1 - Macro logic review

Check:
- Introduction flowchart intact (Background, Limitations, Goal/Key Idea, Challenges, Methodology, Contributions)
- Contributions map one-to-one with methodology modules and section numbers
- Experiments validate main claims
- Related Work covers necessary prior art
- Running example consistent across Introduction, Methodology, Experiments

### Step 2: Dimension 2 - Writing details review

Check:
- Every paragraph has a topic sentence
- Paragraphs transition smoothly; no orphan paragraphs
- Paragraphs not over 10 lines; split if so
- No repeated or redundant passages
- Abstract covers problem, method, result

### Step 3: Dimension 3 - English grammar review

Check:
- Article use (a, an, the)
- Subject-verb agreement (third-person singular)
- Tense consistency (Related Work past, method present)
- Passive-voice overuse
- Which versus that
- Sentence length; split at "Specifically,"
- Chinglish patterns

### Step 4: Dimension 4 - LaTeX format review

Check:
- Equation numbering contiguous; every numbered equation referenced
- Figures and tables have captions; captions detailed
- Citations use correct command and non-breaking tilde (ResNet~\cite{X})
- Labels use underscores, not spaces or hyphens
- Vector figure format; no raster
- Page-limit compliance

### Step 5: Dimension 5 - Figure quality review

For each figure:
- Vector format
- Font size large enough post-scaling
- Colour-blind-safe palette; dual encoding
- Self-contained caption with finding in first sentence
- No chartjunk
- Motivated example concrete and failure-revealing
- Solution overview labels match section titles

### Step 6: Banned-vocabulary and em-dash scan

Scan for:
- Em-dashes used as sentence connectors (banned)
- AI-tone words: innovative, pioneering, revolutionary paradigm, transformative framework, superior, surpass, excel, remarkable, unprecedented, breakthrough performance, general-purpose, is capable of, notably, yet, yielding, at its essence, encompass, differentiate, reveal, underscore, pave the way for, highlight the potential of, profound challenges, stems from, rigid, impede

Flag each occurrence. Em-dashes are MAJOR; banned AI-tone words are MAJOR if appearing 3+ times.

### Step 7: Integrity gate

Before emitting:
1. Every finding quotes specific text; no vague claims without citation
2. Every CRITICAL finding has concrete fix suggestion
3. No fabricated quotes; only text actually present
4. Severity assignments follow taxonomy
5. Grammar findings cite specific rule
6. Banned-vocabulary scan run in full
7. Final score matches CRITICAL + MAJOR count

## Output Format

### Summary
- CRITICAL: <n>
- MAJOR: <m>
- MINOR: <k>
- Top three fixes first: ...

### Dimension 1: Macro logic
| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | <quoted text> | CRITICAL or MAJOR or MINOR | <fix> |

### Dimension 2: Writing details
<same table shape>

### Dimension 3: English grammar
<same table shape, citing grammar-rule>

### Dimension 4: LaTeX format
<same table shape>

### Dimension 5: Figure quality
<same table shape>

### Banned-vocabulary and em-dash scan
<list with line references>

### Integrity gate result
- Gate 1 through 7: <pass or fail>

### Final score (1-10)
<score>

### Submission recommendation
- <Ready to submit | Needs 1-2 days more work | Needs major revision before submission>
