---
name: intro-drafter
description: "Drafts 6-paragraph Introduction outline for technical papers: background, limitations, goal, challenges, solution, contributions. Positions paper as Technique or New Problem/Setting."
user-invocable: true
when_to_use: "When drafting or outlining a technical paper Introduction."
category: research
keywords: [introduction, outline, paper, structure, writing]
argument-hint: "[paper-idea-description]"
metadata:
  author: hkustdial
  version: "1.0.0"
  license: CC-BY-4.0
---

# Introduction Drafter

## Overview

The Introduction is the compressed version of the entire paper. In 1.5-2 pages it must state the research object, why the problem matters, why existing work falls short, what the paper contributes, and how contributions map to section numbers. This skill produces a six-paragraph outline with explicit purpose and writing points for each paragraph.

## Input

Paper description for Introduction:
<user-input>$ARGUMENTS</user-input>

## Core Procedure

### Step 1: Paper-type positioning

Decide:
- **Technique Paper**: main contribution is new method/mechanism. Narrative axis is Key Idea. Goal is one sentence.
- **New Problem/Setting Paper**: main contribution is new problem formulation. Narrative axis is Goal. Key Idea supports feasibility.

### Step 2: Paragraph-by-paragraph outline

For each of six paragraphs, return:
- **Purpose**: one sentence
- **Writing points**: 3-5 actionable bullets
- **Gaps**: what inputs don't cover, tagged with severity (CRITICAL, MAJOR, MINOR)

Paragraphs:
1. Background and Motivation. Running example. Why problem matters.
2. Limitations of existing work (at most 3).
3. Problem essence and Our Goal. Hard constraints explicit.
4. Key challenges (at most 3).
5. Solution overview. One-to-one challenge-to-module mapping.
6. Contributions (3-4 bullets, each with section reference).

### Step 3: Running example design

Propose 2-3 candidate examples if not provided. Ensure chosen example threads through paper.

### Step 4: Contribution alignment check

For each contribution:
- Maps to challenge/module/experiment
- Specific, not vague
- Cites section number

### Step 5: Flowchart consistency check

Verify logical throughline:
- Paragraph 1's running example referenced in Paragraph 5
- Paragraph 2's limitations motivate Paragraph 4's challenges
- Paragraph 3's goal aligns with Paragraph 6's contribution 1
- Paragraph 4's challenges map one-to-one with Paragraph 5's modules
- Paragraph 5's modules appear in Paragraph 6's contributions

### Step 6: Integrity gate

Before emitting:
1. Running example in Paragraph 1 reappears in Paragraph 5/6
2. Limitations are at most 3, each specific
3. Challenges are at most 3, each explains why naive fails
4. Challenge-to-module mapping is one-to-one
5. Contributions are 3-4, each maps to section number
6. No vague contribution language
7. Paper-type positioning reflected in Paragraph 3's weight

## Output Format

### 0. Type positioning
- Type: <Technique Paper or New Problem/Setting Paper>
- Rationale: <one sentence>

### 1. Paragraph 1: Background and Motivation
- Purpose: <...>
- Running example: <...>
- Writing points: <list>
- Gaps: <list with severity>

### 2. Paragraph 2: Limitations
- Purpose: <...>
- Writing points: Limitation 1, 2, 3
- Gaps: <list>

### 3. Paragraph 3: Problem Essence and Our Goal
- Purpose: <...>
- Goal sentence: "<...>"
- Writing points: <list>
- Gaps: <list>

### 4. Paragraph 4: Key Challenges
- Purpose: <...>
- Writing points: Challenge 1, 2, 3
- Gaps: <list>

### 5. Paragraph 5: Solution Overview
- Purpose: <...>
- Challenge-to-module mapping
- Writing points: <list>
- Gaps: <list>

### 6. Paragraph 6: Contributions
1. ... (Section X)
2. ... (Section Y)
3. ... (Section Z)
4. ... (Section W if applicable)
- Gaps: <list>

### 7. Flowchart consistency
- Running-example loop: <pass/fail>
- Limitations-challenges link: <pass/fail>
- Goal-contribution1 link: <pass/fail>
- Challenge-module mapping: <pass/fail>
- Contribution-section mapping: <pass/fail>

### 8. Integrity gate result
- Gate 1-7: <pass/fail>
- Severity: <n> CRITICAL, <m> MAJOR, <k> MINOR
