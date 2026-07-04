---
name: tech-paper-template
description: "Structures a technical paper's logical skeleton using thinking-template table (research background, limitations, key idea, challenges, methodology, contributions)."
user-invocable: true
when_to_use: "When brainstorming a paper, discussing with advisor, or planning before drafting."
category: research
keywords: [paper, template, structure, planning, research]
argument-hint: "[paper-idea-or-description]"
metadata:
  author: hkustdial
  version: "1.0.0"
  license: CC-BY-4.0
---

# Tech Paper Template

## Overview

Before drafting prose, a technical paper needs a full logical skeleton: research background, limitations of prior work, key idea or research goal, technical challenges, methodology modules, and contributions. This skill fills in that skeleton via a standardised thinking-template table, positions the paper type, and runs four self-consistency checks.

## Input

Paper idea or description:
<user-input>$ARGUMENTS</user-input>

## Core Procedure

### Step 1: Paper-type positioning

Decide **Technique** versus **New Problem/Setting**:
- In **Technique**, the Key Idea carries the narrative and Our Goal is a short bridge
- In **New Problem/Setting**, Our Goal is the contribution and the Key Idea justifies feasibility

### Step 2: Fill the thinking template

Fill the seven cells:

1. **Research background**: Scenario, importance, motivation
2. **Limitations 1-3**: Specific limitations of prior work (2 is acceptable; more than 3 is not)
3. **Key idea or Our Goal**: One sentence
4. **Challenges 1-3**: Technical challenges preventing naive solution
5. **Methodology modules**: One module per challenge
6. **Contributions**: 3 or 4, each mapped to a section

If a cell is incomplete, mark it as a gap with severity.

### Step 3: Run four self-consistency checks

Run each check:
1. **Limitations to Key Idea**: Does the Key Idea/Goal address stated Limitations?
2. **Key Idea to Challenges**: Do Challenges arise naturally from implementing the Key Idea?
3. **Challenges to Methodology**: Does each methodology module address one challenge?
4. **Methodology to Contributions**: Do contributions cover each module or experimental result?

Every failure is CRITICAL.

### Step 4: Generate methodology outline

From challenges, derive methodology outline: topic sentence, per-module subsection names, per-module one-sentence summary. This becomes the skeleton for Section 3 or 4.

### Step 5: Integrity gate

Before emitting:
1. Paper-type positioning consistent with user's actual contribution
2. Limitations are specific and cite-able
3. Key Idea/Goal is single sentence a reviewer could quote
4. Challenges derive from implementing Key Idea; not invented
5. Methodology modules have one-to-one mapping with challenges
6. Contributions map to methodology modules and specific sections
7. All four self-consistency checks pass

## Output Format

### 1. Paper-type positioning
- Type: <Technique Paper or New Problem/Setting Paper>
- Rationale: <one sentence>

### 2. Thinking template

| Stage | Your content |
|---|---|
| Research background | ... |
| Limitation 1 | ... |
| Limitation 2 | ... |
| Limitation 3 (if applicable) | ... |
| Key Idea / Our Goal | ... |
| Challenge 1 | ... |
| Challenge 2 | ... |
| Challenge 3 (if applicable) | ... |
| Methodology topic sentence | ... |
| Module A (addresses Challenge 1) | ... |
| Module B (addresses Challenge 2) | ... |
| Module C (addresses Challenge 3) | ... |
| Contribution 1 | ... (Section <X>) |
| Contribution 2 | ... (Section <Y>) |
| Contribution 3 | ... (Section <Z>) |

### 3. Self-consistency checks
- Check 1 Limitations -> Key Idea: <pass or fail>
- Check 2 Key Idea -> Challenges: <pass or fail>
- Check 3 Challenges -> Methodology: <pass or fail>
- Check 4 Methodology -> Contributions: <pass or fail>

### 4. Severity summary
- <n> CRITICAL, <m> MAJOR, <k> MINOR
- Top three fixes first: ...

### 5. Next suggested skill
- If all checks pass: Use intro-drafter to produce Introduction paragraph outline
- If checks fail: Address flagged chain breaks first
