---
name: idea-evaluator
description: "Evaluates research ideas against five-dimension framework (Higher, Faster, Stronger, Cheaper, Broader) with lifecycle matching, paradigm-shift probing, and fatal-flaws audit. Returns reviewer-style verdict."
user-invocable: true
when_to_use: "When evaluating a research idea before committing to implementation or writing."
category: research
keywords: [research, evaluation, novelty, feasibility, paper, phd]
argument-hint: "[research-idea-description]"
metadata:
  author: hkustdial
  version: "1.0.0"
  license: CC-BY-4.0
---

# Idea Evaluator

## Overview

This skill evaluates a preliminary research idea from the combined perspective of a top-venue reviewer and an experienced advisor. It scores the idea against five improvement dimensions (Higher, Faster, Stronger, Cheaper, Broader), matches the idea's lifecycle against the user's actual capability and available hours per week, probes whether the idea has paradigm-shift potential, flags fatal flaws, and returns one of three verdicts: Strong Accept, Accept with Revisions, or Reject and Pivot.

## Input

Research idea description:
<user-input>$ARGUMENTS</user-input>

## Core Procedure

### Step 1: First impression and paper-type positioning

Read the user's idea description. State whether the idea reads as Novel Problem, Novel Method, or New Setting. Is the story compelling in one sentence?

### Step 2: Fatal-flaws audit (early gate)

Run the fatal-flaws audit **before** the scoring steps. Identify at most two fatal flaws. For each, state the flaw, cite the detection rule, and recommend a concrete defense.

**Short-circuit rule:** If any fatal flaw is CRITICAL (single-handedly causes rejection, unfixable within the lifecycle), stop and emit:
- Verdict: Reject and Pivot
- Output sections 1 (First impression), 2 (Fatal flaws), and 8 (Verdict) only
- Do NOT run five-dimension scoring, paradigm-shift probe, or feasibility check

If no CRITICAL flaw is found, continue to Step 3.

### Step 3: Lifecycle and capability matching

Map the idea onto one of six categories (Application, Foundational Theory, Cross-Disciplinary, Frontier Exploration, Data-Intensive, Innovative Technique). Match against the user's declared capability (effective hours per week, skill depth, theoretical versus applied strength). Output a mismatch flag if lifecycle is shorter than the user's realistic execution window.

### Step 4: Five-dimension scoring

Score the idea on each of:
- **Higher**: effectiveness and accuracy gains
- **Faster**: efficiency and cost reduction
- **Stronger**: robustness, noise tolerance, generalisation
- **Cheaper**: data, annotation, or solution cost reduction
- **Broader**: cross-domain transplantation or unification

Score each 1-10 with explicit evidence. Identify the two or three dimensions where the idea has the highest ceiling.

### Step 5: Paradigm-shift probe

Test the idea against four questions:
1. Does it challenge a hidden assumption the field takes for granted?
2. Does it address an elephant-in-the-room problem everyone sees but nobody wants to touch?
3. Does it ride a technology-cycle shift?
4. If this problem solved itself, would the field change meaningfully? (Hamming's Rule)

Two or more yes answers means disruptive potential.

### Step 6: Feasibility check

Assess against the user's stated resources:
- Compute risk: does the experiment fit on stated hardware?
- Data risk: is the required data accessible?
- Engineering risk: does the implementation match the user's skill stack?
- Timeline risk: does the estimated duration fit within the idea's lifecycle?

### Step 7: Integrity gate

Before emitting the verdict:
1. Every dimension score cites specific evidence from the user's stated contribution
2. Feasibility claims reference the user's stated resources
3. Novelty claims either cite specific prior work or are labelled "unverified; literature check required"
4. Fatal flaws are specific and actionable
5. Verdict is consistent with scoring: Strong Accept requires at least two dimensions at 8+ and zero CRITICAL flaws
6. Paradigm-shift claim cites which probing question was answered positively
7. Lifecycle prediction is reasoned from the field's recent pace

### Step 8: Final verdict

Issue one of three verdicts:
- **Strong Accept**: execute now. Two or more dimensions at 8+, no fatal flaws, capability match green, lifecycle fit.
- **Accept with Revisions**: pivot the scope per recommendations before starting.
- **Reject and Pivot**: do not pursue this version.

## Output Format

### 1. First impression
- Paper type: <Novel Problem or Novel Method or New Setting>
- One-sentence story: <...>

### 2. Fatal-flaws audit
| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | ... | CRITICAL or MAJOR | ... |

### 3. Lifecycle and capability match
| Aspect | User's input | Assessment |
|---|---|---|
| Idea category | ... | ... |
| Lifecycle | ... months | ... |
| Weekly effective hours | ... | ... |
| Fit | ... | Green or Yellow or Red |

### 4. Five-dimension radar
| Dimension | Score 1-10 | Evidence | Lift suggestion |
|---|---|---|---|
| Higher | ... | ... | ... |
| Faster | ... | ... | ... |
| Stronger | ... | ... | ... |
| Cheaper | ... | ... | ... |
| Broader | ... | ... | ... |

### 5. Paradigm-shift probe
| Probe | Yes or No | Rationale |
|---|---|---|
| First Principles | ... | ... |
| Elephant in the Room | ... | ... |
| Technology Cycle | ... | ... |
| Hamming's Rule | ... | ... |

Disruptive potential: <none, possible, strong>

### 6. Feasibility
| Risk | Level | Mitigation |
|---|---|---|
| Compute | ... | ... |
| Data | ... | ... |
| Engineering | ... | ... |
| Timeline | ... | ... |

### 7. Integrity gate result
- Gate 1 through 7: <pass or fail>

### 8. Verdict
**<Strong Accept or Accept with Revisions or Reject and Pivot>**

Top three actions to take first:
1. ...
2. ...
3. ...
