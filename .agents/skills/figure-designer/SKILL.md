---
name: figure-designer
description: "Advises on designing figures for technical papers: Motivated Example (Figure 1), Solution Overview, and Experimental Results."
user-invocable: true
when_to_use: "When designing figures for a paper or auditing existing figures."
category: research
keywords: [figure, design, paper, visualization, chart]
argument-hint: "[figure-description-or-image-path]"
metadata:
  author: hkustdial
  version: "1.0.0"
  license: CC-BY-4.0
---

# Figure Designer

## Overview

A top-venue paper carries six to eight figures, with three carrying storytelling weight: Motivated Example (Figure 1), Solution Overview (Methodology), and Experimental Results. Reviewers scan these in under a minute.

This skill takes the user's intent plus context and returns: paradigm recommendation, layout sketch, labelling guidance, tool suggestion, and quality-control audit.

## Input

Figure design request or image path:
<user-input>$ARGUMENTS</user-input>

## Core Procedure

### Step 1: Figure-type identification

Decide: motivated-example, solution-overview, or experimental-results.

If \`figure-audit\` mode with image path, load image with Read tool before Step 2.

### Step 2: Paradigm recommendation

Each figure type has 2-3 canonical paradigms. Pick one and explain why others fit less well.

### Step 3: Layout sketch

Text description of layout: panel positions, element placement, arrows, colour assignments.

### Step 4: Labelling and annotation guidance

- Name every visible element concretely (no "Module A")
- Annotate critical points (failure, success, comparison)
- Specify font sizes and colour palette (ColorBrewer or Viridis)

### Step 5: Tool suggestion

Default recommendations:
- Motivated Example / Solution Overview: PowerPoint (draft), Figma (polish)
- Experimental Results: Matplotlib/Seaborn
- LaTeX-integrated: TikZ or PGFPlots

### Step 6: Universal rule audit

Verify against:
- Vector format (PDF, EPS, SVG)
- Font size ≥ 8pt post-scaling
- Colour-blind-safe palette; no colour-only encoding
- Self-contained caption (first sentence = core finding)
- Honest axis ranges
- No 3D effects, no chartjunk

### Step 7: Integrity gate

Before returning:
1. Paradigm matches figure type
2. Layout sketch is concrete enough to draw from
3. Labels are real entity names
4. Tool suggestion matches complexity
5. Universal rule audit run; no CRITICAL unaddressed
6. User verifies: running example matches Introduction
7. Chart type matches data type

## Output Format

### 1. Figure type
- Type: <motivated-example or solution-overview or experimental-results>
- Reason: <one sentence>

### 2. Paradigm recommendation
- Paradigm: <name>
- Why: <rationale>
- Alternatives rejected: <list>

### 3. Layout sketch
- Canvas: <size>
- Panels: <list>
- Arrows: <list>
- Colour assignment: <mapping>

### 4. Labelling and annotations
- Element names: <list>
- Highlights: <list>
- Font sizes: <target>
- Colour palette: <name>

### 5. Tool suggestion
- Primary: <tool>
- Alternative: <tool>

### 6. Universal rule audit
- [ ] Vector: <pass/fail>
- [ ] Font size: <pass/fail>
- [ ] Colour-blind safe: <pass/fail>
- [ ] Self-contained caption: <pass/fail>
- [ ] Honest axis: <pass/fail>
- [ ] No chartjunk: <pass/fail>

### 7. Severity summary
- <n> CRITICAL, <m> MAJOR, <k> MINOR
