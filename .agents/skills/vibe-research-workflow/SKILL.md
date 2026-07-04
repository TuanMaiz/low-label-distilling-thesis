---
name: vibe-research-workflow
description: "Guides AI-assisted research with behavioral rules: Vibe Coding, Vibe Figure, Vibe Writing. Recommends tools (Cursor, Codex, Figma, Gemini) while keeping user in charge of academic judgment."
user-invocable: true
when_to_use: "When starting AI-assisted research work or choosing AI tools."
category: research
keywords: [ai, workflow, research, coding, writing, vibe]
argument-hint: "[task-description-or-phase]"
metadata:
  author: hkustdial
  version: "1.0.0"
  license: CC-BY-4.0
---

# Vibe Research Workflow

## Overview

Vibe Research is the modern research workflow where AI handles mechanical tasks (implementation, figure rendering, language polish) while the researcher retains full ownership of research direction, problem framing, experimental design, and factual accuracy. Goal: 2-5x productivity gain without compromising academic integrity.

Three sub-flows: **Vibe Coding**, **Vibe Figure**, **Vibe Writing** - each governed by behavioral rules.

## Input

Task or phase:
<user-input>$ARGUMENTS</user-input>

## Core Procedure

### Step 1: Phase classification

Decide: coding, figure, writing, or mixed.

### Step 2: Behavioral rules recap

State six rules at session start:
1. AI permitted for: literature search, code/debugging support, language polish
2. User owns: research ideas, problems, designs, experimental plans, conclusions, novelty
3. Every AI output verified against actual process/results
4. No fabricated citations; references from user's reading
5. No academic misconduct (fabricated data, plagiarism concealment)
6. Honor venue/school AI-disclosure requirements

### Step 3: Phase-specific procedure

**Vibe Coding**: Plan Mode, Small Steps, Clear Requirements
**Vibe Figure**: Four-step workflow (design, render, review, polish)
**Vibe Writing**: Red-line rules (no AI-generated conclusions, verify all citations)

### Step 4: Tool selection

Match tool to phase:
- **Coding**: Cursor (IDE-native) or Codex (agentic CLI)
- **Figure**: PowerPoint + Figma (static), Matplotlib + Seaborn (results), Gemini (sketches)
- **Writing**: Codex/ChatGPT (polish), Grammarly (grammar), Overleaf (LaTeX)

### Step 5: Integrity gate

Before ending session:
1. Behavioral rules stated at start
2. No fabricated citations introduced
3. User owns research direction/framing/contributions
4. All AI-generated code reviewed and tested
5. All AI-drafted paragraphs rewritten or verified
6. AI-disclosure requirements checked
7. User's expertise driving project, AI is accelerator

## Output Format

### 1. Phase
- Primary: <coding/figure/writing/mixed>
- Secondary: <list>

### 2. Behavioral rules
- Rules 1-6 acknowledged

### 3. Workflow plan
| Time block | Phase | Activity | Tool | User check |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

### 4. Tool recommendations
| Phase | Primary | Alternative | Reason |
|---|---|---|---|
| Coding | ... | ... | ... |
| Figure | ... | ... | ... |
| Writing | ... | ... | ... |

### 5. Red-line reminders
- (from phase-specific rules)

### 6. Integrity gate plan
- Verification points: ...
- AI-disclosure for target venue: ...
