---
title: "DBLP-ACM Qwen Phase 4 Implementation"
date: 2026-09-03
status: implementation-complete-awaiting-review
---

# DBLP-ACM Qwen Phase 4 Implementation

## Context

DBLP-ACM Phase 3 established a dataset-aware offline labeling path. Phase 4
needed to prove that the dataset can enter the Qwen gold-versus-LLM-hard
experiment lifecycle without touching the hash-bound WDC implementation or
requiring the current CPU machine to load a model.

## What Happened

- Added a DBLP Qwen student config whose only WDC difference is the approved
  scholarly-publication instruction.
- Added a strict execution profile for dataset/version identity, independent
  train/validation counts, derived training schedule, portable paths, and the
  gold-first lifecycle.
- Added CPU-only preflight and fixture result/package verification. Canonical
  pairs are re-derived from `dblp:` and `acm:` record IDs instead of trusting
  split-qualified presentation IDs.
- Added a shell dispatcher that lists safe actions and renders both future GPU
  command paths while keeping setup/training/evaluation/package execution locked.
- Kept the official test split, OpenRouter, Torch, Transformers, CUDA, model
  weights, training, and validation prediction outside this phase.

## Reflection

The strongest review findings concerned apparently valid but forgeable local
evidence. The implementation was hardened to reject symlink aliases, bind target
manifests to training bytes, bind predictions to ordered validation IDs, bind
training summaries to checkpoint manifests, and re-open the actual gold archive
before permitting the `llm_hard` fixture package. It also corrected the
important distinction between normalized split name `validation` and upstream
DBLP source name `valid`.

## Decisions

- WDC files remain byte-for-byte unchanged; DBLP uses adjacent files.
- Phase 4 proves orchestration with fixtures only. Real DBLP labels and targets
  are prerequisites for official preflight, not outputs of this phase.
- Repository-relative identities remain stable across checkout relocation;
  resolved absolute paths are runtime-only data.
- The frozen schedule is 464 optimizer steps per epoch, 4,640 planned steps, and
  464 warmup steps for 7,417 training pairs.

## Next

Researcher review decides whether Phase 4 is accepted. Only then should Phase 5
run the compatibility/handoff verification. Paid DBLP labeling and GPU execution
remain separately gated after that offline work.
