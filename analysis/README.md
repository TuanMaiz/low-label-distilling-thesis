# Analysis Pipeline

Target home for Phase 06-07 result aggregation, active-vs-random selection
analysis, cost summaries, and thesis figures.

Planned modules:

- `plot_label_efficiency.py`: low-label F1/precision/recall curves.
- `analyze_failures.py`: qualitative error slices and teacher-noise inheritance.
- `analyze_selection.py`: planned active-vs-random selected-pair and student
  outcome comparison.
- `cost_summary.py`: teacher-generation and training-cost summaries.

Implemented diagnostics:

- `order_swap_probe.py`: scores WDC pairs in original and swapped record order
  to test whether the Qwen reranker judges identity symmetrically. Runbook:
  [order-swap-probe.md](order-swap-probe.md).
