# Task 13: Accuracy vs joules graph for sparse-parity solvers

**Priority**: MEDIUM
**Status**: OPEN
**Agent**: unassigned
**Source**: Telegram chat-yaroslav, 2026-05-08T01:02:55Z — *"One thing that would be cool to get is an explicit accuracy as a function of joules graph"*

## Context

We have per-solver `(accuracy, ARD, DMC, run-wallclock)` numbers in [docs/research/survey.md](../research/survey.md) and [DISCOVERIES.md](../../DISCOVERIES.md). What we don't have is the **single graph** Yaroslav is asking for: x-axis = joules consumed, y-axis = test accuracy, one line per solver family — so the accuracy/joule frontier is visible at a glance.

This is the **headline visualization** for the energy-efficient-training argument. It's also a slide we'll want for the next Sutro Group meeting and any external write-up.

**Relevance to SutroYaro**: SutroYaro tracks per-solver DMC + ARD. Joules = DMC + (per-level pJ from Bill Dally numbers). Both halves exist in the harness; this task is the plumbing that produces the frontier graph.

## Tasks

### Phase 1 — Pick the joule conversion function

DMC measures total `ceil(sqrt(stack_depth))` per access. To convert to joules we need a conversion that goes from "stack depth" → "memory level" → "pJ per access." Two options:

1. **Cache-tracker-aware**: use `src/sparse_parity/cache_tracker.py` (the LRU sim with explicit L1/L2/HBM pJ numbers). Output is honest per-access joules per level, summed.
2. **Closed-form approximation**: invert ByteDMD's `ceil(sqrt(d))` cost back into bytes-touched, multiply by Dally's average-per-byte pJ. Cheaper, less accurate.

Recommendation: start with (1), it's already in the codebase and produces real numbers.

- [ ] Document the joule conversion in `docs/research/joule-conversion.md` (one paragraph + the formula)
- [ ] Add a small `joules_from_cache_trace(trace) -> float` function to `src/sparse_parity/cache_tracker.py` if it's not already there

### Phase 2 — Generate the graph

- [ ] Add a script `bin/plot-accuracy-vs-joules` that:
  - Iterates over all solvers in `results/scoreboard.tsv` (or `DISCOVERIES.md` table)
  - For each, runs to convergence at multiple sample sizes (100, 1k, 10k examples)
  - Records `(joules, test_accuracy)` per run
  - Plots all solvers on a single matplotlib axis, log-x (joules), linear-y (accuracy 0-1)
  - Saves to `docs/research/figures/accuracy-vs-joules.png`
- [ ] Include a Pareto frontier line (lowest-joules solver at each accuracy threshold)

### Phase 3 — Write-up

- [ ] One-paragraph caption explaining what the reader should see (e.g. "KM-min sits 2-3 orders of magnitude left of SGD at the same accuracy threshold")
- [ ] Cross-link from `DISCOVERIES.md`, `docs/index.md`, `docs/research/survey.md`
- [ ] Surface the figure on the homepage if it's good enough for a hero

## Acceptance

- `docs/research/figures/accuracy-vs-joules.png` exists and is readable
- A script that regenerates it deterministically from the current scoreboard
- The figure is referenced from at least 3 of: DISCOVERIES.md, docs/index.md, docs/research/survey.md, the catchups index

## Dependencies

- Stable joule conversion (Phase 1); should be aligned with whichever cost model is current (1D Dally now, possibly 2D-grid after Task 012 lands)
- If the cost model changes mid-task, the graph regenerates from the same scoreboard — no rework on the data side
