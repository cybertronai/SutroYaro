# Task 12: 2D-grid Dally model extension to sparse-parity

**Priority**: HIGH
**Status**: OPEN
**Agent**: unassigned
**Source**: Telegram chat-yaroslav, 2026-05-08T00:56:38Z — *"Looking into extending sparse-parity challenge to use the 2D grid model. Might need adding instructions to https://github.com/cybertronai/simplified-dally-model , trying to see how few I can get away [with]"*

## Context

The current sparse-parity benchmark + ByteDMD scoring assumes a **1D memory hierarchy** (registers → L1 → L2 → HBM with hardcoded per-level pJ costs from Bill Dally's numbers). Yaroslav is now extending to a **2D grid model** (`cybertronai/simplified-dally-model`) where memory locality is two-dimensional — closer to actual physical layout of arithmetic units on a die — and exploring how few primitive instructions are needed to capture the relevant cost dynamics.

This affects the sparse-parity challenge in two ways:
1. **The scoring metric changes** — submissions need to be re-scored against the 2D-grid model, not just the 1D Dally-numbers heuristic.
2. **The instruction surface for tracked code may shrink** — a smaller primitive set means fewer ways for an algorithm to "leak" untracked operations.

**Relevance to SutroYaro**: SutroYaro is the harness + scoring + leaderboard for sparse-parity. If the underlying physical model changes, the entire scoring pipeline (`src/bytedmd/`, `src/sparse_parity/eval/`, the eval-environment) needs to follow.

References:
- `cybertronai/simplified-dally-model` (the 2D grid + reduced-instruction-set repo Yaroslav is iterating on)
- ByteDMD spec/reference: https://github.com/cybertronai/ByteDMD
- 1D Dally numbers (current model): register 5 pJ, L1 (64KB) 20 pJ, L2 (256KB) 100 pJ, HBM 640 pJ per float access

## Tasks

### Phase 1 — Read + summarize the 2D-grid model

- [ ] Read `cybertronai/simplified-dally-model` README + any current iteration notes
- [ ] Document: what's the 2D-grid coordinate system? what costs does it compute? what's the minimum instruction set Yaroslav has converged on?
- [ ] Compare against the current 1D Dally cost model in `src/sparse_parity/cache_tracker.py`
- [ ] Write to `docs/research/2d-grid-model-summary.md`

### Phase 2 — Identify which sparse-parity solvers re-score most differently

The current top solvers ([RESULTS](../research/survey.md)) are KM-min, GF(2) Gaussian elimination, KM influence estimation, SMT backtracking, SGD baseline. Each has a distinct memory-access pattern:
- KM-min: ~20 reads / 3,578 DMC — small, cache-friendly
- GF(2): ~420 reads / ~203K DMC — k-independent but heavier
- SGD: 8,504 reads / 1,278,460 DMC — many activations, gradient-add re-fetches

Under 2D-grid scoring, locality matters more (or differently). Hypothesis: the gap between KM-min and SGD widens further; GF(2)'s position shifts depending on whether matrix-fused operations register as nearby in the 2D layout.

- [ ] Re-score top 5 solvers under the 2D-grid model
- [ ] Diff against current 1D ranking
- [ ] Write to `docs/findings/2d-grid-rescoring.md`

### Phase 3 — Submission-pipeline plumbing

- [ ] If the 2D-grid model is going to become the primary metric, draft the change to the submission pipeline at `cybertronai/sparse-parity-challenge` (issue, then PR)
- [ ] Backwards-compat: keep 1D ByteDMD as a fallback metric for now; add a new `--metric=2d-grid` flag to the eval harness
- [ ] Update `docs/research/eval-environment.md` to document both

## Acceptance

- A `docs/research/2d-grid-model-summary.md` that any new agent can read to understand what changed
- A `docs/findings/2d-grid-rescoring.md` showing top-N solvers under both metrics side-by-side
- A draft plan for the eval-harness change (PR or issue link)

## Dependencies

- `cybertronai/simplified-dally-model` must reach a stable instruction set first (Yaroslav is still iterating per Telegram)
- This task should not start Phase 3 until Yaroslav explicitly signals "2D grid is the new primary"
