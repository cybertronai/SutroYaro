# Task 4: Reproduce Germain's depth-1/hidden-64 ARD result

**Priority**: MEDIUM
**Status**: TODO
**Source**: Yaroslav verification doc, Meeting #8

## Context

Germain's supervisor/researcher harness found:

- Baseline ARD: ~48.1-48.9
- Depth-1 + Hidden-64 ARD: ~33.1-34.5
- Accuracy gate passed (>=0.90), time under 2s

But Yaroslav flagged: "the winning change coincides with a big drop in total_accesses (92k to 64k). That might be legitimately less work (fewer layers/params/state)... more research needed."

We need to check: is this a real energy improvement, or just doing less computation? If total work drops proportionally to the ARD drop, it's not a real locality win.

## Tasks

- [ ] Run our baseline with depth=1, hidden=64 using fast.py
- [ ] Measure ARD, total_accesses, accuracy, wall time
- [ ] Compare ARD per unit of useful work (normalize by total_accesses or by accuracy)
- [ ] Check if it solves n=20/k=3 reliably (Germain's was on a simpler config?)
- [ ] Write findings

## References

- Yaroslav verification: docs/google-docs/yaroslav-verification.md (Germain section)
- Our baseline: src/sparse_parity/fast.py (hidden=200, 2 layers)
- G B's prior result in CLAUDE.md: "Architecture experiments (depth-1/hidden-64, ARD ~33-35)"
