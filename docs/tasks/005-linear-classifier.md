# Task 5: Linear classifier for parity (arXiv:2309.06979)

**Priority**: MEDIUM
**Status**: TODO
**Source**: Telegram message #710 (reading group)

## Context

From the Telegram channel: "In my Thursday reading group we are covering https://arxiv.org/abs/2309.06979, which shows that a linear classifier can solve the parity task, so that's an alternative approach that's halfway between current world and blank slate."

This is interesting because:
- All 4 local learning rules (Hebbian, PC, EP, TP) failed at chance level
- A linear classifier that solves parity would mean there's a feature representation where parity IS linear
- Could have very good ARD (simple model, small working set)

## Tasks

- [ ] Read arXiv:2309.06979, understand the method
- [ ] Implement as experiment following LAB.md template
- [ ] Test on n=20/k=3 with our standard config
- [ ] Measure accuracy, wall time, ARD
- [ ] Compare against GF(2), SGD baseline, and Fourier
- [ ] Add to proposed-approaches.md if not already there

## References

- Paper: https://arxiv.org/abs/2309.06979
- Telegram message #710
- Our local learning failures: DISCOVERIES.md "Local Learning Rules" section
- Experiment template: src/sparse_parity/experiments/_template.py
