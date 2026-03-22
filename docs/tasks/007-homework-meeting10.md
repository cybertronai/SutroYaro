# Task 007: Homework for Meeting #10

**Priority**: HIGH
**Status**: DONE
**Due**: Monday 2026-03-23 (Meeting #10)
**Source**: Meeting #9 homework assignment

## Assignment

From Meeting #9:

1. Get agents to improve sparse parity using **DMC** (Data Movement Complexity) as the energy proxy
2. Iterate on prompts and meta-approaches to go from "metric spec + problem spec" to experiments quickly

## Checklist

### DMC Baseline Sweep

- [x] Run DMC measurement for GF(2) Gaussian Elimination
- [x] Run DMC measurement for KM Influence Estimation
- [x] Run DMC measurement for SGD (standard config: n=20, k=3, hidden=200, lr=0.1)
- [x] Run DMC measurement for SMT Backtracking
- [x] Run DMC measurement for Fourier/Walsh-Hadamard
- [x] Compile comparison table: method / accuracy / DMC / ARD / wall time

### DMC Optimization

- [x] Pick the most promising method for DMC optimization
- [x] Run at least one experiment targeting DMC reduction
- [x] Record results in findings format (hypothesis, method, result, key number)
- [x] Update DISCOVERIES.md if findings answer an open question

### Presentation Prep

- [x] Prepare results summary (table + key insight)
- [x] Note any differences between DMC ranking and ARD ranking
- [x] Record video or prepare demo if presenting async

## Context

DMC is already tracked in MemTracker alongside ARD. Formula: `DMC = sum(sqrt(stack_distance))` for all float accesses. Current baseline from CLAUDE.md: ARD 4,104 / DMC 300,298.

Key question: does optimizing DMC lead to different algorithmic choices than optimizing ARD? If the rankings change, that's a finding.

## Files

- `src/sparse_parity/tracker.py` -- MemTracker with DMC
- `src/sparse_parity/cache_tracker.py` -- CacheTracker with LRU simulation
- `src/sparse_parity/harness.py` -- Locked evaluation harness (DO NOT MODIFY)
