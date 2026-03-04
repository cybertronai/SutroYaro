# Experiment Curriculum: Curriculum Learning for Scaling Sparse Parity

**Date**: 2026-03-04
**Status**: SUCCESS
**Answers**: Open question #3 — "Can curriculum learning help at scale?"

## Hypothesis

If we train on small n first (where grokking is fast) and then expand W1 to larger n, the network will transfer its learned feature detector and solve the larger problem with far fewer total epochs than direct training.

## Config

| Parameter | Value |
|-----------|-------|
| hidden | 200 |
| lr | 0.1 |
| wd | 0.01 |
| batch_size | 32 |
| n_train | 1000 |
| n_test | 200 |
| seed | 42 |
| method | numpy mini-batch SGD + curriculum |

## Results

### Test 1: n-curriculum [10 -> 20], k=3

| Method | Best Acc | Total Epochs | Wall Time |
|--------|----------|-------------|-----------|
| Direct n=20 | 98.0% | 33 | 0.11s |
| Curriculum 10->20 | 100.0% | 19 | 0.07s |
| **Speedup** | — | **1.7x fewer epochs** | **1.6x faster** |

### Test 2: n-curriculum [10 -> 30 -> 50], k=3

| Method | Best Acc | Total Epochs | Wall Time |
|--------|----------|-------------|-----------|
| Direct n=50 | 95.5% | 292 | 1.03s |
| Curriculum 10->30->50 | 98.5% | 20 | 0.06s |
| **Speedup** | — | **14.6x fewer epochs** | **17.2x faster** |

### Test 3: k-curriculum n=20, [k=2 -> k=3 -> k=5]

| Method | Best Acc | Total Epochs | Wall Time |
|--------|----------|-------------|-----------|
| Direct n=20/k=5 | 96.5% | 232 | 0.64s |
| Curriculum k=2->3->5 | 95.0% | 157 | 0.46s |
| **Speedup** | — | **1.5x fewer epochs** | **1.4x faster** |

## Key Table

| Method | Target | Acc | Epochs | Time | vs Direct |
|--------|--------|-----|--------|------|-----------|
| Direct | n=20/k=3 | 98.0% | 33 | 0.11s | baseline |
| n-curr 10->20 | n=20/k=3 | 100.0% | 19 | 0.07s | 1.7x |
| Direct | n=50/k=3 | 95.5% | 292 | 1.03s | baseline |
| n-curr 10->30->50 | n=50/k=3 | 98.5% | 20 | 0.06s | **14.6x** |
| Direct | n=20/k=5 | 96.5% | 232 | 0.64s | baseline |
| k-curr 2->3->5 | n=20/k=5 | 95.0% | 157 | 0.46s | 1.5x |

## Analysis

### What worked

- **n-curriculum is dramatically effective**: 10->30->50 solves n=50/k=3 in 20 total epochs vs 292 for direct training — a 14.6x improvement.
- **Transfer is near-instant**: After expanding W1 from n=10 to n=20, the network achieves 100% test accuracy in epoch 1. The trained feature detector on the 3 secret bits transfers perfectly because the new columns are initialized small and don't interfere.
- **n=50 solved via curriculum**: Direct training on n=50/k=3 previously failed at 54% in exp_d (200 epochs). Here with 500 epochs it reaches 95.5%. But curriculum solves it in just 20 epochs total — it bypasses the grokking plateau entirely.
- **k-curriculum also helps**: 1.5x speedup for k=5, meaningful but less dramatic than n-curriculum.

### What didn't work

- **k-curriculum has limited transfer**: Going from k=2 to k=5 doesn't transfer as cleanly because the parity function changes structure (from 2-way to 5-way XOR). The network must still learn the new interaction pattern.
- **k-curriculum for k=5 still took 140 epochs** in the final phase — most of the work. The k=2 and k=3 warmup helped but didn't short-circuit the k=5 learning.

### Surprise

The n-curriculum expansion is almost free. When going from n=10 to n=20 to n=50, each expansion phase solves in 1 epoch. The network has already learned "look at bits 1, 5, 8 and compute their product" — adding irrelevant input columns (initialized with near-zero weights) doesn't break this at all. The entire cost is the initial n=10 training (18 epochs), which is trivially fast.

This means **n-curriculum completely neutralizes the n^k scaling wall for k=3**. The hard part (finding the secret bits) is done at small n where it's cheap. Expansion is free.

## Open Questions (for next experiment)

- Can n-curriculum solve n=100 or n=200 with k=3? The transfer should still work since we're just adding more noise columns.
- Does n-curriculum help when k is large (k=5, k=7)? The initial n=10/k=5 training itself might be slow.
- Can we combine n-curriculum + k-curriculum? Start with n=10/k=2, scale up both.
- What is the ARD profile of curriculum training? Fewer epochs means fewer total memory accesses, but each expansion phase re-reads W1 at the new size.

## Files

- Experiment: `src/sparse_parity/experiments/exp_curriculum.py`
- Results: `results/exp_curriculum/results.json`
