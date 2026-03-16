# Experiment exp_egd: Egalitarian Gradient Descent

**Date**: 2026-03-16
**Status**: PARTIAL
**Answers**: TODO.md "SGD Under 10ms" / EGD hypothesis

## Hypothesis

If we replace gradient singular values with 1 (EGD, arXiv:2510.04930), then the
grokking plateau shortens because all gradient directions evolve at equal speed,
and we can push toward sub-10ms wall time.

## Config

| Parameter | SGD Baseline | EGD |
|-----------|-------------|-----|
| n_bits | 20 | 20 |
| k_sparse | 3 | 3 |
| hidden | 200 | 200 |
| lr | 0.1 | 0.1 |
| wd | 0.01 | 0.01 |
| batch_size | 32 | 32 |
| max_epochs | 200 | 200 |
| n_train | 1000 | 1000 |
| seeds | 42-46 | 42-46 |

## Results

### Part 1: Grokking Elimination (CPU, numpy)

EGD at lr=0.1 is the clear winner. Lower lr values slow convergence.

| Method | Avg Acc | Ep to 90% | Ep to Solve | Time (CPU) | Solved |
|--------|---------|-----------|-------------|------------|--------|
| SGD (lr=0.1) | 100% | 33 | 40 | 0.112s | 5/5 |
| EGD (lr=0.1) | 100% | **14** | **21** | 0.297s | 5/5 |
| EGD (lr=0.05) | 99.8% | 27 | 38 | 0.552s | 5/5 |
| EGD (lr=0.01) | 96.8% | 129 | --- | 1.280s | 4/5 |
| EGD (lr=0.005) | 75.6% | 120 | --- | 1.529s | 1/5 |

EGD at lr=0.1 reaches 90% in 14 epochs vs SGD's 33 (2.4x faster in epochs).
Solves in 21 vs 40 (1.9x faster).

CPU wall time is 2.7x worse due to SVD overhead (0.12ms per SVD on 200x20 matrix,
~31 batches/epoch = 3.7ms/epoch overhead).

### Part 2: Sub-10ms Push (CPU, small configs)

Small hidden (50) fails for both EGD and SGD at n_train=200. At n_train=500 with
hidden=50, SGD solves 4/5 seeds (avg 144 epochs, 0.166s) while EGD solves 2/5 (avg
136 epochs, 0.393s). Neither approach breaks 10ms.

The capacity/data bottleneck at small sizes dominates -- optimizer choice matters
less than having enough hidden units and training data.

### Part 3: GPU Timing (Modal L4)

| Config | Method | Avg ms | Ep to 90% | Ep to Solve | Solved |
|--------|--------|--------|-----------|-------------|--------|
| Parity h=200 | SGD | 1,068 | 36 | 42 | 5/5 |
| Parity h=200 | EGD | 1,207 | **14** | **19** | 5/5 |
| Parity h=50 | SGD | 2,012 | 139 | 164 | 5/5 |
| Parity h=50 | EGD | 3,140 | **69** | 106 | 5/5 |

On GPU, EGD is 12% slower in wall time despite 2x fewer epochs (parity h=200).
SVD overhead per batch is not amortized by the epoch reduction.

For h=50, EGD is 56% slower -- smaller matrices make SVD overhead proportionally larger.

### Part 4: Sparse Sum (GPU)

| Config | Method | Avg ms | Ep to 90% | Ep to Solve | Solved |
|--------|--------|--------|-----------|-------------|--------|
| Sum h=200 | SGD | 3,853 | --- | --- | **0/5** |
| Sum h=200 | EGD | 1,484 | **10** | **24** | **5/5** |

SGD at lr=0.1 diverges on sum (MSE loss, targets in [-3, 3]). The gradient
magnitude from MSE scales with target range, making lr=0.1 too aggressive.

EGD handles this because SVD normalization removes gradient magnitude
information. All directions get unit-norm updates regardless of target scale.

Note: SGD would solve sum with a lower lr (e.g., 0.01) or target normalization.
The failure is a hyperparameter issue, not a fundamental limitation. But EGD's
robustness to learning rate / target scale is a real property.

## Key Table

| Metric | SGD | EGD | Ratio |
|--------|-----|-----|-------|
| Parity: epochs to 90% (GPU) | 36 | 14 | 2.6x fewer |
| Parity: epochs to solve (GPU) | 42 | 19 | 2.2x fewer |
| Parity: GPU wall time (h=200) | 1,068ms | 1,207ms | 0.88x (slower) |
| Sum: solved (GPU, lr=0.1) | 0/5 | 5/5 | EGD robust |

## Analysis

### What worked

- EGD shortens the grokking plateau by 2-2.5x on parity. Equalizing gradient
  singular values helps the network find the sparse feature directions faster.
- EGD is robust to gradient scale. On sum, where MSE gradients are large, EGD's
  normalization prevents divergence that kills SGD at the same lr.
- lr=0.1 works for both SGD and EGD on parity (same lr, no tuning needed).

### What didn't work

- Sub-10ms is not achievable. SVD overhead per batch (~0.12ms CPU, similar GPU)
  outweighs the epoch savings. Even with 2x fewer epochs, wall time is 12% worse.
- Small hidden (50) is capacity-limited. Both methods struggle. The optimizer
  cannot compensate for insufficient model capacity.
- Lower EGD learning rates (0.01, 0.005) are too slow -- normalizing gradients to
  unit norm means the lr directly controls step size, and small lr means slow
  convergence.

### Surprise

SGD fails on sparse sum at lr=0.1 (0/5 seeds) while EGD solves it (5/5).
EGD's gradient normalization makes it robust to loss/target scale, independent
of its convergence speed benefit.

## Open Questions (for next experiment)

- Does EGD + curriculum learning compound? Curriculum gave 14.6x on n=50,
  EGD gives 2x on convergence. If multiplicative, that's ~30x.
- Can a cheaper approximation to SVD (e.g., power iteration for top singular
  vectors only) reduce the overhead while keeping the convergence benefit?
- What happens with EGD on k=5? The grokking plateau is much longer there.
- SGD on sum with lr=0.01: does it then match EGD's convergence speed?

## Files

- CPU experiment: `src/sparse_parity/experiments/exp_egd.py`
- GPU experiment: `bin/gpu_egd.py`
- CPU results: `results/exp_egd/results.json`
- GPU results: `results/exp_egd/gpu_results.json`
