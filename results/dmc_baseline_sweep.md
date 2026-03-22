# DMC Baseline Sweep

**Date:** 2026-03-22
**Issue:** #17
**Config:** n=20, k=3, seed=42, hidden=200, lr=0.1, wd=0.01, batch_size=32, n_train=1000, max_epochs=200

## Sparse Parity (all 5 methods)

| Method  | Accuracy | ARD          | DMC              | Total Floats | Time (s) |
|---------|----------|--------------|------------------|--------------|----------|
| GF2     | 1.0      | 420          | 8,607            | 860          | 0.048    |
| KM      | 1.0      | 92           | 20,633           | 4,420        | 0.049    |
| SMT     | 1.0      | 3,360        | 348,336          | 6,720        | 0.048    |
| SGD     | 1.0      | 8,504        | 1,278,460        | 24,470       | 0.321    |
| Fourier | 1.0      | 11,980,500   | 78,140,662,852   | 23,961,000   | 0.066    |

## Sparse Sum

| Method  | Accuracy | ARD       | DMC           | Total Floats | Time (s) |
|---------|----------|-----------|---------------|--------------|----------|
| SGD     | 1.0      | 20        | 2,862         | 1,300        | 0.001    |
| KM      | 1.0      | 92        | 20,633        | 4,420        | 0.001    |
| OLS     | 1.0      | 20,980    | 3,043,279     | 42,040       | 0.001    |
| Fourier | 1.0      | 220,500   | 187,661,233   | 441,000      | 0.001    |
| GF2     | N/A      | --        | --            | --           | --       |

GF2 is not applicable to sparse-sum (it exploits GF(2) linearity specific to XOR/parity).

## Sparse AND

| Method  | Accuracy | ARD          | DMC              | Total Floats | Time (s) |
|---------|----------|--------------|------------------|--------------|----------|
| KM      | **0.81** | 92           | 20,633           | 4,420        | 0.001    |
| SGD     | 1.0      | 29,164       | 52,885,890       | 1,105,329    | 0.011    |
| Fourier | 1.0      | 11,980,500   | 78,140,662,852   | 23,961,000   | 0.018    |
| GF2     | N/A      | --           | --               | --           | --       |

GF2 is not applicable to sparse-AND. SMT is only implemented for sparse-parity.

## Key Findings

### 1. DMC rankings differ from ARD rankings on sparse-parity

ARD ranking (best first): KM (92) > GF2 (420) > SMT (3,360) > SGD (8,504) > Fourier (11.9M)

DMC ranking (best first): GF2 (8,607) > KM (20,633) > SMT (348,336) > SGD (1.28M) > Fourier (78.1B)

**GF2 wins on DMC despite KM winning on ARD.** GF2 has higher per-access reuse distance (420 vs 92), but accesses far fewer total floats (860 vs 4,420). DMC accounts for total data volume, not just average distance, which is why the rankings flip. This is a meaningful distinction: GF2 moves less data overall even though each individual access is farther from its last write.

### 2. Fourier is catastrophically expensive on DMC

Fourier's DMC is 78 billion -- roughly 9 million times worse than GF2. It reads the entire dataset (x and y, ~21K floats) for each of the C(20,3) = 1,140 subsets, creating enormous stack distances. This makes Fourier the most energy-wasteful method by far, despite being fast in wall-clock time (0.066s).

### 3. SGD on sparse-sum converges in 1 epoch

Linear SGD solves sparse-sum in a single epoch (DMC = 2,862), beating even GF2 on sparse-parity. Sum has first-order structure that gradient descent exploits immediately. This confirms that sparse-sum is a fundamentally easier problem than sparse-parity.

### 4. KM fails on sparse-AND with default influence_samples=5

KM achieved only 81% accuracy on sparse-AND, finding the wrong secret [8, 17, 19] instead of [0, 15, 17]. For AND, flipping a secret bit only changes the output when all other k-1 secret bits are +1, which happens with probability 1/2^(k-1) = 25%. With only 5 samples, the signal is too noisy.

### 5. Fourier DMC is identical across sparse-parity and sparse-AND

Both produce DMC = 78,140,662,852. This is because Fourier reads the same data volume (x and y buffers of the same size) for the same C(20,3) subsets regardless of the target function. The DMC depends on access pattern and data size, not on what the function computes.

## Cross-Challenge Summary (DMC, best method per challenge)

| Challenge     | Best Method | DMC     | Runner-up | DMC (runner-up) |
|---------------|-------------|---------|-----------|-----------------|
| sparse-sum    | SGD         | 2,862   | KM        | 20,633          |
| sparse-parity | GF2         | 8,607   | KM        | 20,633          |
| sparse-and    | KM          | 20,633  | SGD       | 52,885,890      |
