# Experiment: GF(2) with Noisy Labels

**Hypothesis**: GF(2) Gaussian elimination assumes exact parity. With label noise, the linear system becomes inconsistent. We test at what noise rate it fails and whether robust methods can recover.

**Date**: 2026-03-09
**Status**: SUCCESS (robust solver) / FAILED (basic solver)

## Summary

| Method | Noise Tolerance | Notes |
|--------|-----------------|-------|
| Basic GF(2) | 0% (1% causes failure) | Inconsistent system, no solution |
| Robust (subset sampling) | Up to 10-15% | Finds clean subset of equations |
| Robust at 15% noise | 65% accuracy | Degraded but usable |
| Robust at 20% noise | 20% accuracy | Too much noise |

**Key finding**: GF(2) is extremely fragile to noise, but subset-sampling recovers the secret at moderate noise levels by finding clean equations.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| n_samples | 21-500 (varied) |
| noise_rate | 0-30% (varied) |
| seeds | 10-50 per config |
| max_subsets | 100 (robust solver) |

## Results

### Experiment 1: Basic GF(2) Noise Sweep

At n_samples=100, basic GF(2) with Gaussian elimination:

| Noise Rate | Correct | Inconsistent |
|------------|---------|--------------|
| 0% | 20/20 | 0 |
| 1% | 0/20 | 20 |
| 2%+ | 0/20 | 20 |

**Even 1 corrupted label out of 100 causes complete failure.** The linear system over GF(2) has no solution when any equation is wrong.

### Experiment 2: More Samples Don't Help Basic Solver

| n_samples | Noise 1% | Noise 2% | Noise 5% |
|-----------|----------|----------|----------|
| 21 | 9/10 | 9/10 | 4/10 INC |
| 30 | 10/10 | 10/10 | 0/10 INC |
| 50 | 10/10 | 0/10 INC | 0/10 INC |
| 100 | 0/10 INC | 0/10 INC | 0/10 INC |
| 500 | 0/10 INC | 0/10 INC | 0/10 INC |

Counter-intuitively, **more samples make things worse** for the basic solver. With 21 samples and 1% noise (0.21 expected flips), you might get lucky and have no flips. With 100 samples and 1% noise (1 expected flip), you're guaranteed inconsistency.

### Experiment 3: Robust Subset-Sampling Solver

The robust solver samples multiple subsets of n equations and votes on the most common solution:

| Noise | Basic GF(2) | Robust | Improvement |
|-------|-------------|--------|-------------|
| 0% | 20/20 | 20/20 | +0 |
| 2% | 0/20 | 20/20 | +20 |
| 5% | 0/20 | 20/20 | +20 |
| 10% | 0/20 | 20/20 | +20 |
| 15% | 0/20 | 13/20 | +13 |
| 20% | 0/20 | 4/20 | +4 |

**The robust solver works because**: With 100 samples and 10% noise, ~90 equations are clean. If we sample subsets of 20, we'll find at least one clean subset with high probability. The true solution appears in a majority of consistent subsets.

### Algorithm: Robust GF(2) Solver

```python
def gf2_solve_robust(x, y, n_bits, max_subsets=100):
    """
    For noisy data, sample multiple subsets of n equations and vote.
    """
    A = ((x + 1) / 2).astype(np.uint8)  # Convert to GF(2)
    b = ((y + 1) / 2).astype(np.uint8)

    solution_counts = {}

    for indices in random_subsets(n_samples, n_bits, max_subsets):
        solution, rank = gf2_gauss_elim(A[indices], b[indices])
        if solution is not None:  # consistent
            key = tuple(solution.tolist())
            solution_counts[key] = solution_counts.get(key, 0) + 1

    # Return most common solution
    best = max(solution_counts, key=solution_counts.get)
    return best
```

## Analysis

### Why is GF(2) so fragile?

GF(2) Gaussian elimination solves A·s = b over the binary field. Each equation is:
```
x[secret[0]] ⊕ x[secret[1]] ⊕ ... ⊕ x[secret[k-1]] = y
```

A single flipped label makes one equation wrong. In the augmented matrix [A|b], this creates a row where the A part is in the row space, but the b part doesn't match. The consistency check fails.

### Why does subset sampling work?

With m samples and noise rate p:
- Expected clean samples: m·(1-p)
- Probability a random n-subset is clean: (1-p)^n

For m=100, p=0.10, n=20:
- Clean samples: 90
- P(clean subset) ≈ 0.9^20 ≈ 12%

With 100 subset samples, we expect ~12 clean subsets, all producing the same (correct) solution. Wrong solutions appear in at most a few subsets. Majority voting filters them out.

### Comparison with MDL

DISCOVERIES.md notes that MDL is noise-robust at 5% noise. Comparison:

| Method | 5% noise | 10% noise | ARD | Time |
|--------|----------|-----------|-----|------|
| MDL | 100% | untested | 1,147,375 | 0.013s |
| GF(2) basic | 0% | 0% | ~500 | 0.0005s |
| GF(2) robust | 100% | 100% | ~500 | ~0.01s |

GF(2) robust is comparable to MDL at 5% noise, faster at 0.01s vs 0.013s, and has 2000x better ARD (~500 vs 1,147,375).

## Open Questions

1. **What's the theoretical noise threshold?** The robust solver works up to ~15% empirically. Is there a closed-form bound?

2. **Optimal subset count?** We used 100 subsets. Fewer might work, reducing compute.

3. **Comparison with error-correcting codes?** LDPC or BCH codes are designed for exactly this problem. Would they be faster?

4. **Higher k?** With k=5 or k=7, the secret is larger (more bits in the solution). Does noise tolerance scale?

## Files

- Experiment: `src/sparse_parity/experiments/exp_gf2_noise.py`
- Results: `results/exp_gf2_noise/results.json`
- This document: `findings/exp_gf2_noise.md`
