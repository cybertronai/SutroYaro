# Experiment exp_km_sat_hybrid: KM-min + SAT Hybrid for Sparse Parity

**Date**: 2026-04-27
**Contributor**: SethTS (Seth Stafford)
**Status**: SUCCESS
**Answers**: First ByteDMD measurement of pure SAT backtracking on sparse parity, plus a KM-min + SAT hybrid that verifies the influence-probe result with constraint satisfaction.

## Hypothesis

KM-min's 268 ByteDMD score is low because the solve step is just a sequential label scan. Can a more general approach get close: influence probing to prune candidates, then SAT verification?

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 40 |
| k_sparse | 3, 5 |
| method | Pure SAT backtracking, KM-min + SAT hybrid |
| seed | 42 |
| oracle model | Pre-paired samples (not a sparse-parity-challenge submission) |

## Results

### n=20/k=3

| Method | ByteDMD | geo_LB | Correct |
|--------|--------:|--------:|--------|
| KM-min (ref) | 268 | 103 | yes |
| Pure SAT backtrack | 53,828 | 20,718 | yes |
| KM + SAT hybrid | 872 | 336 | yes |

### n=40/k=5

| Method | ByteDMD | geo_LB | Correct |
|--------|--------:|--------:|--------|
| KM-min | 598 | 230 | yes |
| Pure SAT backtrack | untraceable (129ms wall) | — | — |
| KM + SAT hybrid | 3,382 | 1,302 | yes |

## Key findings

Pure SAT backtracking at 53,828 lands between GF(2) (101,501) and KM-min (268) on the ByteDMD leaderboard. Better than expected given the O(n^k) search space, because early pruning (first inconsistent sample) cuts most branches before they are explored.

The hybrid's SAT phase adds ~600 ByteDMD over KM-min alone at n=20/k=3 — the overhead of checking 1 subset against 21 samples. At n=40/k=5 the overhead grows (3,382 vs 598) because the larger x_data sits further down the LRU stack during Phase 2.

## Notes

- Z3 not implemented under ByteDMD: native C++ library, invisible to the Python tracer. Backtracking fallback only.
- Oracle-query model. KM-min uses pre-paired samples, not a fixed random dataset.
- Converting to the `sparse-parity-challenge` interface would require a Walsh-Fourier estimator (already exists as `exp_fourier.py`, scores 5,156,954, uncompetitive).

## Files

- `src/sparse_parity/experiments/exp_km_sat_hybrid.py` — experiment script
- `results/exp_km_sat_hybrid/results.json` — full results

## Provenance

Originally submitted as [PR #94](https://github.com/cybertronai/SutroYaro/pull/94), 2026-04-27. The script and results were extracted to main on 2026-05-11 at the contributor's request, preserving the headline measurement and the reproducibility path.
