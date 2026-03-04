# Experiment exp_perlayer_batch: Per-layer + batching combined

**Date**: 2026-03-04
**Status**: SUCCESS
**Answers**: Open Question #4 — "Does per-layer + batching combine?"

## Hypothesis

If we combine per-layer forward-backward with mini-batch SGD (batch=32), we'll get the convergence reliability of batching with the ARD benefit of per-layer, without harming convergence speed.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 200 |
| lr | 0.1 |
| wd | 0.01 |
| batch_size | 1 (single) / 32 (batch) |
| max_epochs | 200 |
| n_train | 1000 |
| seeds | 42, 43, 44, 45, 46 |
| method | standard / perlayer x single / batch |

## Results

| Metric | Value |
|--------|-------|
| All variants solve? | Yes, 5/5 seeds each |
| Best convergence speed | single-sample (5.2 epochs avg) |
| Best wall time | standard+single (0.058s) |
| Per-layer + batch epoch count | 40.6 avg (vs 41.4 standard+batch) |
| Per-layer + batch wall time | 0.665s (3.7x slower than standard+batch due to re-forward) |

## Key Table

| Variant | Solved | Avg Epochs | Avg Wall Time |
|---------|--------|-----------|---------------|
| standard+single | 5/5 | 5.2 | 0.058s |
| standard+batch | 5/5 | 41.4 | 0.177s |
| perlayer+single | 5/5 | 5.2 | 0.221s |
| perlayer+batch | 5/5 | 40.6 | 0.665s |

## Analysis

### What worked
- Per-layer + batch converges reliably (5/5 seeds, 100% test accuracy)
- Per-layer + batch needs slightly fewer epochs than standard+batch (40.6 vs 41.4) — marginal, ~2% improvement
- Per-layer does not hurt convergence in either single or batch mode
- Single-sample variants are dramatically faster in epochs (5.2 vs ~41)

### What didn't work
- Per-layer + batch is 3.7x slower in wall time than standard+batch (0.665s vs 0.177s) because it requires a re-forward pass through layer 2 after updating W1
- Batching always needs ~8x more epochs than single-sample for this problem size
- The per-layer epoch-count advantage over standard is negligible with batching (2% vs identical for single-sample)

### Surprise
- Single-sample SGD is dramatically better than batch for this problem: 5.2 vs 41 epochs. The conventional wisdom that batching helps convergence is wrong here — single-sample SGD's frequent updates let it find the phase transition much faster. Batching's value (from exp1) was about training stability, not speed.

## Open Questions (for next experiment)

- Does per-layer + batch have better ARD than standard+batch? The re-forward pass adds compute but keeps W1 in cache during its update-then-reuse cycle. Need cache-simulation MemTracker to measure this properly.
- Can we reduce the epoch gap between single-sample and batch by using smaller batches (e.g., batch=4 or batch=8)?
- At larger scale (n=30+), does the per-layer batch advantage grow?

## Files

- Experiment: `src/sparse_parity/experiments/exp_perlayer_batch.py`
- Results: `results/exp_perlayer_batch/results.json`
