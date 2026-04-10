# Experiment SethTS-005: NoProp for Sparse Parity

**Date**: 2026-04-10
**Status**: FAILED (negative result — NoProp does not improve over SGD+Curriculum)
**Answers**: Does a diffusion-inspired local learning rule (NoProp) outperform SGD on sparse parity, either in convergence or DMD?

## Hypothesis

If we train each layer independently as a denoiser (NoProp), then (a) the denoising objective will find k-th order interactions more efficiently than SGD's hinge loss, and (b) eliminating the backward pass across layers will reduce per-step data movement.

## Config

| Parameter | Value |
|-----------|-------|
| n_bits | 20, 50 |
| k_sparse | 3, 5 |
| hidden | 200 |
| lr (NoProp) | 0.05 (MSE gradients ~2x hinge, so half of SGD's 0.1) |
| lr (SGD) | 0.1 |
| wd | 0.01 |
| batch_size | 32 |
| max_epochs | 500–1000 |
| n_train | 2000 |
| seeds | [42, 43, 44, 45, 46] |
| method | NoProp (T=5 layers, noise=[0.9, 0.7, 0.5, 0.3, 0.1]) |

## NoProp Adaptation

- Target y ∈ {-1,+1} noised by flipping with probability t
- T=5 independent layers, noise schedule [0.9, 0.7, 0.5, 0.3, 0.1] (high → low)
- Each layer receives [x | noisy_y] (n+1 inputs), trains with local MSE loss against clean y
- No gradients flow between layers
- Inference: chain layers sequentially starting from y=0, each refining the prediction
- No learned inverse mappings (simpler than TargetProp)

## Results

| Regime | SGD | NoProp | SGD+Curriculum | NoProp+Curriculum |
|---|---|---|---|---|
| n=20/k=5 | 100%, 70 ep | 100%, 90 ep | 100%, 25 ep | 100%, 33 ep |
| n=50/k=3 | 100%, 58 ep | 100%, 42 ep | 100%, 10 ep | 100%, 8 ep |
| n=50/k=5 | 0% | 0% | 100%, 34 ep | 100%, 42 ep |

**Per-step ARD** (n=20, hidden=200): SGD 8,747 vs NoProp 9,145 (1.05x — NoProp slightly more expensive)

**Total DMD on n=50/k=5**: SGD+Curriculum ~297K (34 ep × 8,747). NoProp+Curriculum ~1.92M (42 ep × 5 layers × 9,145). NoProp is ~6.5x more expensive overall.

## Analysis

### What worked
- NoProp is a viable learning algorithm — it solves the problems SGD solves (unlike TargetProp, which failed even on n=20/k=3)
- NoProp+Curriculum solves n=50/k=5 (100% solve rate), as does SGD+Curriculum
- On n=50/k=3, NoProp direct (42 ep) beats SGD direct (58 ep) — the denoising objective may help with the n-scaling problem slightly

### What didn't work
- NoProp is consistently slower than SGD+Curriculum across all regimes when curriculum is added to both
- Per-step ARD is marginally higher than SGD (5 independent local backward passes vs 1 global backward pass)
- Total DMD is ~6.5x worse on the hardest regime because NoProp needs more epochs and does 5x more layer-passes per epoch
- NoProp does not break the k=5 wall on its own (same as SGD direct)

### Surprise
The initial result (NoProp+Curriculum vs SGD direct) looked exciting — NoProp+Curriculum solved n=50/k=5 while SGD failed. But adding SGD+Curriculum as the proper baseline revealed that curriculum alone is sufficient; NoProp adds nothing. The confound was comparing NoProp+Curriculum against vanilla SGD rather than SGD+Curriculum.

## Why NoProp Fails to Help

The bottleneck for sparse parity isn't the learning rule — it's the n-scaling problem (irrelevant noise dimensions dominating the gradient signal). Curriculum solves this by keeping n small during the critical learning phase, regardless of whether the learning rule is SGD, GrokFast, or NoProp.

NoProp's denoising objective gives each layer the true label directly (with noise), so each layer optimizes for the same global task. This doesn't help with feature selection any differently than hinge loss — both require identifying the k relevant bits among n, and the signal lives at the k-th order Fourier coefficient regardless of loss function.

## Context: Why TargetProp Also Failed

TargetProp (exp_target_prop, implemented by Yad) uses learned inverse mappings to propagate targets backward — also local learning. It couldn't solve n=20/k=3 in 200 epochs. NoProp is simpler and does solve parity, but no better than SGD+Curriculum. Both methods underperform because the bottleneck is global feature selection, not the form of the local loss.

## Open Questions

- Does NoProp's slightly better n-scaling (42 vs 58 epochs on n=50/k=3 direct) hold at larger n? Could be a weak signal worth exploring.
- Would a larger T (more denoising steps) or different noise schedules change anything for the k=5 regime?
- NoProp's parallel training (T layers train independently) could reduce wall-clock time on multi-core hardware — not a DMD advantage but potentially a practical one.

## Files

- Experiment: `src/sparse_parity/experiments/exp_noprop.py`
- Results: `results/exp_noprop/results.json`
