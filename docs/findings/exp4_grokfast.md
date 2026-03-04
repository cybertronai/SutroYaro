# Experiment 4: GrokFast (Low-Pass Gradient Filter)

**Date**: 2026-03-03
**Status**: COMPLETED — Both methods achieved >90% test accuracy (BREAKTHROUGH)

## Hypothesis

Amplifying slow gradient components via GrokFast (Lee et al. 2024) will accelerate the grokking phase transition on 20-bit sparse parity.

GrokFast maintains an EMA of gradients and amplifies the slow component:
```
g_slow = alpha * g_slow + (1-alpha) * grad
grad_modified = grad + lambda * g_slow
```
With alpha=0.98 (strong smoothing) and lambda=2.0 (2x amplification of slow component).

## Configuration

| Parameter | Value |
|-----------|-------|
| n_bits | 20 |
| k_sparse | 3 |
| hidden | 1000 |
| n_train | 500 |
| n_test | 200 |
| LR | 0.1 |
| WD | 0.01 |
| max_epochs | 200 |
| batch_size | 1 (single-sample cyclic) |
| GrokFast alpha | 0.98 |
| GrokFast lambda | 2.0 |
| seed | 42 |

Secret parity indices: [0, 3, 8]

## Results

| Method | Best Test Acc | Final Train/Test | Epochs to >90% | Time | Weight Movement |
|--------|-------------|------------------|-----------------|------|-----------------|
| GrokFast | **99.0%** | 100.0% / 98.0% | ~12 | 383.7s | 441,826 |
| Baseline SGD | **100.0%** | 99.8% / 100.0% | ~5 | 22.7s | 5,300 |

## Key Findings

### 1. BREAKTHROUGH: 20-bit sparse parity is solvable with corrected hyperparameters

Both methods achieved >90% test accuracy on 20-bit (k=3) sparse parity. The previous failure (54% accuracy, coin-flip) was entirely due to bad hyperparameters, not an inherent difficulty requiring GrokFast.

The critical fixes vs. the old `SCALE_CONFIG`:
- **LR**: 0.5 -> 0.1 (lower learning rate prevents overshooting)
- **n_train**: 200 -> 500 (more training samples)
- **WD**: 0.01 (kept same, but with lower LR it's more effective)
- **max_epochs**: 50 -> 200 (more time, though convergence happens by epoch 10)

### 2. GrokFast is COUNTERPRODUCTIVE in this regime

Contrary to the hypothesis, GrokFast significantly hurt performance:
- **Slower convergence**: Baseline reaches 100% test at epoch 10. GrokFast peaks at 99% around epoch 12-13 and never reaches 100%.
- **Massive weight movement**: GrokFast moves weights 83x more (441K vs 5.3K L1 norm). The gradient amplification causes overshooting.
- **Loss instability**: GrokFast loss reached 55,726 at epoch 10 before stabilizing. Baseline loss stayed below 6.0 throughout.
- **Training time**: 383.7s vs 22.7s (17x slower). The EMA computation adds overhead per step, and 200 epochs vs 10 needed epochs wastes compute.

### 3. Why GrokFast hurts here

GrokFast was designed for settings where grokking takes thousands of epochs (modular arithmetic, algorithmic tasks). In those settings:
- The model memorizes first, then slowly generalizes
- The generalizing gradient signal is weak and slow
- GrokFast amplifies this weak signal

But on 20-bit sparse parity with corrected hyperparams:
- Generalization happens almost immediately (epoch 4-10)
- There is no extended memorization phase
- The gradient signal is already strong enough
- Amplifying it causes instability and overshooting

### 4. Weight movement as diagnostic

The ||w_t - w_0||_1 metric confirmed hidden progress:
- Baseline: steady growth from 3,927 (epoch 1) to 5,300 (epoch 10) — smooth convergence
- GrokFast: explosive growth from 13,021 (epoch 1) to 441,826 (epoch 14) — 83x more movement indicates the gradient amplification is way too aggressive

## Conclusions

1. **The 20-bit problem is solved.** LR=0.1, WD=0.01, n_train=500 with hidden=1000 achieves 100% test accuracy in 10 epochs (~5,000 steps, 22.7s).

2. **GrokFast is not needed and is harmful** when the base hyperparameters are already good enough for fast convergence. It's designed for regimes with extended memorization phases (thousands of epochs), which don't apply here.

3. **The real fix was hyperparameter correction** (Experiment 1 territory): lower LR and more training data. The original SCALE_CONFIG with LR=0.5 and n_train=200 was the bottleneck.

4. **Recommendation**: If GrokFast is to be retried, it should be on a harder configuration (e.g., n_bits=40, k_sparse=5) where the phase transition is genuinely delayed. On the current 20-bit setup, standard SGD is the clear winner.

## Files

- Experiment script: `src/sparse_parity/experiments/exp4_grokfast.py`
- Results JSON: `results/exp4_grokfast_20260303_222209.json`

## References

- Lee et al. 2024, "GrokFast: Accelerated Grokking by Amplifying Slow Gradients" — https://github.com/ironjr/grokfast
- Barak et al. 2022, "Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit" — https://arxiv.org/abs/2207.08799
