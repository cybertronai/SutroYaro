# Experiment Results Index

**4 experiments** | newest first

## Research Experiments (20-bit sparse parity)

| Experiment | Date | Method | 20-bit Accuracy | Key Finding | Details |
|------------|------|--------|-----------------|-------------|---------|
| Exp 4: GrokFast | 2026-03-03 | EMA gradient filter (α=0.98, λ=2.0) | 99% (12 epochs) | Counterproductive — baseline SGD hits **100% in 5 epochs** | [findings](../findings/exp4_grokfast.md) |
| Exp 1: Fix Hyperparams | 2026-03-03 | Mini-batch SGD (LR=0.1, batch=32) | **99%** (52 epochs) | Grokking phase transition at epoch 40-52 | [findings](../findings/exp1_fix_hyperparams.md) |

## Baseline Pipeline Runs (3 training variants x 2 scales)

| Run | Date | 3-bit Acc | 3-bit ARD (best) | 20-bit Acc | 20-bit ARD (best) | Report |
|-----|------|-----------|-------------------|------------|--------------------|----|
| run_20260303_200353 | 2026-03-03 | 100% | 9,674 (perlayer) | 54% | 69,094 (perlayer) | [report](run_20260303_200353/20260303_200353_report.md) |

## Summary

**Best 20-bit result**: 100% accuracy in 5 epochs / 22.7s (standard SGD, LR=0.1, n_train=500)

**Best 3-bit ARD**: 9,674 floats (per-layer forward-backward, 9.1% improvement over standard backprop)

**Key insight**: The 20-bit bottleneck was hyperparameters (LR=0.5 should be 0.1, n_train=200 should be 500), not the optimizer. With correct settings, no grokking delay occurs.

---

Each run/experiment directory contains:
- `results.json` — full structured metrics
- Findings in `findings/` — human-readable analysis
- Plots (where generated) — loss, accuracy, and ARD charts
