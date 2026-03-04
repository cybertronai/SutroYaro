# Beat 20-bit Sparse Parity: Research Plan

> **For Claude:** Execute this plan as an autonomous research cycle. Each experiment is independent. Run, measure, record findings, adapt. Use `python3 -m sparse_parity.run` infrastructure for results.

**Problem**: Our 20-bit (k=3, 17 noise) sparse parity gets 54% accuracy (coin flip). We need >90%.

**Date**: 2026-03-04

## Why We're Failing (Literature Diagnosis)

Per [Barak et al. 2022 "Hidden Progress in Deep Learning"](https://arxiv.org/abs/2207.08799):

1. **Not enough iterations**: SGD needs ~n^O(k) iterations for phase transition. For n=20, k=3 that's ~8,000+ steps. We run 50*200 = 10,000 but with wrong hyperparams.
2. **Learning rate too high**: We use LR=0.5. Literature uses LR=0.1 with batch size 32.
3. **Single-sample training**: Literature uses batch_size=32. Our single-sample cyclic is noisier.
4. **Grokking phenomenon**: Sparse parity exhibits grokking — appears stuck, then suddenly generalizes. We may be stopping too early. ([Merrill et al. 2023](https://arxiv.org/abs/2303.11873))
5. **No hidden progress tracking**: Loss/accuracy are blind to actual progress. Need to track ||w_t - w_0||_1.

## Experiment Plan (6 experiments, priority order)

### Experiment 1: Fix Hyperparameters (highest priority)
**Hypothesis**: Matching Barak et al.'s hyperparams will trigger the phase transition.

Changes to `config.py`:
- LR: 0.5 → 0.1
- Add batch_size: 32 (need to modify train.py for mini-batch SGD)
- max_epochs: 50 → 500 (= 100,000 steps with 200 samples)
- Track ||w_t - w_0||_1 as hidden progress measure

**Success criteria**: >90% test accuracy on 20-bit, k=3

### Experiment 2: Weight Decay Sweep
**Hypothesis**: Higher weight decay accelerates grokking on sparse parity.

Sweep: WD ∈ {0.001, 0.01, 0.1, 1.0, 2.0}
Keep best LR from Experiment 1.

**Success criteria**: Find WD that gives fastest phase transition

### Experiment 3: Sign SGD
**Hypothesis**: Sign SGD matches SQ lower bound per [Kou et al. 2024](https://arxiv.org/abs/2404.12376).

Changes to backward:
- Replace `W -= lr * grad` with `W -= lr * sign(grad)`
- This normalizes gradient magnitudes, helps with sparse features

**Success criteria**: Converges with fewer iterations than standard SGD

### Experiment 4: GrokFast (Low-Pass Gradient Filter)
**Hypothesis**: Amplifying slow gradient components accelerates grokking per [Lee et al. 2024](https://github.com/ironjr/grokfast).

Implementation:
- Maintain EMA of gradients: `g_slow = alpha * g_slow + (1-alpha) * grad`
- Amplify slow component: `grad_modified = grad + lambda * g_slow`
- alpha=0.98, lambda=2.0 (from GrokFast paper)

**Success criteria**: Phase transition happens 2-10x faster than Experiment 1

### Experiment 5: Cross-Entropy Loss
**Hypothesis**: Hinge loss saturates; cross-entropy provides continuous gradient signal.

Changes:
- Replace hinge loss with binary cross-entropy + sigmoid output
- This gives non-zero gradients even for correctly classified samples

**Success criteria**: Smoother training curves, potentially faster convergence

### Experiment 6: Hidden Progress Dashboard
**Hypothesis**: We need to see what SGD is actually doing during the "silent" phase.

Track and plot per epoch:
- ||w_t - w_0||_1 (weight movement norm)
- Fourier coefficients of learned function (correlation with each k-subset)
- Sparsity of W1 (fraction of near-zero rows)
- Gradient norm per layer

**Success criteria**: Can visually identify progress before phase transition

## Research Cycle Protocol

For each experiment:
1. Create `src/sparse_parity/experiments/exp_{N}_{name}.py`
2. Run experiment, save results to `results/run_{timestamp}/`
3. Write findings to `findings/exp_{N}_{name}.md`
4. Update `results/index.md`
5. Commit with descriptive message
6. If accuracy >90%, stop and document the winning config
7. If not, analyze why, update hypothesis, proceed to next experiment

## Key References

- [Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit](https://arxiv.org/abs/2207.08799) — Barak et al. 2022, NeurIPS
- [Matching the SQ Lower Bound for k-Sparse Parity with Sign SGD](https://arxiv.org/abs/2404.12376) — Kou et al. 2024, NeurIPS
- [A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks](https://arxiv.org/abs/2303.11873) — Merrill et al. 2023
- [GrokFast: Accelerated Grokking by Amplifying Slow Gradients](https://github.com/ironjr/grokfast) — Lee et al. 2024
- [Feature Learning Dynamics under Grokking in Sparse Parity](https://openreview.net/forum?id=gciHssAM8A) — 2024
- [parity-nn: Minimal codebase for sparse parity experiments](https://github.com/Tsili42/parity-nn)
