# Experiment: GrokFast + Curriculum Scaling

**Date**: 2026-03-23
**Status**: COMPLETED — maps the scaling frontier
**Researcher**: SethTS

## Question

How far can GrokFast + Curriculum scale? exp_grokfast_curriculum showed it solves
n=50/k=5 in 14 epochs / 77ms. Does this extend to n=100, n=200?

## Hypothesis

Since curriculum makes n-expansion nearly free (1 epoch per phase), scaling n
should be cheap. The bottleneck should be the initial small-n phase where k
determines difficulty.

## What was performed

GrokFast + Curriculum vs Curriculum alone across 6 configurations, 5 seeds each
(60 runs). Curriculum stages: [10, 30, 50] for n=50, adding 100 and 200 for
larger targets.

| Parameter | Value |
|-----------|-------|
| hidden | 200 |
| lr | 0.1 |
| wd | 0.01 |
| batch_size | 32 |
| n_train | 2000 |
| GrokFast | a=0.98, l=2.0 |
| seeds | 42-46 |

## What was produced

### k=3: scales effortlessly to n=200

| Config | Method | Solve% | Epochs | Time |
|--------|--------|--------|--------|------|
| n=50/k=3 | GF+Curr | 100% | 7 | 0.035s |
| n=100/k=3 | GF+Curr | 100% | 8 | 0.046s |
| n=200/k=3 | GF+Curr | 100% | 11 | 0.095s |
| n=200/k=3 | Curr | 100% | 13 | 0.087s |

Each expansion phase (n=30->50, n=50->100, n=100->200) takes 1 epoch except
n=200 which takes 3. The cost is almost entirely in the initial n=10 training.

### k=5: wall between n=50 and n=100

| Config | Method | Solve% | Epochs | Time |
|--------|--------|--------|--------|------|
| n=50/k=5 | GF+Curr | 100% | 15 | 0.099s |
| n=100/k=5 | GF+Curr | 60% | 219 | 1.726s |
| n=100/k=5 | Curr | 40% | 337 | 2.132s |
| n=200/k=5 | GF+Curr | 0% | 719 | 7.722s |
| n=200/k=5 | Curr | 0% | 837 | 6.504s |

Phase breakdown for n=100/k=5 (seed 42):
- n=10: solved in 9 epochs (0.054s)
- n=30: solved in 3 epochs (0.016s)
- n=50: solved in 4 epochs (0.027s)
- n=100: **stalls at 94% for 500 epochs** (3.8s)

The expansion from n=50 to n=100 is where it fails. The k=5 feature detector
learned at small n isn't robust enough to survive 50 new noise dimensions.

## Can it be reproduced?

Yes. Script: `src/sparse_parity/experiments/exp_grokfast_curriculum_scale.py`
Results: `results/exp_grokfast_curriculum_scale/results.json`

## Finding

**Curriculum + GrokFast scales n effortlessly for k=3 (n=200 in 95ms) but hits a
wall at n=100 for k=5.** The n-expansion phases are nearly free (1 epoch each),
confirming curriculum fully neutralizes input dimension for low-k problems. But
for k=5, the feature detector learned at small n breaks during large expansion
steps. The frontier is between n=50 and n=100 at k=5.

## Analysis

### Why k=3 scales and k=5 doesn't

For k=3, the learned detector (a 3-way interaction) is robust to adding noise
dimensions. The 3 secret weights are large and well-separated from the noise
weights after training at small n. Adding 50 or 100 noise columns with tiny
initial weights doesn't disrupt the existing solution.

For k=5, the detector is more fragile. A 5-way interaction requires 5 weights
to be coordinated precisely. When expanding from n=50 to n=100, the 50 new noise
columns create 50 new gradient directions that can interfere with the delicate
5-way coordination. The model needs to "re-grok" at the larger scale, and with
n=100 the grokking plateau is too long.

### GrokFast helps at the margin

On n=100/k=5, GrokFast improves solve rate from 40% to 60%. It doesn't fix the
fundamental problem but makes marginal cases converge. This is consistent with
exp_grokfast_v2: GrokFast helps when the k-th order signal is weak.

### Possible fixes for the k=5 wall

- More gradual expansion (e.g., [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
  might keep each step small enough that the detector survives
- Larger hidden layer might make the detector more robust to noise
- Freezing the secret-bit weights during expansion (so only noise weights train)
- Lower learning rate during expansion phases to avoid disrupting learned features

## Open questions

- Does more gradual expansion (steps of 10 instead of 50) fix the k=5 wall?
- Does hidden=500 or hidden=1000 make the k=5 detector more robust to expansion?
- Is there a theoretical limit on the expansion ratio (new_n / old_n) that a
  k-th order detector can survive?

## Files

- Experiment script: `src/sparse_parity/experiments/exp_grokfast_curriculum_scale.py`
- Results JSON: `results/exp_grokfast_curriculum_scale/results.json`
