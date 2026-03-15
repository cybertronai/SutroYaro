# Experiment: Which Proxy Predicts GPU Energy?

**Date**: 2026-03-14
**Status**: INCONCLUSIVE
**Issue**: #6

## Question

Yaroslav said ARD is too coarse as an energy proxy. Is DMC better?

## Setup

Ran 8 experiments on an NVIDIA L4 via Modal Labs (`bin/gpu_energy.py`). Measured real watts via pynvml during execution. Compared against ARD and DMC from the local harness.

Methods too fast for the power sampler (under 5ms) were excluded because pynvml couldn't get a reading before they finished.

## Data

| Challenge | Method | Joules | ARD | DMC | Time (ms) |
|-----------|--------|--------|-----|-----|-----------|
| parity | sgd | 8.601 | 8,504 | 1,278,460 | 533 |
| parity | fourier | 0.878 | 11,980,500 | 78,140,662,852 | 54 |
| sum | sgd | 0.033 | 20 | 2,862 | 2.0 |
| sum | km | 0.051 | 92 | 20,632 | 3.2 |
| and | sgd | 0.291 | 29,164 | 52,885,890 | 18 |
| and | km | 0.046 | 92 | 165,060 | 2.9 |

GF(2) and parity-KM finished in under 5ms, before the power sampler could take a reading.

## Correlation

| Predictor | Spearman r | p-value |
|-----------|-----------|---------|
| Wall-clock time | 1.000 | 0.000 |
| ARD | 0.812 | 0.050 |
| DMC | 0.771 | 0.072 |

## Why this is inconclusive

**The workloads are too small to stress the GPU.** The L4 drew constant 16.1W across all methods. That's idle power. The GPU wasn't doing enough work for memory access patterns (what ARD and DMC measure) to affect power draw. Joules = 16.1W * seconds for every data point.

This means:
- We cannot tell whether ARD or DMC is the better energy proxy from this data. The question requires workloads large enough to cause variable power draw.
- Only 6 data points. ARD's p-value is 0.050 (borderline). DMC's is 0.072 (not significant).
- Two runs gave different idle wattages (12.5W and 16.1W), so the absolute joule numbers vary between runs.

## What we did learn

1. **Measuring GPU energy on sparse parity via pynvml is not useful.** The workloads are too small. The GPU is idle. You're measuring how long the CPU takes to set up the problem, not how much energy the algorithm uses.

2. **The ARD vs DMC question is still open.** Answering it requires a workload that actually stresses GPU memory (nanoGPT scale, or at minimum n=1000+ with large batch sizes).

3. **The Modal + pynvml pipeline works.** `bin/gpu_energy.py` runs, measures watts, costs under $0.001. The infrastructure is ready for when the workloads get large enough to produce real data.

## Next step: PTX-level energy measurement

Yaroslav's actual goal is to verify the Bill Dally energy numbers (register 5pJ, L1 20pJ, L2 100pJ, HBM 640pJ) on real hardware. That means writing CUDA kernels (or PTX instructions) that explicitly target each memory tier and measuring the energy difference per operation.

This is not something pynvml can do (it reports total GPU power, not per-instruction energy). It requires either:
- NVIDIA's Nsight Compute profiler with energy counters
- Custom PTX kernels that isolate register vs shared memory vs L2 vs HBM access, timed and power-sampled at high frequency
- Or a simpler version: two CUDA kernels doing the same computation, one using only shared memory, the other hitting HBM. Measure the power difference.

The Modal L4 supports all of these. The infrastructure (`bin/gpu_energy.py`, Modal account, pynvml) is a starting point but the experiment needs CUDA code, not Python numpy.

## How to reproduce

```bash
# Local proxy metrics (ARD, DMC, time)
PYTHONPATH=src python3 bin/reproduce-all

# GPU energy measurement (requires Modal account, costs ~$0.001)
pip install modal
modal token set
modal run bin/gpu_energy.py
```

## Files

- Script: `bin/gpu_energy.py`
- This document: `findings/exp_proxy_comparison.md`
- GPU energy baseline: `findings/gpu_energy_baseline.md`
- Reproduce: `modal run bin/gpu_energy.py` (requires Modal account)
