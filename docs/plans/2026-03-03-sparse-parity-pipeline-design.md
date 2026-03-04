# Sparse Parity Pipeline Design

**Date**: 2026-03-03
**Status**: Approved
**Goal**: Build end-to-end sparse parity challenge pipeline (all 5 homework tasks)

## Summary

Fresh implementation of the Sutro Group sparse parity challenge with Yaroslav's Sprint 1 code as read-only reference. Pure Python, no PyTorch, <1 second runtime. Modular package structure optimized for Claude Code / Ralph Wiggum autonomous iteration.

## Constraints

- Pure Python (no numpy, no torch)
- Total runtime <1 second for 3-bit; <2 seconds for 20-bit
- Fixed random seeds for reproducibility
- Each module small enough for Claude to read/edit in one pass
- Tests for each phase so Ralph can verify correctness

## Directory Structure

```
src/sparse_parity/
  __init__.py          # Package init
  config.py            # All constants (N_BITS, HIDDEN, LR, etc.)
  data.py              # Phase 1: dataset generation
  model.py             # Phase 2: MLP init + forward pass
  tracker.py           # Phase 3: MemTracker (ARD measurement)
  train.py             # Training loop - standard backprop
  train_fused.py       # Phase 4a: Fused layer-wise updates
  train_perlayer.py    # Phase 4b: Per-layer forward-backward
  metrics.py           # Loss, accuracy, JSON/markdown reporting
  run.py               # Main entry point: all phases sequentially

src/sparse_parity/reference/
  sparse_parity_benchmark.py   # Yaroslav's original from cybertronai/sutro (read-only)

tests/
  test_data.py         # Verify parity labels match secret indices
  test_model.py        # Verify forward pass shapes and basic correctness
  test_tracker.py      # Verify ARD values are sane
  test_train.py        # Verify >90% accuracy on 3-bit parity
  test_scaling.py      # Verify 20-bit still converges

results/               # Output directory (git-tracked)
  {timestamp}_{phase}.json     # Structured metrics per experiment
  {timestamp}_report.md        # Auto-generated findings
  {timestamp}_plots.png        # Loss/accuracy charts
  {timestamp}_ard_comparison.json  # ARD across all methods
```

## Phases

### Phase 1: Data Generation (`data.py`)
- `generate(n_bits, k_sparse, n_samples, seed)` → (xs, ys, secret)
- Random secret parity indices selected once
- Inputs are random {-1, +1} values
- Labels are product of inputs at secret indices
- Test: verify labels by recomputing parity manually

### Phase 2: Baseline Training (`model.py` + `train.py`)
- 2-layer MLP: input → Linear(W1,b1) → ReLU → Linear(W2,b2) → scalar
- Hinge loss: max(0, 1 - f(x)·y)
- SGD with weight decay
- Single-sample cyclic training (no batching)
- Kaiming initialization
- Config: HIDDEN=1000, LR=0.5, WD=0.01
- Target: >90% test accuracy on 3-bit parity
- Test: accuracy check

### Phase 3: ARD Measurement (`tracker.py`)
- MemTracker class tracks read/write operations
- Clock advances by buffer SIZE (floats), not operation count
- Reuse distance = current_clock - last_write_clock for each read
- Weighted average: each float's distance counted equally
- Instrument forward + backward with optional `tracker` parameter
- Output: per-buffer summary, overall weighted ARD
- Test: total floats matches expected for network size

### Phase 4a: Fused Layer-wise Updates (`train_fused.py`)
- Same math as Phase 2 but reordered:
  - Compute Layer 2 gradients → update W2,b2 immediately
  - Compute Layer 1 gradients → update W1,b1 immediately
- Gradient buffers consumed right after creation (short ARD)
- Expected improvement: ~16% ARD reduction (matching Sprint 1)
- Test: same accuracy as Phase 2, lower ARD

### Phase 4b: Per-Layer Forward-Backward (`train_perlayer.py`)
- Radical change: compute each layer's forward, backward, and update before proceeding to next layer
- Layer 1: forward → backward → update W1,b1
- Layer 2: forward → backward → update W2,b2
- This CHANGES the math (gradients computed with already-updated parameters)
- Parameters stay in cache between use and update (minimal ARD)
- Expected: significant ARD reduction, accuracy may differ
- Test: check if it still converges; compare ARD

### Phase 5: Scale Up (in `run.py`)
- Re-run Phases 2-4b with N_BITS=20, K_SPARSE=3 (17 noise bits)
- May need more epochs and larger hidden layer
- Compare convergence and ARD across all methods
- Test: verify convergence at scale

## Output Artifacts

Each run produces (in `results/`):
1. **JSON metrics** per phase: accuracy, loss, ARD values, timing, config
2. **Markdown report**: auto-generated summary comparing all methods
3. **PNG plots**: loss curves, accuracy curves, ARD comparison bar chart

## Key Design Decisions

1. **Modular files** over monolithic: each file <200 lines, Claude can read/edit in one pass
2. **Optional tracker** parameter: same forward/backward code works with or without instrumentation
3. **Phase 4b changes the math**: this is intentional and documented; we compare accuracy AND ARD
4. **Results are git-tracked**: reproducible, diffable across runs
5. **Tests per phase**: Ralph can run `python -m pytest tests/` to verify before moving on

## Reference

- Yaroslav Sprint 1: `docs/google-docs/yaroslav-technical-sprint-1.md`
- Original code: `src/sparse_parity/reference/sparse_parity_benchmark.py`
- Challenge spec: `docs/google-docs/challenge-1-sparse-parity.md`
