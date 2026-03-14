# Research TODO — Sparse Parity Energy Efficiency

> Each task is self-contained. Run them independently in separate Claude Code sessions.

## Status

- [x] Task 1-5 (original homework): DONE — 100% accuracy on 20-bit
- [x] Task A: ARD on winning config — [findings](findings/exp_a_ard_winning.md)
- [x] Task B: Instrument mini-batch SGD for ARD — [findings](findings/exp_b_batch_ard.md)
- [x] Task C: Per-layer forward-backward on winning config — [findings](findings/exp_c_perlayer_20bit.md)
- [x] Task D: Scale stress test (n=50, n=100, k=5) — [findings](findings/exp_d_scaling.md)
- [x] Task E: Forward-Forward on sparse parity — [findings](findings/exp_e_forward_forward.md)
- [x] Task F: Document prompting strategies — [findings](docs/research/survey.md) §7 (AI Research Process)
- [ ] Task G: Karpathy Names task
- [ ] Task H: Autonomous ARD improvement loop

---

## Summary

**Phase 1** (Tasks A-F) complete. Key results:
- Per-layer backprop: 3.8% ARD improvement, same convergence
- Batch-32: 17x higher ARD in software metric, but L2 cache eliminates all misses
- SGD breaks when n^k > ~100k gradient steps
- Forward-Forward fails at 58.5% — local objectives can't find k-th order interactions
- Curriculum learning (n=10 → n=50): 14.6x speedup

**Phase 2** (17 experiments) explored alternative algorithms. See [survey](docs/research/survey.md) for full results.

**Remaining:**
- Task G: Independent homework (names.txt character model)
- Task H: Meta-experiment for autonomous ARD optimization (partially addressed by Pebble Game)

---

## Task G: Karpathy Names Task

**Priority**: LOW — separate homework from Meeting #5
**Estimated time**: 45-60 min

### Context
Completely independent task from Meeting #5. Build a character-level model that predicts last 3 characters of names from Karpathy's makemore dataset. Optimize total operations without reducing accuracy. Emmett got 2x memory reduction using Aster.

### What to do
1. Read `docs/homework/index.md` → Meeting #5 section
2. Download names.txt from https://github.com/karpathy/makemore/blob/master/names.txt
3. Create `src/names_task/` directory with:
   - `data.py` — load names, split train/test (1000 each)
   - `model.py` — simple character-level model (bigram or small transformer)
   - `train.py` — training loop with operation counting
   - `run.py` — baseline + optimized comparison
4. Write findings to `findings/names_task.md`
5. Commit locally

### Files to read first
- `docs/homework/index.md` (Meeting #5 section)

---

## Task H: Autonomous ARD Improvement Loop

**Priority**: LOW — meta-experiment, addresses Challenge Q2
**Estimated time**: 30-45 min

### Context
Challenge Question 2: "Can modern AI improve (memory) energy usage?" This is asking whether we can set up an autonomous loop where Claude Code tries different algorithms and measures ARD, iterating toward lower energy.

**Related work**: The [Pebble Game experiment](findings/exp_pebble_game.md) explored 5,758 topological orderings of the backward pass and found optimal orderings. This is a partial answer.

### What to do
1. Read `src/sparse_parity/tracker.py` and `src/sparse_parity/train.py`
2. Read `findings/exp_pebble_game.md` for prior work on operation reordering
3. Consider: is there more to explore beyond ordering? (e.g., algorithmic changes, not just reordering)
4. If yes, create `src/sparse_parity/experiments/exp_h_auto_improve.py`
5. Write findings to `findings/exp_h_auto_improve.md`
6. Commit locally

### Key question to answer
Can a systematic search over backward pass orderings find a better ARD than hand-designed fused/per-layer? (Pebble Game says: 2.2% improvement possible, but read-after-write hazards can break training)

### Files to read first
- `src/sparse_parity/train.py`
- `src/sparse_parity/train_fused.py`
- `src/sparse_parity/train_perlayer.py`
- `src/sparse_parity/tracker.py`
- `findings/exp_pebble_game.md`

---

## Sparse Sum (Challenge 2)

New challenge: y = sum(x[secret_indices]). First-order structure (unlike parity).
Tests whether the infrastructure generalizes. See `docs/research/adding-a-challenge.md`.

- [ ] Run Hebbian on sparse sum (expect success, unlike parity)
- [ ] Run Predictive Coding on sparse sum (expect success)
- [ ] Compare ARD of SGD on sum vs parity at n=20, 50, 100
- [ ] Test if AGENT.md loop works on sparse sum without human help
- [ ] Scale test: sparse sum at n=100/k=10

---

## Sparse AND (Challenge 3)

New challenge: y = product((x[secret]+1)/2). Logical AND over {0,1}.
Output is 1 only when ALL k secret bits are +1. P(y=1) = 1/2^k.
Tests class-imbalanced k-th order interactions. See `docs/research/adding-a-challenge.md`.

- [ ] SGD baseline on sparse AND (n=20, k=3)
- [ ] KM influence baseline on sparse AND (increase influence_samples for low signal)
- [ ] Fourier baseline on sparse AND
- [ ] Compare SGD convergence: AND vs parity at same config
- [ ] Test KM with more influence samples (5 → 20) to handle 1/2^(k-1) signal
- [ ] Scale test: sparse AND at k=5 (P(y=1) = 3%, severe imbalance)
