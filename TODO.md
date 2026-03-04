# Research TODO — Sparse Parity Energy Efficiency

> Each task is self-contained. Run them independently in separate Claude Code sessions.
> Working directory: `/Users/yadkonrad/dev_dev/year26/feb26/SutroYaro`
> Do NOT push to GitHub. Commit locally only.

## Status

- [x] Task 1-5 (original homework): DONE — 100% accuracy on 20-bit
- [ ] Task A: ARD on winning config
- [ ] Task B: Instrument mini-batch SGD for ARD
- [ ] Task C: Per-layer forward-backward on winning config
- [ ] Task D: Scale stress test (n=50, n=100, k=5)
- [ ] Task E: Forward-Forward on sparse parity
- [ ] Task F: Document prompting strategies
- [ ] Task G: Karpathy Names task
- [ ] Task H: Autonomous ARD improvement loop

---

## Task A: Measure ARD on the Winning Config

**Priority**: HIGH — this is the core deliverable for the next Sutro meeting
**Estimated time**: 10-15 min

### Context
We solved 20-bit sparse parity with LR=0.1, batch_size=32, n_train=500 (100% accuracy in 5 epochs). But we only measured ARD on the OLD broken config (54% accuracy). The challenge asks us to measure AND improve energy usage on a working solution.

### What to do
1. Read `src/sparse_parity/experiments/exp1_fix_hyperparams.py` to understand the winning config
2. Read `src/sparse_parity/tracker.py` for the MemTracker API
3. Create `src/sparse_parity/experiments/exp_a_ard_winning.py` that:
   - Uses the winning config: n_bits=20, k_sparse=3, hidden=1000, LR=0.1, WD=0.01, n_train=500, n_test=200
   - Trains with mini-batch SGD (batch_size=32) until >95% test accuracy
   - Then instruments ONE training step with MemTracker (tracker on step 0 of next epoch)
   - Runs all 3 training variants: standard backprop, fused, per-layer
   - Compares ARD across methods
   - Saves results to `results/exp_a_ard_winning/`
4. Write findings to `findings/exp_a_ard_winning.md`
5. Commit locally (no push)

### Key question to answer
Does the per-layer update still give better ARD on the winning config? How much?

### Files to read first
- `src/sparse_parity/train.py` (standard backprop with tracker)
- `src/sparse_parity/train_fused.py` (fused variant)
- `src/sparse_parity/train_perlayer.py` (per-layer variant)
- `src/sparse_parity/tracker.py` (MemTracker)
- `src/sparse_parity/experiments/exp1_fix_hyperparams.py` (winning config reference)

---

## Task B: Instrument Mini-Batch SGD for ARD

**Priority**: HIGH — needed for accurate energy measurement
**Estimated time**: 20-30 min
**Depends on**: Can run independently, but findings feed into Task A

### Context
Our MemTracker instruments single-sample forward+backward passes. But the winning config uses mini-batch SGD (batch_size=32) which has fundamentally different memory access patterns:
- Read 32 samples, accumulate gradients, then update once
- Parameters are read once per batch (not 32 times)
- Gradient buffers are accumulated (written 32 times, read once for update)
This changes ARD significantly — batch training should have BETTER ARD because parameters stay in cache across the batch.

### What to do
1. Read `src/sparse_parity/tracker.py` for the current MemTracker
2. Read `src/sparse_parity/experiments/exp1_fix_hyperparams.py` for the mini-batch training loop
3. Create `src/sparse_parity/experiments/exp_b_batch_ard.py` that:
   - Implements a mini-batch training step with MemTracker instrumentation
   - For each batch of 32 samples:
     - Track parameter reads (W1, b1, W2, b2) — read ONCE at batch start
     - Track per-sample forward+backward (activations, gradients)
     - Track gradient accumulation across samples
     - Track parameter update (write ONCE at batch end)
   - Compare single-sample ARD vs batch-32 ARD
   - Report: how much does batching improve ARD?
4. Write findings to `findings/exp_b_batch_ard.md`
5. Commit locally

### Key question to answer
How much does mini-batch SGD improve ARD compared to single-sample? Is batch_size a lever for energy efficiency?

### Files to read first
- `src/sparse_parity/tracker.py`
- `src/sparse_parity/experiments/exp1_fix_hyperparams.py`

---

## Task C: Per-Layer Forward-Backward on Winning Config

**Priority**: MEDIUM — tests if the novel algorithm works at scale
**Estimated time**: 15-20 min

### Context
`train_perlayer.py` implements per-layer forward-backward: update each layer before proceeding to the next. This CHANGES the math (gradients computed with already-updated parameters). On 3-bit parity it gave 9.1% ARD improvement with no accuracy loss. We haven't tested it on 20-bit with the correct hyperparams.

Sprint 1's conclusion was: "you'd need a fundamentally different algorithm — like computing Layer 1's backward and update before proceeding to Layer 2's forward." That's exactly what train_perlayer.py does.

### What to do
1. Read `src/sparse_parity/train_perlayer.py`
2. Read `src/sparse_parity/experiments/exp1_fix_hyperparams.py` for the winning config
3. Create `src/sparse_parity/experiments/exp_c_perlayer_20bit.py` that:
   - Uses winning config: n_bits=20, k_sparse=3, hidden=1000, LR=0.1, WD=0.01, n_train=500
   - Trains with per-layer forward-backward (single-sample, like train_perlayer.py)
   - Tracks accuracy and ARD
   - Compares convergence: does per-layer still solve 20-bit? How many epochs?
   - If it converges, compare ARD against standard backprop
4. Write findings to `findings/exp_c_perlayer_20bit.md`
5. Commit locally

### Key question to answer
Does per-layer forward-backward still converge on 20-bit? If yes, what's the ARD improvement?

### Files to read first
- `src/sparse_parity/train_perlayer.py`
- `src/sparse_parity/experiments/exp1_fix_hyperparams.py`

---

## Task D: Scale Stress Test (n=50, n=100, k=5)

**Priority**: MEDIUM — maps the frontier
**Estimated time**: 30-60 min (longer runs)

### Context
We solved n=20, k=3. The literature says SGD needs ~n^O(k) iterations. For n=50/k=3 that's ~125,000 steps. For k=5 it's ~n^5 which gets very expensive. At what point do we need fundamentally different algorithms?

### What to do
1. Read `src/sparse_parity/experiments/exp1_fix_hyperparams.py`
2. Create `src/sparse_parity/experiments/exp_d_scaling.py` that:
   - Tests configs: (n=30,k=3), (n=50,k=3), (n=20,k=5), (n=50,k=5)
   - Uses winning hyperparams: LR=0.1, batch=32, WD=0.01
   - For each config: hidden=2*n, n_train=max(500, 10*n), max_epochs=500
   - Tracks: epochs_to_90pct, total_steps, wall_time, final_accuracy
   - Plots scaling curve: n vs epochs_to_solve
   - If pure Python is too slow, reduce hidden or use fewer epochs and note where it fails
3. Write findings to `findings/exp_d_scaling.md` with scaling table
4. Commit locally

### Key question to answer
Where does standard SGD become impractical? At what n/k do we NEED energy-efficient alternatives?

### Files to read first
- `src/sparse_parity/experiments/exp1_fix_hyperparams.py`
- `research/sparse-parity-literature.md` (for theoretical complexity bounds)

---

## Task E: Forward-Forward on Sparse Parity

**Priority**: MEDIUM — core Sutro Group research interest
**Estimated time**: 45-60 min

### Context
Hinton's Forward-Forward algorithm (Meeting #2 topic) replaces backprop with two forward passes — one positive, one negative. Each layer has its own local objective (goodness = sum of squared activations). This should have much smaller ARD by design since parameters are only accessed locally.

The question: can FF actually solve sparse parity? It's a harder learning algorithm but potentially much more energy-efficient.

### What to do
1. Read `research/sparse-parity-literature.md` for context
2. Read the homework for Meeting #2 in `docs/homework/index.md`
3. Create `src/sparse_parity/experiments/exp_e_forward_forward.py` that:
   - Implements Forward-Forward for sparse parity:
     - Positive pass: real data with correct labels → maximize goodness
     - Negative pass: data with wrong labels → minimize goodness
     - Goodness = sum of squared ReLU activations per layer
     - Each layer updates independently using only local information
   - Config: start with n_bits=3, k_sparse=3 (easy case first)
   - If 3-bit works, try n_bits=20, k_sparse=3
   - Instrument with MemTracker to measure ARD
   - Compare ARD against standard backprop
4. Write findings to `findings/exp_e_forward_forward.md`
5. Commit locally

### Key question to answer
Can Forward-Forward solve sparse parity? If yes, what's the ARD compared to backprop?

### Hints
- For negative data generation: shuffle labels randomly, or generate random inputs with random labels
- Threshold for classification: if goodness > threshold → positive, else negative
- Start with a simple 2-layer version matching our existing architecture
- This is exploratory — partial results are valuable

### Files to read first
- `src/sparse_parity/model.py` (existing architecture to adapt)
- `src/sparse_parity/tracker.py` (ARD measurement)
- `docs/homework/index.md` (Meeting #2 FF exercises)

---

## Task F: Document Prompting Strategies

**Priority**: LOW — meta-question from the challenge
**Estimated time**: 15 min

### Context
Challenge Question 3: "what are the prompting strategies/approaches that are useful here?"

We used a specific workflow to go from 54% to 100%: literature review → hypothesis → experiment → measure. This is worth documenting as a reusable template.

### What to do
1. Read `docs/plans/2026-03-04-beat-20bit-research-plan.md`
2. Read `findings/exp1_fix_hyperparams.md` and `findings/exp4_grokfast.md`
3. Create `findings/prompting-strategies.md` documenting:
   - The workflow that worked: literature search → diagnose → fix → verify
   - What prompts were effective (asking for arxiv papers, asking for practical hyperparams)
   - What didn't work (GrokFast was counterproductive — literature isn't always right for your regime)
   - Template for future research cycles
4. Commit locally

### Files to read first
- `docs/plans/2026-03-04-beat-20bit-research-plan.md`
- `findings/exp1_fix_hyperparams.md`
- `findings/exp4_grokfast.md`

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

### What to do
1. Read `src/sparse_parity/tracker.py` and `src/sparse_parity/train.py`
2. Create `src/sparse_parity/experiments/exp_h_auto_improve.py` that:
   - Starts with the winning config (LR=0.1, batch=32)
   - Measures baseline ARD
   - Tries N variations of the backward pass (reorder operations, fuse reads, cache values)
   - For each variation: verify accuracy is maintained, measure ARD
   - Report best ARD improvement found
   - The variations should be programmatically generated (e.g., try all permutations of update order)
3. Write findings to `findings/exp_h_auto_improve.md`
4. Commit locally

### Key question to answer
Can a systematic search over backward pass orderings find a better ARD than hand-designed fused/per-layer?

### Files to read first
- `src/sparse_parity/train.py`
- `src/sparse_parity/train_fused.py`
- `src/sparse_parity/train_perlayer.py`
- `src/sparse_parity/tracker.py`
