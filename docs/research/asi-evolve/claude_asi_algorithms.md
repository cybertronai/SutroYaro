# ASI-Evolve: Algorithm Search Strategy for ByteDMD-Optimized Learning Rules

**Agent**: Claude Code / GLM-5.1 | **Role**: Algorithms Expert
**Date**: 2026-04-07
**Source**: ASI-Evolve paper (arXiv:2603.29640) | ByteDMD (github.com/cybertronai/ByteDMD)

## Executive Summary

ASI-Evolve discovered SOTA RL algorithms and neural architectures through evolutionary search over mathematical program space, not hyperparameter grids. Its key mechanism is an LLM proposing complete algorithmic formulations (loss functions, gating mechanisms) conditioned on a literature-indexed cognition base and accumulated experimental memory. This report extracts the specific mutation operators and search-space constraints that would let our agent invent learning rules achieving high spatial locality under ByteDMD.

The core insight: ByteDMD penalizes non-local reads at sqrt(depth). Any learning rule that shuffles data through temporaries (standard backprop) or reads the same weight matrix repeatedly (SGD) is structurally doomed. The winning approaches keep active variables at the top of the LRU stack by processing data in small, sequential blocks and never materializing large intermediates.

## 1. What ASI-Evolve Actually Did in RL Algorithm Design

### The search space

ASI-Evolve did not sweep hyperparameters. It searched over the mathematical formulation of the loss function itself. The Researcher LLM proposed complete algorithm designs including:

- Advantage estimation mechanisms (how to convert rewards into learning signals)
- Gradient clipping strategies (asymmetric, dynamic radius)
- Gradient regularization (dropout on high-influence tokens)
- Update budget constraints (bounded policy shift per step)

Over 300 evolutionary rounds, it generated candidates that modified the loss function at the level of mathematical equations. Ten beat GRPO; three held up at 14B parameter scale.

### The two winning algorithms

**Pairwise Asymmetric Optimization** (best on AMC32: 80.0 vs GRPO's 67.5):
- Replaces group-mean advantage with tanh-normalized pairwise reward differences
- Asymmetric clipping: positive and negative advantages get different epsilon bounds
- High-impact gradient dropout: masks gradients for tokens where `probability * advantage` is highest

**Budget-Constrained Dynamic Radius** (best on AIME25: 30.0 vs GRPO's 20.0):
- Percentile-based advantage normalization: `(r - c) / s` where c, s are reward distribution percentiles
- Global update budget: each token gets a trusted update radius inversely proportional to its advantage magnitude
- Exponential bound: `exp(c) * |A| <= z_cap` guarantees total policy shift stays within budget

### What made the search work

Three structural properties of ASI-Evolve's loop produced these results:

1. **Cognition base priming**: The search started with 10 post-GRPO papers on variance reduction and KL penalties. This gave the LLM "plausible mathematical directions" rather than blind random walks.

2. **Analyzer compression**: Each round's full experimental output (metrics, training dynamics, benchmark breakdowns) was distilled into a compact decision-oriented report. Subsequent rounds could retrieve and build on these analyses.

3. **UCB1 sampling**: When cognition is present, simple upper-confidence-bound exploitation beats quality-diversity methods (MAP-Elites). The cognition base already provides directional guidance; MAP-Elites' enforced diversity becomes redundant.

## 2. What ASI-Evolve Discovered in Architecture Design

### The consistent pattern

Across 1,773 exploration rounds producing 105 SOTA architectures, one pattern dominated: **adaptive multi-scale routing replacing fixed allocation**. Every winning architecture dynamically routes computation based on input content rather than using static pathways.

Five representative architectures, all beating DeltaNet:

| Architecture | Key Innovation | Score |
|---|---|---|
| FusionGated-FIRNet | Independent sigmoid gates replacing softmax | 52.01% (+0.97) |
| Hier-GateNet | Dynamic learnable floor values for delta-path | 51.79% |
| Content-SharpRouter | Learnable temperature for routing sharpness | 51.61% |
| PathGate-FusionNet | Hierarchical budget allocation across ranges | 51.48% |
| AdaMulti-PathGateNet | Entropy penalties preventing mode collapse | 51.11% |

DeltaNet baseline: 51.04%.

### Specific mutations that produced winners

ASI-Evolve uses diff-based editing (localized modifications over parent programs) rather than full code generation from scratch. The mutations that produced the top architectures:

1. **Softmax to independent sigmoid**: Removing mutual exclusion from routing decisions. Softmax forces a probability distribution; independent sigmoids allow multi-path activation simultaneously. This was the single most impactful mutation (FusionGated-FIRNet).

2. **Delta-path protection**: Adding explicit mechanisms to prevent the long-range memory update path from collapsing. The delta-rule path is critical for reasoning; without protection, routing entropy collapses it.

3. **Hierarchical budget allocation**: Two-stage gating where stage 1 determines coarse allocation (local vs contextual) and stage 2 distributes within each category.

4. **Learnable temperature**: Making routing sharpness a trainable parameter rather than a fixed constant. Prevents premature commitment to pathways.

5. **Entropy-based anti-collapse**: Adding penalty terms that prevent routing from degenerating to a single path.

### Engineering safeguards

Three mechanisms filtered proposals before expensive training:

- **Static check agent**: Verifies complexity bounds, chunk-wise structure, causal mask correctness
- **Debug agent**: Inspects error traces, attempts targeted fixes, retries (up to 2)
- **Novelty check**: Rejects proposals whose motivation is too similar to existing entries

## 3. ByteDMD Mechanics and Implications

### How ByteDMD counts

```
C = sum over all bytes b of ceil(sqrt(D(b)))
```

D(b) = depth of byte b in the LRU stack. Only reads cost; writes and computation are free. A float32 at element depth d occupies 4 byte positions, each charged at its byte-level depth.

Cost per element at depth d (bpe=4 for float32):
```
cost = sum_usqrt(d * 4) - sum_usqrt((d-1) * 4)
```

### Key benchmarks from ByteDMD repo

| Operation | ByteDMD Cost | Why |
|---|---|---|
| 4x4 matmul (i-j-k) | 948 | Standard loop order |
| 4x4 matmul (snake-j) | 906 | Cache-friendly traversal |
| 4x4 Strassen | 2,435 | Temporaries bury data in stack |
| microGPT forward pass | 7,047 | Residual connections, multiple layers |
| Naive attention N=128 | 13.7M | Full NxN matrix materialized |
| Flash attention N=128 | 4.2M | Block-wise, no full matrix |

Strassen is 2.6x worse than simple tiled matmul. Flash attention is 3.25x better than naive at N=128. The lesson: algorithmic complexity (FLOP count) is nearly irrelevant; data movement pattern is everything.

### The 10 rules of ByteDMD-efficient algorithms

1. **Keep the working set small.** Every simultaneous value competes for stack positions.
2. **Use data immediately after reading it.** Delay pushes it deeper.
3. **Loop ordering matters more than FLOP count.** Snake traversal beats Strassen.
4. **Minimize temporaries.** Each intermediate result is a new allocation that pushes others deeper.
5. **Use smaller data types.** int8 costs 4x less than float32 at the same element depth.
6. **Writes are free.** Overwriting costs nothing; reading the old value is what costs.
7. **Branching decisions are cheap.** Control flow booleans escape the stack.
8. **Constants are nearly free.** Hardcoded values avoid the tracking model.
9. **The cost plateaus at moderate depth.** Depths 1-4 cost 1-2; depths 5-9 cost 3.
10. **Locality beats parallelism.** Complete one layer before starting the next.

## 4. Mutation Operators for Our Agent

Based on ASI-Evolve's search strategy and ByteDMD's cost model, here are specific mutation operators the SutroYaro agent should use when searching for ByteDMD-efficient learning rules.

### Category 1: Data Flow Mutations

These modify how data moves through the algorithm, which is the primary cost factor.

**M1: Working Set Reduction**
- Mutation: Reduce the number of simultaneously live variables.
- ByteDMD rationale: Fewer live variables means shallower stack depths.
- Application: Replace algorithms that maintain multiple buffers (SGD with momentum maintains weight, gradient, velocity, momentum buffer) with single-buffer variants.
- ASI-Evolve analogue: The static check agent verifies complexity bounds before execution.

**M2: Access Reordering**
- Mutation: Change the order in which arrays are traversed.
- ByteDMD rationale: Snake traversal beats standard loop orders. Sequential access keeps data at the top of the LRU stack.
- Application: In any weight update loop, iterate in memory-layout order (row-major for C-contiguous arrays).
- ASI-Evolve analogue: Diff-based editing for localized modifications to loop structure.

**M3: Temporaries Elimination**
- Mutation: Replace intermediate computations with in-place operations.
- ByteDMD rationale: Temporaries are new allocations that push other data deeper. Strassen loses to tiled matmul because of this.
- Application: `w += lr * grad` (one write, free) instead of `delta = lr * grad; w = w + delta` (one read of delta, one read of w, one write).

**M4: Quantization**
- Mutation: Reduce data type width (float32 to float16 to int8 to binary).
- ByteDMD rationale: Cost is per-byte. float32 at depth d costs 4x what int8 costs at the same element depth.
- Application: Sign SGD (already proven to solve parity in DISCOVERIES.md) is a ByteDMD win because the gradient is 1 bit.
- ASI-Evolve analogue: Not directly explored, but the architecture search found that simpler routing mechanisms (sigmoid vs softmax) won.

### Category 2: Algorithmic Structure Mutations

These modify the mathematical form of the learning rule.

**M5: Single-Pass Constraint**
- Mutation: Enforce that each data element is read at most once per training step.
- ByteDMD rationale: Re-reading data after intervening reads pushes it deeper. Single-pass algorithms keep depth bounded.
- Application: Replace multi-epoch training with one-shot methods. GF(2) (DMC 8,607) and KM-min (DMC 3,578) both solve parity in a single pass.
- ASI-Evolve analogue: The chunk-wise computation pattern in architecture design is essentially a single-pass constraint.

**M6: Local Update Radius**
- Mutation: Bound the update magnitude inversely proportional to the variable's read depth.
- ByteDMD rationale: ASI-Evolve's Budget-Constrained Dynamic Radius showed that capping updates for high-influence parameters prevents overfitting. For ByteDMD, variables at high depth should get smaller updates (they are expensive to read, so we should need fewer reads of them).
- Application: In SGD, W1 is at high depth (it dominates reads). Give W1 a smaller learning rate than W2, or skip W1 updates on some steps.
- ASI-Evolve analogue: Budget-Constrained Dynamic Radius (`exp(c) * |A| <= z_cap`).

**M7: Asymmetric Gradient Treatment**
- Mutation: Apply different update rules to positive vs negative gradients.
- ByteDMD rationale: ASI-Evolve's Pairwise Asymmetric Optimization showed that asymmetric clipping (different epsilon for positive/negative advantages) beats symmetric clipping. For ByteDMD, this means: positive gradients (which reinforce features) might be worth the read cost, while negative gradients (which suppress noise) could be applied more cheaply (e.g., via sign only).
- Application: `if grad > 0: w += lr * grad (full precision read)` vs `if grad < 0: w -= lr (sign only, no gradient magnitude read)`.
- ASI-Evolve analogue: Pairwise Asymmetric Optimization's asymmetric clipping.

**M8: Gradient Dropout for High-Cost Variables**
- Mutation: Stochastically skip updates for variables at high LRU depth.
- ByteDMD rationale: ASI-Evolve's high-impact gradient dropout masks gradients for the most influential tokens. For ByteDMD, the "most expensive" variables (those at high depth) get their updates dropped probabilistically.
- Application: On each step, skip updating W1[i] with probability proportional to its stack depth. This reduces total reads without losing convergence (similar to dropout regularization).
- ASI-Evolve analogue: High-impact gradient dropout from Pairwise Asymmetric Optimization.

### Category 3: Search Space Constraints

These are hard constraints that filter proposals before execution, like ASI-Evolve's static check agent.

**C1: Working Set Size Bound**
- Rule: The number of simultaneously live tracked values must not exceed a fixed budget (e.g., 20 floats = 80 bytes for float32).
- Rationale: With working set <= 20, all reads are at depth <= 20, costing at most ceil(sqrt(20)) = 5 per byte. A float32 at depth 20 costs sum_usqrt(80) - sum_usqrt(76) = 22 per access. Compare to SGD's W1 at depth ~5000: ceil(sqrt(20000)) = 142 per byte.
- Implementation: Static analysis of the algorithm's live variable count before running experiments.

**C2: No Full-Matrix Re-read**
- Rule: No training step may read all elements of any matrix with more than N elements (where N is a tunable threshold, e.g., 200 floats).
- Rationale: SGD reads all 4000 elements of W1 (20 * 200) every forward and backward pass. This is the dominant cost.
- Implementation: TrackedArray already counts reads. The constraint is: any single training step's trace must not contain more than N reads from any single array.

**C3: Temporaries Budget**
- Rule: The number of new allocations per training step must not exceed a fixed budget.
- Rationale: Each temporary is a new stack entry that pushes existing data deeper. Strassen's 7 sub-multiplications create 7 temporaries.
- Implementation: Count `allocate()` calls in the LRU stack per training step.

**C4: Data Type Preference**
- Rule: Proposals using smaller data types get a multiplicative bonus in the fitness function.
- Rationale: ByteDMD cost is per-byte. int8 is 4x cheaper than float32 at identical element depth.
- Implementation: Fitness = accuracy / (ByteDMD_cost * bytes_per_element). This makes int8 proposals inherently fitter than float32 at the same algorithmic cost.

## 5. Putting It Together: A Search Protocol

### Step 1: Initialize cognition base (manual)

Populate the cognition base with proven ByteDMD insights from DISCOVERIES.md:

```
Entry 1: "KM-min solves parity with DMC 3,578. Single influence sample per bit.
          Working set: 20 floats. All reads at depth <= 20."

Entry 2: "SGD baseline has DMC 1,278,460. W1 dominates 75% of reads.
          Each element read ~40 times across epochs."

Entry 3: "Strassen is 2.6x worse than tiled matmul in ByteDMD despite fewer FLOPs.
          Temporaries are the enemy."

Entry 4: "Sign SGD solves parity 2x faster than standard SGD in epochs.
          Gradient magnitude is 1 bit: minimal ByteDMD cost for the update."

Entry 5: "GF(2) Gaussian elimination solves in 500us, DMC 8,607.
          Single pass over n+1 samples. k-independent."

Entry 6: "All local learning rules fail on parity (Hebbian, PC, EP, TP).
          Parity requires k-th order interaction detection."

Entry 7: "Writes are free in ByteDMD. Reading old values is what costs.
          In-place updates are strictly better than creating new arrays."

Entry 8: "Flash attention is 3.25x better than naive at N=128.
          Block-wise processing never materializes full NxN matrix."

Entry 9: "Loop ordering matters: snake-j matmul beats i-j-k by 4% at 4x4.
          Access data in memory layout order."

Entry 10: "Cost plateaus at moderate depth. Depths 1-4 cost 1-2, depths 5-9 cost 3.
           Keeping working set under 9 floats means max cost of 3 per element."
```

### Step 2: Define the search space

The agent searches over learning rules parameterized as:

```python
class LearningRule:
    # What data to read (constrained by C2)
    read_pattern: str  # "full" | "random_k" | "influence_top" | "sequential_block"

    # How to compute the update (mutated by M5-M8)
    update_fn: str  # "sgd" | "sign_sgd" | "hebbian" | "influence" | "hybrid"

    # Data type (mutated by M4)
    dtype: str  # "float32" | "float16" | "int8" | "binary"

    # Update frequency (mutated by M6, M8)
    update_schedule: str  # "every_step" | "every_k_steps" | "probabilistic_skip"

    # Access order (mutated by M2)
    traversal_order: str  # "row_major" | "col_major" | "snake" | "random"

    # Working set budget (constrained by C1)
    max_live_variables: int  # budget, e.g., 20

    # Temporaries budget (constrained by C3)
    max_temporaries_per_step: int  # budget, e.g., 5
```

### Step 3: Run the evolutionary loop

Each round:

1. **Sample** 3 parent nodes from the database via UCB1 (not random, not MAP-Elites; ASI-Evolve showed UCB1 wins with a good cognition base).
2. **Retrieve** relevant cognition entries (ByteDMD heuristics above) via embedding similarity.
3. **Generate** a candidate learning rule via diff-based editing of a parent. The LLM proposes modifications to the `LearningRule` parameters, motivated by the cognition entries.
4. **Static check** (before execution): verify C1, C2, C3 constraints. Reject if violated.
5. **Execute** the rule on the sparse parity benchmark. Record ByteDMD via TrackedArray.
6. **Analyze**: distill results into a compact report. Fitness = accuracy / ByteDMD_cost (higher is better, only valid when accuracy > 90%).
7. **Store** the node (motivation, rule parameters, results, analysis, fitness) in the database.

### Step 4: Expected search trajectory

Based on ASI-Evolve's dynamics (78% of top performers came from the second half of search, built on first-half discoveries):

1. **Rounds 1-50** (cognition-driven): The agent discovers that single-pass methods (KM-min, GF2-like) dominate SGD variants. Sign SGD with sequential access is a strong baseline. DMC in the 3,000-10,000 range.

2. **Rounds 50-150** (experience-driven): The agent combines insights: sign SGD + influence-based selective reading + int8 quantization. Working set drops below 10 floats. DMC approaching 1,000.

3. **Rounds 150-300** (novel exploration): The agent discovers genuinely new update rules that avoid reading W1 entirely by maintaining a separate compressed state. Potential for DMC under 1,000.

## 6. Specific Candidate Algorithms to Seed the Search

Based on the analysis, these are the most promising starting points for ByteDMD-optimized learning rules:

### Candidate A: Block-Local Sign SGD

```
For each block of k consecutive weights in W1:
    Read the block (depth <= k)
    Compute sign of gradient from 1 sample
    Update: w[i:i+k] += lr * sign(grad)
    (Write is free)
```

ByteDMD analysis: Working set = k + n + 1 floats. All reads at depth <= k+n+1. For k=3, n=20: depth <= 24, cost per float32 <= 25 per access. Total DMC estimate: ~1,200.

This exploits M1 (small working set), M3 (no temporaries, in-place update), M4 (sign = 1-bit gradient), M5 (each weight read once per step), and M2 (sequential block access).

### Candidate B: Influence-Guided Sparse Read

```
For each bit i:
    Read x[i] (depth 1-20)
    Read y (depth 1)
    Compute influence proxy: sign(x[i] * y) accumulated over 1 sample
    If |influence| > threshold:
        Update w1[i, :] += lr * influence * x[i]
```

ByteDMD analysis: Only reads n+1 values per step. Working set = 2 floats at any time. DMC estimate: ~800.

This is essentially KM-min as a learning rule rather than a solver. It exploits the single-sample influence property proven in DISCOVERIES.md.

### Candidate C: Quantized Per-Layer with Skip

```
For each layer l:
    With probability p_l (inversely proportional to layer size):
        Read block of W_l in sequential order
        Compute int8 gradient from 1 sample
        w_l[block] += lr * int8(grad)
```

ByteDMD analysis: Large layers get updated less frequently. int8 saves 4x on byte cost. Sequential access keeps depth bounded. DMC estimate depends on p_l tuning.

This exploits M6 (local update radius via skip probability), M4 (int8), and M2 (sequential).

## 7. What Not to Do (Anti-Patterns from ASI-Evolve + ByteDMD)

ASI-Evolve's novelty check rejected proposals too similar to existing entries. Our equivalent should reject proposals matching these anti-patterns:

1. **Full-batch gradient computation**: Reading all training data to compute one update. Fourier's DMC of 78 billion is the canonical example.

2. **Momentum/Adam-style multi-buffer**: Maintaining velocity, momentum, and second-moment buffers triples the working set. ASI-Evolve found that simpler mechanisms (asymmetric clipping, budget constraints) beat complex optimizer state.

3. **Strassen-like decomposition**: Breaking a matrix operation into sub-problems that create temporaries. 2.6x worse than simple tiled access.

4. **Random access patterns**: Accessing array elements in non-sequential order (e.g., random feature selection). Each random access sends the target to the top of the LRU stack but buries whatever was there.

5. **Cross-layer interleaving**: Reading from W1, then W2, then back to W1. Each switch pushes the previous layer's data deeper. ASI-Evolve's per-layer architecture search validates completing one layer before moving to the next.

## Files

- Analysis: `docs/research/asi-evolve/claude_asi_algorithms.md` (this file)
- Source Paper: https://arxiv.org/abs/2603.29640
- ByteDMD Implementation: https://github.com/cybertronai/ByteDMD
- ByteDMD Examples: https://github.com/cybertronai/ByteDMD-examples
- SutroYaro Context: DISCOVERIES.md, CLAUDE.md, LAB.md, AGENT.md
- Companion Analysis (Memory): `docs/research/asi-evolve/kimi_asi_memory.md`
