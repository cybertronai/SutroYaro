# ASI-Evolve: Execution Pipeline Design

**Author**: Qwen 3.6 Plus via Qwen-Code
**Role**: Implementation Engineer (Design + Experiment loop)
**Date**: 2026-04-07

## 1. ASI-Evolve Execution Architecture — What We Have vs What They Have

ASI-Evolve uses a 4-module closed loop: **Researcher → Engineer → Analyzer → Database**. Here is how each maps to our existing infrastructure:

| ASI-Evolve Module | Their Role | Our Equivalent | Gap |
|---|---|---|---|
| **Researcher** | LLM generates Python code from priors (parent code, cognition items, task spec) | Any agent dispatched to write experiment code, prompted with `DISCOVERIES.md` + context | No automated parent-code injection — agents get context via prompts, not via embedding-retrieved ancestors |
| **Engineer** | Sandboxed execution with timeouts, parallelism, scalar fitness | `src/harness.py` — deterministic evaluation across 3 challenges | No timeout, no quick-test rejection, no parallel workers |
| **Analyzer** | Distills raw logs into compact decision-oriented report | Our findings writing step + INDEX.md tracking | No auto-analysis — human/agent writes findings manually |
| **Database (D)** | Persistent store of {motivation, code, results, analysis, score} | `research/log.jsonl` + `findings/*.md` + `results/*/results.json` | Structured but not queryable by embedding similarity |
| **Cognition Base (C)** | Human priors for cold-start, semantic retrieval | `DISCOVERIES.md` + agent prompts | No semantic search / embedding-based retrieval |

The key gap is **orchestration**: ASI-Evolve automates the loop. We run one experiment at a time, manually.

## 2. Wrapping harness.py in an ASI-Evolve Loop

The minimal loop structure to close our execution pipeline:

```
for round_t in range(max_rounds):
    # LEARN: sample from log.jsonl, retrieve relevant DISCOVERIES.md entries
    parents = sample_parents(log_jsonl, discoveries_md, policy="ucb1")

    # DESIGN: generate code (handled by memory-kimi / algorithms-claude prompts)
    candidate_code = generate(parents)

    # EXPERIMENT: our harness
    timeout=300  # 5 min hard limit
    metrics = execute_with_timeout(candidate_code, harness_args, timeout)

    # ANALYZE: distill metrics into finding
    finding = analyze_experiment(candidate_code, metrics, parents)

    # STORE: append to log, update findings/
    append_to_log_jsonl(finding)
```

### 2.1 Experiment Execution (Engineer module)

ASI-Evolve's Engineer does three things I think we should adopt:

**Quick-test rejection**: Before full evaluation, run a lightweight check (e.g., single seed, 50 epochs instead of 200). On n=20/k=3, baseline SGD takes 40 epochs / 0.12s. A 50-epoch quick test would cost <0.15s but reject ~80% of clearly broken proposals (syntax errors, wrong output shape, divergent loss). This saves wall time by avoiding full 200-epoch runs on garbage code.

**Configurable timeout**: Our current experiment loop has no timeout. An agent could generate infinite-loop code. We need a 5-minute hard limit that kills the subprocess and marks it as "failed execution" (not a fitness score — just a binary reject).

**Parallel fitness evaluation**: ASI-Evolve runs 4 workers per config. We could run 2-3 parallel seeds on the same candidate to verify robustness. One seed is not enough for a reliable score — we know from DISCOVERIES.md that seed sensitivity matters (e.g., LR=0.5 never triggers grokking, but LR=0.1 does).

### 2.2 Fitness Function Design

ASI-Evolve uses a scalar fitness from experiments. For our ByteDMD-optimized loop, fitness should be:

```
fitness = -bytedmd_score  # lower ByteDMD = better, so negate

penalty if:
    - accuracy < 90%: fitness = -infinity (reject, no ByteDMD to optimize)
    - timeout: fitness = -infinity
    - seed variance > 20%: fitness = -infinity (non-robust)
```

The fitness is the ByteDMD cost itself. We want to minimize data movement while maintaining >90% accuracy.

## 3. Parsing ByteDMD Output for Diagnostics

This is critical: **ByteDMD outputs a single integer** (total DMC cost). It provides no stack trace, no per-operation breakdown, no "which read was expensive" feedback. This is by design — it's a static metric, not a profiler.

What we CAN extract to diagnose *why* a candidate was penalized:

### 3.1 What ByteDMD Tells You

Given `bytedmd(function, (arg1, arg2)) == <integer>`, the cost decomposes as:

1. **Every read is priced**: C = ∑ ceil(√D(b)) where D(b) is each byte's depth from LRU stack top
2. **Writes are free**: Only reads cost anything
3. **Order matters**: `b + c` vs `c + b` may cost the same but produce different stack states for subsequent operations

### 3.2 What Our MemTracker Tells You (Harness-Accessible Diagnostics)

The harness tracks far more detail than raw ByteDMD:

| Diagnostic | Source | What It Reveals |
|---|---|---|
| `weighted_ard` | `tracker.summary()` | Which operations have the worst reuse distance |
| `dmc` | `tracker.summary()` | Total data movement complexity |
| `total_floats_accessed` | `tracker.summary()` | Pure volume of memory traffic |
| `operations` list | `tracker.operations` | Exact sequence of reads and writes |

### 3.3 Diagnostic Strategy — Reconstructing the ByteDMD Stack Trace

To figure out *why* a candidate's ByteDMD score is high:

**Step 1: Decompose by operation.** For each `tracker.read()` / `tracker.write()` call in the harness, record the byte count and the current maximum stack depth. The MemTracker already tracks `stack_distance` per access — this is the D(b) in the ByteDMD formula.

**Step 2: Compute per-operation ByteDMD contribution.** For each MemTracker operation, compute `ceil(√D(b))` for each byte read. This gives a per-line breakdown:

```
Read W1 (3,200 floats = 12,800 bytes): avg depth 3,200 → ceil(√3200) * 12,800 = 721,408
Read x (20 floats = 80 bytes): avg depth 80 → ceil(√80) * 80 = 720
Read h (200 floats = 800 bytes): avg depth 400 → ceil(√400) * 800 = 16,000
```

This is implementable with a simple wrapper around the existing MemTracker — no harness changes needed. A `ByteDMDTracker` subclass of MemTracker that additionally records per-operation cost.

**Step 3: Identify the top offender.** The single operation accounting for most ByteDMD cost is the "hot path." In our SGD baseline, this is always W1 reads (75% of ARD, and correspondingly likely ~75% of ByteDMD). Any algorithm proposing to avoid W1 reads must explain how it avoids reading the weight matrix during training.

**Step 4: Propose fix to the designer.** Feed the top-3 operations by ByteDMD cost back into the next generation's context: "Candidate X scored ByteDMD=Y. Top costs: Op1=40%, Op2=35%, Op3=15%. Fix the hot path." This is exactly what ASI-Evolve's Analyzer does — distill raw output into actionable feedback.

## 4. What Changes Are Needed to Our Infra

### 4.1 No Harness Modifications (LAB Rule #9)

None of the following touches `harness.py`, `tracker.py`, `cache_tracker.py`, `data.py`, or `config.py`.

### 4.2 New Orchestration Layer

A new module would sit between the agent prompt system and the harness:

```
src/orchestrator/
    loop.py          # Main ASI-Evolve loop: sample → design → execute → analyze
    sampler.py       # UCB1 / MAP-Elites / random sampling over log.jsonl
    analyzer.py      # Decompose ByteDMD score into per-operation breakdown
    quick_test.py    # Fast rejection: syntax check + 50-epoch smoke test
```

None of these modifies measurement code. The orchestrator is a consumer of the harness, not a modifier.

### 4.3 Timeout Wrapper

A simple subprocess wrapper with a 300-second hard kill:

```python
# Conceptual, not implementation
import subprocess
result = subprocess.run(cmd, timeout=300, capture_output=True)
if result.returncode == -9:  # killed by timeout
    fitness = -float('inf')
```

This is orchestration, not measurement.

### 4.4 Quick-Test Protocol

ASI-Evolve runs "lightweight quick tests for early rejection of flawed candidates." For our harness:

1. **Syntax check**: `python -m py_compile candidate.py` — instant
2. **Import check**: `python -c "import candidate"` — verify imports work
3. **Smoke test**: Run harness with `--n_bits 3 --k_sparse 3 --max_epochs 50` — should complete in <1s
4. **Full evaluation**: Only if smoke test passes, run the real config (n=20/k=3 or harder)

This rejects malformed code before wasting compute.

## 5. Sampling Policies for Parent Selection

ASI-Evolve uses UCB1 and MAP-Elites as pluggable sampling policies. These would directly apply to our `research/log.jsonl`:

### UCB1 over Experiment Log

For each prior experiment in log.jsonl with a fitness score:
```
UCB1(i) = mean_fitness(i) + 1.414 * sqrt(ln(total_rounds) / visits(i))
```
This balances exploiting high-fitness ancestors with exploring under-sampled ones. Currently, our agents just pick "the next unchecked TODO" — no fitness-guided parent selection.

### MAP-Elites Quality-Diversity

Partition the archive by behavioral features. For our case, natural feature dimensions:
- **Complexity**: lines of code in the proposed method
- **Diversity**: embedding distance to prior submissions (or just method family: SGD-family, GF2-family, Fourier-family)
- **Fitness**: negative ByteDMD score

This prevents the loop from over-exploiting one local optimum and encourages exploring genuinely different algorithmic approaches.

## 6. The End-to-End Loop — Complete Design

```
┌─────────────────────────────────────────────────┐
│  MEMORY/COGNITION (Kimi's domain)               │
│  DISCOVERIES.md + log.jsonl + UCB1 sampling     │
│  Retrieves: top-3 parents + 5 relevant findings │
└──────────────────┬──────────────────────────────┘
                   │ parent code + retrieved context
                   ▼
┌─────────────────────────────────────────────────┐
│  DESIGN (algorithms-claude's domain)            │
│  Agent generates Python code conditioned on:    │
│  - parent source code (diff-based mutation)     │
│  - top ByteDMD diagnostics from parent eval     │
│  - DISCOVERIES.md constraints (what's failed)   │
└──────────────────┬──────────────────────────────┘
                   │ candidate.py
                   ▼
┌─────────────────────────────────────────────────┐
│  QUICK TEST                                     │
│  py_compile → import check → n=3,k=3 smoke test │
│  Reject immediately if any step fails           │
└──────────────────┬──────────────────────────────┘
                   │ (pass)
                   ▼
┌─────────────────────────────────────────────────┐
│  EXPERIMENT (harness.py — LOCKED)               │
│  PYTHONPATH=src python3 harness.py              │
│  --method custom --n_bits 20 --k_sparse 3       │
│  Returns: accuracy, ARD, DMC, time,             │
│           total_floats, fitness=-ByteDMD        │
└──────────────────┬──────────────────────────────┘
                   │ metrics
                   ▼
┌─────────────────────────────────────────────────┐
│  ANALYZER                                       │
│  Decompose ByteDMD by operation                 │
│  Rank top-3 ops by cost (W1, h, out, etc.)      │
│  Generate: "Op1=40%, fix by ..."               │
└──────────────────┬──────────────────────────────┘
                   │ compact finding
                   ▼
┌─────────────────────────────────────────────────┐
│  STORE                                          │
│  Append to log.jsonl:                           │
│  {source_code, metrics, analysis, fitness}       │
│  Update UCB1 archive for next round             │
└─────────────────────────────────────────────────┘
```

Loop repeats until: max rounds reached, fitness plateaus, or accuracy drops below threshold.

## 7. Specific Recommendations for SutroYaro

1. **Adopt ByteDMD as the primary fitness score** (not ARD or raw DMC). The ByteDMD metric is byte-level and sub-linear (√depth penalty), making it harder to game than ARD's simple average. Candidates that reduce ByteDMD must genuinely improve spatial locality.

2. **Implement the Analyzer as a post-processing step**, not a harness modification. After harness.py returns its dict, the analyzer decomposes the MemTracker's operation log into per-operation ByteDMD contributions. This is pure analysis — no changes to measurement code.

3. **Use UCB1 sampling over log.jsonl** from day one. The simplest sampling policy that works. MAP-Elites is better but more complex. Start with UCB1, add MAP-Elites as a second policy.

4. **The quick-test protocol is the lowest-hanging fruit**. Adding py_compile + import check + 50-epoch smoke test before the full evaluation would save >80% of wall time on broken candidates. This alone makes a closed-loop system practical.

5. **Do not use the Gymnasium environment for ByteDMD evaluation**. The Gymnasium env (`SutroYaro/SparseParity-v0`) has a fixed budget and step-based rewards. It is designed for testing whether agents can *learn* ByteDMD-optimal behavior through RL. But for our evolutionary loop, we want **harness.py** as the fitness oracle — it returns a clean scalar (accuracy, ARD, DMC, time) from a single evaluation. The Gymnasium env is useful only for the RL bit of the loop, not for evaluating candidate algorithms in the evolution cycle.

6. **Feed ByteDMD diagnostics into the designer prompt**. The top-3 operations by cost should be appended to the next generation's context: "Candidate X scored ByteDMD=1,278,460. Top costs: Read W1=40%, Read x=15%, Read h=10%." This is the Analyzer's output feeding into the Researcher's input — the core ASI-Evolve feedback loop.