# DeepSeek Engram Offload — ByteDMD Verification

You are contributing to the **Sutro Group**, a research lab studying energy-efficient AI training. Your task is to verify whether the DeepSeek Engram paper's "<3% offload overhead" claim survives when priced under ByteDMD (byte-level data movement), rather than wall-time.

This is a **two-phase task**. Phase 1 is a paper-reading pass; Phase 2 is a microbenchmark. **Do Phase 1 first** — without the paper's actual `m:M` ratio and prefetch heuristic, Phase 2 models a strawman.

## Project context

Read these files first:

- `DISCOVERIES.md` — what's already known
- `LAB.md` — lab rules (important: do NOT modify measurement code: `tracker.py`, `cache_tracker.py`, `data.py`, `config.py`, `harness.py`, `fast.py`, `src/bytedmd/`)
- `docs/research/bytedmd.md` — ByteDMD metric spec
- `docs/tasks/011-engram-offload-bytedmd.md` — task spec and decision thresholds
- `findings/_template.md` — expected report format

Key context:

- Our metric: **ByteDMD** — byte-granularity data movement. Reading a value at stack depth `d` costs `ceil(sqrt(d))` per byte. Writes are free. Reference: https://github.com/cybertronai/ByteDMD
- The central suspicion: offloading to SSD means reads at stack depth ~`1e6`, which should cost `sqrt(1e6) = 1000` per byte — roughly 31× the cost of HBM reads at depth ~`1e3`. A 3% wall-time overhead claim is hard to reconcile with that unless the heuristic genuinely reduces *how many* deep reads happen (not just hides their latency via async prefetch).

## Phase 1 — Paper extraction

### Read the paper carefully

Primary source: https://deepseek.ai/blog/deepseek-engram-v4-architecture

Extract the following with quoted evidence from the paper where possible:

1. **Overhead type** — does the paper report wall-time, FLOPs, energy, or tokens/sec? The "3%" figure applies to which metric specifically?
2. **`m:M` ratio** — resident set size (in cache/HBM) vs offloaded set size. Can be tokens, KV entries, parameters — whatever the paper offloads. If not stated explicitly, infer from reported configurations.
3. **Lookup frequency `p`** — fraction of forward-pass lookups that hit the offloaded set.
4. **Prefetch heuristic** — random? LRU? model-guided prediction of hot entries? Learned routing? Copy-on-access? Describe the mechanism in enough detail to implement a reduced version.
5. **What "offload" physically means** — CPU RAM? NVMe SSD? Remote tier? All of the above?

### Write `docs/findings/engram-offload-paper-read.md`

Use this structure:

```markdown
# DeepSeek Engram — Paper Extraction

## Source
[URL, date accessed, paper version]

## Overhead claim
[Exact quote. What metric does "3%" refer to?]

## Offload configuration
- m (resident): [size, units, quoted source]
- M (offloaded): [size, units, quoted source]
- m:M ratio: [derived]
- p (lookup fraction): [quoted or inferred]
- Physical tier: [CPU RAM / SSD / other]

## Prefetch heuristic
[2-4 paragraphs describing the mechanism, with code-level detail if possible]

## Gaps
[What the paper does NOT say that Phase 2 will have to assume. Be explicit.]

## Implications for Phase 2
[What values to use for m, M, p, and which heuristic to model]
```

**Do NOT proceed to Phase 2 if:**
- The paper doesn't report concrete `m:M` or `p` (note gaps explicitly; flag for author follow-up)
- The "overhead" metric is ambiguous (could be any of wall-time / FLOPs / tokens-per-sec)

In that case, stop at Phase 1 and document the gaps.

## Phase 2 — ByteDMD microbenchmark

### Reduced model

Create `experiments/engram-offload/model.py`:
- Toy attention block: `d_model=64`, `seq_len=128`, 1 head
- KV memory bank split into two tiers:
  - **resident tier** (size `m`): lives at top of ByteDMD stack (depth ~100-1000)
  - **offloaded tier** (size `M`): forced to depth ~`1e6` via padding/unused writes
- Pure Python values so ByteDMD's tracer sees every read. No numpy in the inner loop (numpy escapes the tracer, see `docs/research/bytedmd.md`)

### Three variants, same compute

Create `experiments/engram-offload/run.py`:

1. **`resident`** — all KV entries resident (baseline). Every lookup is shallow.
2. **`offload_naive`** — `M` offloaded entries at depth `1e6`. Every lookup into the offloaded tier pays `sqrt(1e6) = 1000` per byte. No prefetching.
3. **`offload_prefetch`** — paper's heuristic from Phase 1. Predicted-hot entries promoted to the top of the stack each step, reducing how many reads actually pay the deep cost.

All three process the same `tokens × kv_bank` input and produce the same output (deterministic forward pass; only the memory layout differs).

### Measurement

For each variant:

```python
from bytedmd import bytedmd, traced_eval

cost = bytedmd(forward, (tokens, kv_bank))
trace, out = traced_eval(forward, (tokens, kv_bank))
```

Record:
- Total ByteDMD cost
- Cost per token
- Trace depth distribution (histogram of stack depths at each read)
- Wall-time with `time.perf_counter()` for the same forward pass

Save to `experiments/engram-offload/results.json`.

### Findings

Create `docs/findings/engram-offload-bytedmd.md` following `findings/_template.md`. Must include:

- **Hypothesis** (from the paper-read + your prior)
- **Config** (m, M, p, heuristic — reference Phase 1 extraction)
- **Results table** (all three variants: cost, cost-per-token, wall-time, ratios)
- **Classification** (WIN / LOSS / INVALID / INCONCLUSIVE / BASELINE) per the decision thresholds in `docs/tasks/011-engram-offload-bytedmd.md`
- **Analysis** — what worked, what didn't, the surprise
- **Impact on DISCOVERIES.md** (if any) — does this add to the ByteDMD-vs-wall-time story?

## Decision thresholds

From `docs/tasks/011-engram-offload-bytedmd.md`:

| ByteDMD overhead (prefetch vs resident) | Classification | Action |
|---|---|---|
| **2-5×** | WIN for DeepSeek — claim is real | Positive case study; their heuristic genuinely works |
| **>20×** | LOSS — wall-time gaming | Canonical case study for why SutroYaro uses ByteDMD |
| **5-20×** | INCONCLUSIVE | Re-run with larger `M` and varying `p`; bandwidth-hiding advantage should shrink as working set grows |

## Rules

- **DO NOT** modify measurement code: `src/bytedmd/`, `src/harness.py`, `tracker.py`, `cache_tracker.py`, `data.py`, `config.py`, `fast.py`
- **DO NOT** modify `DISCOVERIES.md`, `LAB.md`, `CLAUDE.md`, `CODEX.md`, or any other project configuration. Updates to `DISCOVERIES.md` are added separately via PR after findings review.
- Pure Python in the inner loop. Numpy escapes ByteDMD; see `docs/research/bytedmd.md`.
- All findings go under `docs/findings/` (the mkdocs-published location). Do NOT create files in the root-level `findings/` directory.
- Phase 1 first. Do not start Phase 2 until the paper extraction is written.

## Out of scope

- Re-training or re-implementing the full DeepSeek model. We're testing the metric on a reduced analogue.
- Modeling PCIe/NVMe energy directly. ByteDMD's `sqrt(depth)` is the proxy.
- Modifying the ByteDMD tracer to "handle" offloading more cleverly. The metric is the spec; we're measuring against it, not tuning it.
