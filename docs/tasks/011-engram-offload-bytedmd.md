# Task 11: DeepSeek Engram Offload — ByteDMD Verification

**Priority**: MEDIUM
**Status**: OPEN
**Agent**: unassigned
**Source**: Issue #77 (observation by Andy via Qwen, experiment plan by Yad)

## Context

The DeepSeek Engram paper ("Conditional Memory via Scalable Lookup") claims that 100B parameters can be offloaded to CPU/SSD with <3% inference overhead. From a ByteDMD perspective this claim is suspicious:

- SSD→GPU bandwidth is 20-60× slower than GPU HBM (~16GB/s vs ~1-3TB/s)
- Under ByteDMD, SSD reads live at stack depth ~millions vs ~hundreds for HBM
- Per-byte cost ratio: `ceil(sqrt(1e6)) / ceil(sqrt(1e3)) ≈ 31×`
- If the "3%" is wall-time (async prefetching hiding latency) rather than energy, ByteDMD should expose that

**Relevance to SutroYaro**: this is exactly the "wall-time ≠ energy" confusion that motivates the ByteDMD metric. Either outcome of this task produces valuable output:

- **Claim validated** → heuristic genuinely reduces deep reads; document the pattern as a positive case study
- **Claim is wall-time gaming** → document as the canonical case study for why SutroYaro uses ByteDMD and not throughput

Paper: https://deepseek.ai/blog/deepseek-engram-v4-architecture

## Tasks

### Phase 1 — Paper extraction (prerequisite)

- [ ] Read https://deepseek.ai/blog/deepseek-engram-v4-architecture carefully
- [ ] Extract whether "overhead" is wall-time, FLOPs, or energy
- [ ] Extract concrete `m:M` ratio (resident : offloaded set sizes)
- [ ] Extract lookup frequency `p` (fraction of tokens that hit the offloaded set)
- [ ] Document the prefetch heuristic (predicted-hot entries? random? LRU? model-guided?)
- [ ] Write to `docs/findings/engram-offload-paper-read.md`

Without these specifics, Phase 2 models a generic offload pattern, not Engram specifically.

### Phase 2 — ByteDMD microbenchmark

- [ ] Create `experiments/engram-offload/model.py` — a toy attention block (d_model=64, seq_len=128) with a KV memory bank split into resident vs offloaded tiers. Pure Python so ByteDMD traces all reads.
- [ ] Create `experiments/engram-offload/run.py` with three variants, identical compute:
  - `resident`: all weights at small stack depth (baseline)
  - `offload_naive`: offloaded bank at depth ~`1e6`, every lookup pays `ceil(sqrt(1e6))=1000` per byte
  - `offload_prefetch`: paper's heuristic (promote predicted-hot entries to top each step)
- [ ] Run `bytedmd(forward, (tokens, kv_bank))` for each variant. Record cost, cost-per-token, trace distribution
- [ ] Wall-time control: time the same three on CPU with `time.perf_counter()` to reproduce the "<3%" wall-time claim locally
- [ ] Write findings to `docs/findings/engram-offload-bytedmd.md` following `findings/_template.md`

### Decision thresholds

- **Go** (claim is real): ByteDMD overhead of `offload_prefetch` vs `resident` is **2-5×**. Heuristic genuinely reduces deep reads.
- **No-go** (wall-time gaming): ByteDMD overhead **>20×** while wall-time overhead stays <3%. Write up as case study.
- **Ambiguous**: **5-20×**. Re-run with larger `M` and varying `p`; bandwidth-hiding advantage should shrink as working set grows.

## Out of scope

- Re-training the 100B model. Testing the metric, not the paper.
- PCIe energy models. ByteDMD's `sqrt(depth)` is the proxy.

## References

- Agent prompt: [docs/agent-prompts/engram-offload.md](../agent-prompts/engram-offload.md)
- Paper: https://deepseek.ai/blog/deepseek-engram-v4-architecture
- ByteDMD metric: [docs/research/bytedmd.md](../research/bytedmd.md)
- ByteDMD repo: https://github.com/cybertronai/ByteDMD
- Origin issue: [#77](https://github.com/cybertronai/SutroYaro/issues/77)
- Pattern precedent: Task 9 (Muon review) — [009-muon-review.md](009-muon-review.md)
