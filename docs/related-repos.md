# Related repos in the `cybertronai` GitHub org

A single source of truth for the repositories adjacent to SutroYaro. Updated 2026-05-09.

## Active research front (May 2026)

These repos are where current work lives. If you're picking up the project, start here.

| Repo | What it is | Latest activity |
|---|---|---|
| **[`SutroYaro`](https://github.com/cybertronai/SutroYaro)** | This repo. Lab notebook for Phase 1+2 work, autonomous-research infrastructure, public docs site, sparse-parity scoreboard. | Active |
| **[`ByteDMD`](https://github.com/cybertronai/ByteDMD)** | The primary cost metric (data-movement complexity, byte-granularity LRU stack). Yaroslav's **active research front** lives at [`experiments/grid`](https://github.com/cybertronai/ByteDMD/tree/dev/experiments/grid) — self-contained experiments. | Active |
| **[`simplified-dally-model`](https://github.com/cybertronai/simplified-dally-model)** | Yaroslav's 2D Manhattan-grid cost model. Single-processor, explicit communication cost. Goal: minimum instruction set that captures the relevant cost dynamics. Will eventually replace 1D Dally pJ numbers in the SutroYaro eval harness. | Active (2026-05-08) |
| **[`sutro`](https://github.com/cybertronai/sutro)** | Main code repo. Owns `sparse_parity_benchmark.py` and the original solver implementations. | Reference |
| **[`sutro-problems`](https://github.com/cybertronai/sutro-problems)** | Small reproducible problems collection. Has the matmul energy-metric work. Andy Zhang is now an owner (Telegram, 2026-05-06) and active on this. | Active |
| **[`sparse-parity-challenge`](https://github.com/cybertronai/sparse-parity-challenge)** | Submission pipeline: submitters open a GitHub Issue with a `solve()` function; CI auto-evaluates under ByteDMD and posts the score. | Active |

## Companion baseline catalogs (shipped May 2026)

Reproducible-baseline implementations of the synthetic learning problems from two distinct paper lineages. Both shipped via the agent-team build pattern; see each repo's `BUILD_NOTES.md` for the session-level details.

| Repo | Paper lineage | Stubs | Site |
|---|---|---:|---|
| **[`hinton-problems`](https://github.com/cybertronai/hinton-problems)** | Hinton 1981–2022 — **representational** toy tasks (4-2-4 encoder, family trees, shifter, capsules, Forward-Forward) | 53 v1 + 2 add-ons | [site](https://cybertronai.github.io/hinton-problems/) |
| **[`schmidhuber-problems`](https://github.com/cybertronai/schmidhuber-problems)** | Schmidhuber 1989–2025 — **algorithmic** capability (long-time-lag indexing, key-value binding, Levin/OOPS search, controller+model+curiosity, World Models) | 50 v1 + 8 v1.5 | [site](https://cybertronai.github.io/schmidhuber-problems/) |

Together: the **representational + algorithmic** baseline pair. Both are pure numpy + matplotlib, laptop-runnable, with paper-comparison metrics per stub. The follow-up `v2` work is to instrument these baselines with ByteDMD and compare data-movement cost across algorithm families.

Tracking issues:
- [hinton-problems #45 (v2 ByteDMD)](https://github.com/cybertronai/hinton-problems/issues/45) and [#46 (v1.5 paper-scale)](https://github.com/cybertronai/hinton-problems/issues/46)
- [schmidhuber-problems #17 (v2 ByteDMD)](https://github.com/cybertronai/schmidhuber-problems/issues/17) and [#18 (v1.5 paper-scale + original-simulator)](https://github.com/cybertronai/schmidhuber-problems/issues/18)

## Adjacent / external

| Repo | Connection |
|---|---|
| [`adotzh/SutroAna`](https://github.com/adotzh/SutroAna) (not in cybertronai org) | Anastasia Zhiboedova's auto-research-loop framework. Presented at meeting #16 (04 May 26). Independent of SutroYaro but solving overlapping problems. |

## Older `cybertronai` repos (reference, not active for this work)

The org has a long history. These are not currently part of the energy-efficient-training thread but are useful background:

- [`scaling-laws`](https://github.com/cybertronai/scaling-laws), [`autograd-hacks`](https://github.com/cybertronai/autograd-hacks), [`autograd-lib`](https://github.com/cybertronai/autograd-lib) — Yaroslav's instrumentation tools
- [`pytorch-sso`](https://github.com/cybertronai/pytorch-sso), [`pytorch-lamb`](https://github.com/cybertronai/pytorch-lamb), [`pytorch-fd`](https://github.com/cybertronai/pytorch-fd) — second-order methods, LAMB, fluctuation-dissipation
- [`gradient-checkpointing`](https://github.com/cybertronai/gradient-checkpointing) — memory-efficient training
- [`transformer-xl`](https://github.com/cybertronai/transformer-xl), [`imagenet18`](https://github.com/cybertronai/imagenet18), [`Megatron-LM`](https://github.com/cybertronai/Megatron-LM), [`bflm`](https://github.com/cybertronai/bflm) — training-at-scale work
- [`ncluster`](https://github.com/cybertronai/ncluster), [`pytorch-aws`](https://github.com/cybertronai/pytorch-aws), [`aws-network-benchmarks`](https://github.com/cybertronai/aws-network-benchmarks), [`autotune`](https://github.com/cybertronai/autotune) — AWS / training-infra tools

## How they connect (conceptual map)

```
                          ┌─────────────────────────┐
                          │     The big question:   │
                          │  energy-efficient AI    │
                          │      training           │
                          └────────────┬────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
        ┌─────────▼──────────┐               ┌──────────────▼─────────────┐
        │  Cost / metric     │               │  Benchmark problems        │
        │                    │               │                            │
        │  ByteDMD          ─┼───────────────┼─►  sparse-parity-          │
        │  simplified-      ─┤               │     challenge              │
        │   dally-model      │               │                            │
        └────────────────────┘               │  hinton-problems  ──┐      │
                  │                          │  schmidhuber-       │      │
                  │                          │   problems          │      │
                  │                          │                     │      │
                  │                          │  sutro-problems   ◄─┘      │
                  │                          │   (matmul, etc.)           │
                  │                          └────────────────────────────┘
                  │                                       │
                  └───────────────────┬───────────────────┘
                                      │
                          ┌───────────▼────────────┐
                          │      SutroYaro         │
                          │  (this repo)           │
                          │                        │
                          │  Lab notebook +        │
                          │  scoreboard +          │
                          │  autonomous research   │
                          │  infrastructure +      │
                          │  public docs site      │
                          └────────────────────────┘
```

`ByteDMD` and `simplified-dally-model` define the cost. `sparse-parity-challenge`, `hinton-problems`, `schmidhuber-problems`, `sutro-problems` provide the problems. `SutroYaro` is the lab notebook that consumes both: it tracks experiments, records what's proven (`DISCOVERIES.md`), runs autonomous research loops, and surfaces the public site.

## Updating this doc

Maintained by hand for now. Bump alongside the [active-threads catch-up](research/active-threads-2026-05-09.md) when:

- A new repo is added to the `cybertronai` org
- Yaroslav signals "this older repo is now active again"
- The conceptual map changes (e.g., `simplified-dally-model` becomes the primary metric and 1D ByteDMD goes to reference)

The `gh repo list cybertronai --limit 50` command is the authoritative source of what exists; this doc is the curated narrative.
