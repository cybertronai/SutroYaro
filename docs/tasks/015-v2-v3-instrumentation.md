# Task 15: V2 / V3 instrumentation of Hinton + Schmidhuber stubs

**Priority**: HIGH
**Status**: IN PROGRESS
**Source**: Telegram chat-yad, 2026-05-09 to 2026-05-12 (Yaroslav's v2/v3 staging clarification, Andy + Seth's first comparator pairs, the IR-expansion question)

## Context

After the hinton-problems (53 stubs) and schmidhuber-problems (58 stubs) catalogs shipped, the next step is measuring them under data-movement metrics. Yaroslav's staging (2026-05-11):

- **V2** = the subset of stubs feasible to instrument with **ByteDMD** (Python-level LRU stack tracer). Output: per-stub ByteDMD cost + reuse-distance distribution.
- **V3** = the further-restricted subset feasible under the **simplified Bill Dally** model (the 2D-grid v0 IR used in `sutro-problems/matmul`). Output: per-stub read-distance histogram in the matmul `doc/access_distance/` format.
- Survivors of both filters become the **candidate pool for the next hill-climbing competition** (matmul-style leaderboard).

This is a sideways move on the [3-axis cube](../research/active-threads-2026-05-09.md): stay on the Hinton/Schmidhuber problem rung, climb one rung on the metric axis (ByteDMD → simplified Bill Dally).

**Relevance to SutroYaro**: the code lands in the per-purpose repos (hinton-problems, schmidhuber-problems, sutro-problems), not here. SutroYaro's job is to be the cross-repo index for the effort — that's what this doc is.

## Where the work lives

### V2 (ByteDMD instrumentation)

| Repo / location | State |
|---|---|
| `hinton-problems/v2-bytedmd/` | Live. Framework by Andy Zhang (zh4ngx). |
| `hinton-problems` PR #59 | **MERGED** 2026-05-10. encoder-8-3-8 backprop vs Boltzmann CD-1/CD-5. Backprop wins 49–135x. |
| `hinton-problems` PR #60 | **OPEN**. bars-rbm (CD-1) vs bars (wake-sleep Helmholtz). Seth Stafford (SethTS). CD-1 wins ~20x. |
| `schmidhuber-problems` | No `v2-bytedmd/` dir yet. The cleanest comparator candidate is `linear-transformers-fwp` vs `fast-weights-key-value` (Schmidhuber proved mathematically equivalent; would confirm whether the ByteDMD numbers also match). |
| Tracking issues | [hinton-problems#45](https://github.com/cybertronai/hinton-problems/issues/45), [schmidhuber-problems#17](https://github.com/cybertronai/schmidhuber-problems/issues/17) |

### V3 (simplified Bill Dally / matmul-style IR)

| Repo / location | State |
|---|---|
| `sutro-problems/matmul/` | The template. `doc/access_distance/` is the histogram + CDF format to follow. |
| `sutro-problems` `dev` branch, `wip-boltzmann-shifter/` | Partial port of Hinton's shifter into the matmul-style submission framework. |
| `sutro-problems` PR #18 | **OPEN**. symmetry (6-bit palindrome) as a third sutro-problems challenge. Seth Stafford. Surfaces the IR-expansion problem: optimal baseline uses `cmp eq`, not in the v0 op set. |
| Andy's hinton→dally agent | Hit the IR roadblock (only IR-trackable algorithms work). Not PR'd. Paused pending the IR-expansion decision. |
| `simplified-dally-model` | Yaroslav's scoring engine. He is still iterating on the minimum instruction set. |

## Open question: does the v0 IR need more ops?

Raised by Andy's roadblock and Seth's symmetry PR. Yaroslav (2026-05-11):

> If agents can figure out a way to solve the toy problems presented by Hinton or Schmidhuber by only utilizing the IR, [good]. If they can't then we should probably increase it by adding ops.

Concrete evidence: PR #18's symmetry baseline (cost 20) uses `cmp eq` to test patterns rather than evaluating the polynomial form, and `cmp eq` is not in the v0 set (`add`, `sub`, `mul`, `copy`).

This decision is tracked separately as an issue on `simplified-dally-model` (or `sutro-problems`) — see the link once filed. It blocks the V3 subset definition.

## Ownership (current)

- **Andy Zhang (zh4ngx)** — V2 encoder framework (done), hinton→dally V3 (blocked on IR)
- **Seth Stafford (SethTS)** — V2 bars pair (PR #60), symmetry V3 challenge (PR #18)
- **Yad** — Schmidhuber V2 (the `v2-bytedmd/` dir that doesn't exist yet) is the open gap; also the cross-repo coordination (this doc)
- **Yaroslav** — IR-expansion decision; will look at V3 seriously after the AI Council talk (2026-05-12)

## Next actions

- [ ] Review + merge `hinton-problems` PR #60 (bars pair)
- [ ] Review + merge `sutro-problems` PR #18 (symmetry challenge)
- [ ] File the IR-expansion issue on `simplified-dally-model`, linking PR #18's `cmp eq` finding
- [ ] Start `schmidhuber-problems/v2-bytedmd/` cloning the hinton-problems pattern; first pair = `linear-transformers-fwp` vs `fast-weights-key-value`
- [ ] Once Yaroslav decides the IR scope, unblock Andy's hinton→dally V3 work

## Acceptance

- A V2 `v2-bytedmd/` dir exists in both hinton-problems and schmidhuber-problems with at least 2 comparator pairs each
- The IR-expansion question has a public decision (either "v0 is enough" or "here is the minimal expanded set")
- At least one V3 read-distance histogram exists for a Hinton or Schmidhuber stub, in the matmul `doc/access_distance/` format
- This doc stays current as the cross-repo index for the effort

## Dependencies

- V3 subset definition is blocked on the IR-expansion decision
- The hill-climbing competition (the eventual payoff) is blocked on having a V3 candidate pool
