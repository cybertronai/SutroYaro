# Task 14: Sparse-parity over time ("morse code" framing)

**Priority**: MEDIUM
**Status**: OPEN
**Agent**: unassigned
**Source**: Telegram general/chat-yaroslav, 2026-05-08T01:01:24Z — Andy Zhang: *"yes! love it. i was looking at sparse parity over time (like morse code)"*

## Context

The current sparse-parity benchmark presents the n-bit input as a single fixed-length vector. Andy is exploring a **temporal / streaming framing**: the n bits arrive one per timestep (like morse code), and the network must compute the parity at the end of the stream — closer to how RNNs / LSTMs / streaming filters consume input in practice.

This is a **reframe, not a different problem**: the underlying boolean function (parity over k secret indices among n) is the same. But the cost profile changes:
- **Same task, different memory access pattern**: the network sees one bit per step, must integrate them across time
- **Surfaces recurrent solutions**: solutions that work on the static version (KM influence estimation, GF(2) Gaussian elimination, RS over weights) may not naturally extend; LSTM / recurrent solutions get a fairer comparison
- **Connects to the schmidhuber-problems lineage**: the rs-parity stub there does exactly N-bit sequence parity by random weight guessing on a recurrent net; this task brings that framing into SutroYaro's primary benchmark

**Relevance to SutroYaro**: SutroYaro's eval-environment + harness are built around the static framing. Adding a streaming variant is a parallel challenge (call it `sparse-parity-temporal`) under the same scoring metric (DMC).

References:
- Andy's hint: morse-code analogy (bits arrive over time)
- schmidhuber-problems `rs-parity` stub: https://github.com/cybertronai/schmidhuber-problems/tree/main/rs-parity (random-weight-guessing, sub-second wallclock at N=50)
- LSTM canonical battery (`adding-problem`, `temporal-order-3bit`): same "T timesteps, decision at end" pattern

## Tasks

### Phase 1 — Define the variant

- [ ] Specify the streaming setup: input shape `(T,)` of ±1, output is parity over the k secret indices, decision at step T
- [ ] Pick T values: maybe 10, 20, 50, 100 (mirroring the rs-parity stub's range)
- [ ] Decide: are the k secret indices fixed across the stream (constant secret) or rotate (harder)?
- [ ] Document at `docs/research/sparse-parity-temporal-spec.md`

### Phase 2 — Reference solvers

Three reference solvers, each of a different family, scored under DMC:

- [ ] **Recurrent baseline**: vanilla tanh-RNN, BPTT, hidden=8. Should fail at long T (vanishing gradient).
- [ ] **LSTM baseline**: forget-gate LSTM, BPTT, hidden=8. Should solve longer T cleanly.
- [ ] **Random-weight-guess baseline**: port the schmidhuber-problems `rs-parity` stub here as-is. Algorithmically simplest; surprisingly competitive on the static benchmark.

For each, record `(T, accuracy, DMC, wallclock)`.

### Phase 3 — Sub-challenge or full challenge?

This is a question for Yad/Yaroslav, not a PR decision the agent should make:

- **Option A — sub-challenge of sparse-parity**: extend the existing challenge with a `--temporal` flag; submitters opt in.
- **Option B — separate challenge**: file a new repo `cybertronai/sparse-parity-temporal-challenge` with its own submission pipeline (mirroring `cybertronai/sparse-parity-challenge`).

Recommendation: do A first (cheap), promote to B only if it gets traction.

## Acceptance

- A spec doc that any submitter can read and code against
- Three reference solvers committed under `src/sparse_parity/experiments/` (or wherever streaming variants will live)
- A first DMC ranking on temporal sparse-parity

## Dependencies

- Andy's framing is exploratory; sync with him before scaffolding to avoid duplicating his work
- Should be aware of the schmidhuber-problems rs-parity / temporal-order stubs — don't reimplement what's already there in pure numpy
