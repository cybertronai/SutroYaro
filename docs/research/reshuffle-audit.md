# Reshuffle Audit (Issue #96, Phase 1)

**Date:** 2026-05-20
**Status:** Phase 1 audit, written after the fact.

## Why this doc exists

Issue #96 specified Phase 1 as: write this audit before migrating anything. The migration ran first instead. Seth's PR #40 (sparse-parity-challenge, merged May 12) copied the experiment layer into that repo; PR #97 (SutroYaro, merged May 14) deleted 144 files here. No per-asset triage was recorded. This doc back-fills that record and triages what is left, so the rest of the strip is a known, safe operation.

## What the migration already did

- **PR #40, into sparse-parity-challenge:** added `src/sparse_parity/` (model, train, metrics, 44 experiments, eval adapters, challenges, reference), `results/`, `tests/`, `checks/`. The core tracker files were already present there and are ahead of SutroYaro's copies.
- **PR #97, in SutroYaro:** deleted the same set (144 files), then restored `tests/test_bytedmd.py` and `tests/test_bytedmd_gotchas.py`.

The migration was a partial strip. The residue below was left behind.

## Verification method

Compared `git ls-files` in SutroYaro against the `sparse-parity-challenge` `main` tree (`gh api .../git/trees/main?recursive=1`) on 2026-05-20. "In SPC" means the path exists there. Byte-level content equality was not checked.

## Residue triage

| Path | What it is | In SPC? | Action |
|------|-----------|---------|--------|
| `src/sparse_parity/__init__.py` | package init | yes | delete |
| `src/sparse_parity/config.py` | locked config | yes | delete |
| `src/sparse_parity/data.py` | locked data gen | yes | delete |
| `src/sparse_parity/fast.py` | numpy SGD | yes | delete |
| `src/sparse_parity/lru_tracker.py` | LRU stack tracker | yes (SPC ahead) | delete |
| `src/sparse_parity/tracked_numpy.py` | TrackedArray | yes (SPC ahead) | delete |
| `src/sparse_parity/tracker.py` | MemTracker | yes | delete |
| `src/sparse_parity/experiments/exp_km_sat_hybrid.py` | experiment | yes | delete |
| `src/sparse_parity/experiments/exp_bytedmd_floor_gap.py` | experiment | **no** | migrate to SPC first (blocker) |
| `src/harness.py` | locked eval harness | yes | delete with `src/sparse_parity/` |
| `src/plot_dmc.py` | DMC plotting script | **no** | migrate to SPC, or archive |
| `results/exp_km_sat_hybrid/` | result json | yes | delete |
| `results/exp_bytedmd_floor_gap/` | result json | **no** | migrate to SPC |
| `results/exp1_20260303_221628/` | result json | **no** | migrate to SPC, or archive |
| `results/exp4_grokfast_20260303_222209/` | result json | **no** | migrate to SPC, or archive |
| `src/bytedmd/` (2 files) | vendored ByteDMD tracer | yes | keep as snapshot; final home is cybertronai/ByteDMD |
| `tests/test_bytedmd.py`, `test_bytedmd_gotchas.py` | tests for `src/bytedmd/` | yes | keep while `src/bytedmd/` stays |

## The blocker

`src/sparse_parity/` cannot be cleanly removed with SutroYaro-only changes. `experiments/exp_bytedmd_floor_gap.py` is not in sparse-parity-challenge and it imports the tracker modules. Deleting the trackers while keeping that file would break it.

So the full strip needs one small cross-repo PR first: move `exp_bytedmd_floor_gap.py`, `plot_dmc.py`, and the three SutroYaro-only `results/` directories into sparse-parity-challenge. After that lands, all of `src/sparse_parity/` plus `src/harness.py` can be deleted here in a single commit. Before migrating `exp_bytedmd_floor_gap.py`, check whether the floor-gap survey is already covered by an existing SPC experiment to avoid duplication.

## Internal dependency check

Within SutroYaro, `src/sparse_parity/` is imported only by `src/harness.py`. `src/bytedmd/` is imported only by `tests/test_bytedmd*.py`. Nothing else in the repo imports either. The skills and top-level docs (`AGENT.md`, `LAB.md`, `AGENT_EVAL.md`, `run-experiment`, `sutro-context`) reference them in prose only. Those references are updated in Phase C, not here.

## Target end-state

After the strip, `src/` holds only lab-memory and dispatcher infrastructure:

- `src/telegram/` (Telegram sync)
- `src/sync_google_docs.py`, `src/docs_config.json` (Google Docs sync)
- `src/bytedmd/` (frozen metric snapshot, until it moves to cybertronai/ByteDMD)

## What stays (the scoped role, not residue)

`DISCOVERIES.md`, `docs/` (tasks, catchups, findings, research, related-repos), `.claude/` (skills, hooks, rules), `bin/`, the sync infrastructure, `mkdocs.yml`. These are the records, index, and dispatcher. They get generalized off sparse parity in Phases B and C, not removed.

## Next steps

1. **Decision needed:** allow one small cross-repo PR to sparse-parity-challenge (unblocks the full strip), or freeze `src/sparse_parity/` in place for now and finish the strip later.
2. **Phase B:** build the four-challenge index so SutroYaro is useful for matmul, grid sparse parity, and wikitext, not only the original sparse parity.
3. **Phase C:** generalize `CLAUDE.md`, `DISCOVERIES.md`, and the skills off sparse parity.
