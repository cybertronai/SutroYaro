# AGENTS.md

This project was built with Claude Code (Anthropic's CLI agent).

## How AI agents were used

**Phase 1** (16 experiments): Single Claude Code sessions running experiments sequentially, each following the template in `src/sparse_parity/experiments/_template.py`.

**Phase 2** (17 experiments): 17 independent Claude Code agents dispatched in parallel, each implementing a different algorithmic approach to sparse parity. Each agent received the approach description, shared module APIs, three test configs, and a findings template. All 17 completed successfully, producing code, results, and findings.

**Survey**: A single agent wrote the Practitioner's Field Guide (`docs/research/survey.md`) synthesizing all 33 experiments, with spec compliance and code quality review passes.

## Key files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions loaded at session start |
| `LAB.md` | Experiment protocol (one hypothesis, baseline, commit discipline) |
| `DISCOVERIES.md` | Accumulated knowledge base -- every agent reads this before starting |
| `docs/research/survey.md` | Full methodology in Section 7 (agentic loop, parallel dispatch, prompting strategies) |
| `docs/tooling/anti-slop-guide.md` | Writing rules applied to all agent-generated prose |

## What worked

- **DISCOVERIES.md as shared memory**: No agent repeated known-bad configurations (LR=0.5, GrokFast).
- **"It's OK to fail" prompts**: Agents produced better negative results when told to document failure reasons rather than forced to succeed.
- **Strict output format**: All 33 experiments are directly comparable because every agent used the same structure.
- **Literature-first prompting**: Giving agents theoretical context ("parity is linear over GF(2)") led to correct implementations.

## What didn't work

- **Pyright import warnings**: Every agent triggered false "Import could not be resolved" errors (PYTHONPATH is set at runtime).
- **Data generation bugs**: Two agents (Hebbian, Binary Weights) used different seeds for train/test. Fixed during execution.
- **Pebble Game exhaustive search**: One agent took 38 minutes exploring 5,758 topological orderings. A smarter search heuristic would have been faster.
