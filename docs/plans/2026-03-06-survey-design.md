# Survey Design: Practitioner's Field Guide to Sparse Parity

**Date**: 2026-03-06
**Status**: Approved
**Author**: Yad Konrad (with Claude Code)

## Purpose

A standalone survey covering all 33 experiments (16 original + 17 new) on the sparse parity challenge. Serves two audiences: Sutro Group members picking methods for new problems, and future researchers reproducing or extending this work.

## Decisions

- **Format**: Practitioner's field guide, not academic paper
- **Structure**: TL;DR up front, narrative arc in the middle, methodology at the end
- **Narrative**: Two phases. Phase 1 (incremental SGD improvements, hit a wall). Phase 2 (broad search across 17 alternative approaches).
- **Decision framework**: Parity-specific quick reference table + generalized principles for new problems
- **Meta-process**: Full methodology including agentic loop, parallel dispatch, prompts, failure modes, cost
- **Deliverables**: MkDocs page + standalone copy + nav updates + DISCOVERIES.md update
- **Writing**: Anti-slop guide applied throughout. No em dashes, no AI vocabulary, no throat-clearing.

## Sections

### 1. TL;DR (half page)
Ranked table of all 33 experiments: method, accuracy, time, ARD, verdict. Three bullets: best for speed, best for energy, best for generality.

### 2. The Problem (1 page)
Sparse parity definition, ARD definition, cache model, energy proxy (register 5pJ, L1 20pJ, L2 100pJ, HBM 640pJ), constraints (under 2s, above 90% accuracy). No group history.

### 3. Phase 1: Incremental Improvements (2-3 pages)
16 original experiments. Arc: broken baseline → fix hyperparams → grokking → try to reduce ARD within SGD → cache simulator reveals raw ARD is misleading → hit the W1 wall (75% of reads) → pivot to algorithm changes (curriculum, Sign SGD) → pivot to blank slate (Fourier, random search).

### 4. Phase 2: Broad Search (3-4 pages)
17 new experiments organized by taxonomy:
- Algebraic/Exact (GF(2), KM, SMT) — the winners
- Information-Theoretic (MI, LASSO, MDL, Random Projections) — all solve it, none beat Fourier meaningfully
- Local Learning Rules (Hebbian, PC, EP, TP) — all failed, structural impossibility
- Hardware-Aware (Tiled W1, Pebble Game, Binary Weights) — mixed
- Alternative Framings (GP, RL, Decision Trees) — mixed

### 5. Results Leaderboard (1 page)
Three ranked tables: by speed, by energy proxy, by generality.

### 6. Decision Framework (2 pages)
- Parity-specific flowchart (mermaid diagram)
- Generalized principles for new problems (numbered list, each referencing source experiments)

### 7. The AI Research Process (3-4 pages)
- 7a. Agentic loop structure (template, shared modules, DISCOVERIES.md)
- 7b. Parallel dispatch (17 agents, prompt structure, failure modes, cost)
- 7c. What worked and didn't (prompting strategies, anti-slop guide, specific prompts)

### 8. Appendix
Links to all 33 findings pages. No duplicated content.

## Deliverables

1. `docs/research/survey.md` — full document (MkDocs page)
2. `survey.md` in repo root — standalone copy
3. Update `mkdocs.yml` nav: add survey + 17 new findings pages
4. Update `docs/research/index.md` to link to survey
5. Update `DISCOVERIES.md` with 17 new experiment results
