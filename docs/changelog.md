# Changelog

All notable changes to this research workspace.

## [Unreleased]

### Planned
- Experiment 2: Weight decay sweep
- Experiment 3: Sign SGD (Kou et al. 2024)
- Scale to n=100 or k=5 where energy-efficient algorithms become necessary
- ARD measurement on the winning config (Exp 1 hyperparams)
- Technical Sprint 2: Forward-Forward on harder instances

---

## [0.5.0] - 2026-03-04

### Added
- **Exp 1: Fix Hyperparams** — 99% accuracy on 20-bit sparse parity (k=3)
    - LR 0.5→0.1, batch_size 1→32, n_train 200→500
    - Classic grokking pattern: phase transition at epoch 52
- **Exp 4: GrokFast** — baseline SGD hits 100% in 5 epochs (22.7s)
    - GrokFast counterproductive when hyperparams are correct
- Literature review: 6 key papers on sparse parity learning
- Research plan for autonomous experiment cycle
- Results organized into per-run directories with auto-generated index

---

## [0.4.0] - 2026-03-03

### Added
- Complete sparse parity pipeline (all 5 phases)
    - 3 training variants: standard backprop, fused, per-layer
    - MemTracker for ARD measurement
    - JSON + markdown + plot output
- 20/20 tests passing
- Per-run results directories with index lookup

---

## [0.3.0] - 2026-03-03

### Added
- MkDocs site with Material theme
- Mermaid diagrams in context and findings
- Changelog tracking

---

## [0.2.0] - 2026-03-03

### Added
- `src/sync_google_docs.py` -- standalone script to sync Google Docs to markdown
- Auto-extracted references (`docs/references_auto.md`)
- Expanded homework archive covering all 7 meetings

---

## [0.1.0] - 2026-03-03

### Added
- Initial research environment setup
- Converted 3 Google Docs to local markdown:
    - Challenge #1: Sparse Parity (spec)
    - Sutro Group Main (meeting index)
    - Yaroslav Technical Sprint 1 (sprint log)
- Extracted 30+ hyperlinks into `docs/references.md`
- Directory structure: docs, findings, plans, research, src
- `CLAUDE.md` with project context for AI assistants
- `CONTEXT.md` with project background and timeline
- Sprint 1 findings documented
- Sprint 2 plan drafted
- Meeting notes index with all external links
- Lectures and homework directories
- `.gitignore` for Python/Jupyter/IDE files
