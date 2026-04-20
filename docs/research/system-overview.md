# SutroYaro: system overview

A structured workspace where coding agents run energy-efficiency research. You point a coding agent at the repo, it reads the context files, runs experiments, and accumulates findings. Multiple people do this independently. The results merge through PRs.

This page describes how the pieces connect. Each piece has its own docs page. This is the map.

## What the system does

The Sutro Group wants to find training methods that use less energy. The toy problem is sparse parity (learn XOR from random bits). The real goal is nanoGPT. The workspace lets agents explore the method space, measure energy cost, and record what they find.

Three things happen in the workspace:

**Agents run experiments.** A coding agent reads CLAUDE.md and DISCOVERIES.md, picks a method, runs it against the locked harness, measures DMC (data movement cost), and writes up the finding. The two-phase protocol separates raw numbers (results.json) from interpretation (findings.md). LAB.md defines the rules. The run-experiment skill defines the steps.

**Agents get evaluated.** The eval environment (Gymnasium) wraps the same experiments into a benchmark. An agent picks from 16 methods, observes DMC, and gets graded on 12 categories (72 points). Did it find GF(2)? Did it notice local learning fails? Did it observe the ARD/DMC ranking disagreement? The answer key has 37 experiments as ground truth.

**Humans review and merge.** Contributors submit PRs. The reviewing agent checks locked files, findings format, log classification. If the result is significant, it gets a changelog entry. If not, it still merges. The weekly catch-up summarizes what happened across Telegram, Google Docs, and GitHub.

## How the pieces connect

```
CLAUDE.md ──────────────────────────── Agent reads this first
    │                                  (problem context, methods table,
    │                                   current best results)
    │
    ├── LAB.md ─────────────────────── Experiment protocol
    │     └── Two-phase results         Phase 1: results.json (numbers)
    │         Reproducibility rules     Phase 2: findings.md (analysis)
    │         Metric isolation          Don't edit tracker.py/harness.py
    │
    ├── DISCOVERIES.md ─────────────── What's proven (read before every experiment)
    │     └── 37 experiments            DMC rankings, failure modes, scaling walls
    │
    ├── AGENT.md ───────────────────── Autonomous loop protocol
    │     ├── AGENTS.md                 Multi-agent variant
    │     ├── CODEX.md                  Codex-specific entry point
    │     ├── CONTEXT.md                Shared problem context
    │     └── Pick hypothesis           From TODO.md or questions.yaml
    │         Run against harness       search_space.yaml bounds
    │         Classify result           WIN / PARTIAL / LOSS
    │         Log to log.jsonl          Append-only
    │
    ├── .claude/ ───────────────────── Claude Code specific
    │     ├── hooks/                    session-start (status), security-guard
    │     │                             (locked files), session-end (summary)
    │     ├── rules/                    Reproducibility, agent coordination
    │     ├── skills/                   run-experiment, weekly-catchup,
    │     │                             prepare-meeting
    │     └── settings.json             Hook configuration
    │
    ├── src/ ───────────────────────── Code
    │     ├── bytedmd/                  Vendored ByteDMD (primary metric since
    │     │                             2026-04-15). Byte-granularity LRU stack.
    │     ├── sparse_parity/            Main package
    │     │     ├── harness.py          Locked evaluation harness
    │     │     ├── tracker.py          Element-level memory tracker (legacy)
    │     │     ├── cache_tracker.py    LRU cache + Bill Dally energy model
    │     │     ├── tracked_numpy.py    TrackedArray (legacy auto DMD)
    │     │     ├── lru_tracker.py      Stack-distance backend
    │     │     ├── metrics.py          ARD / DMC reporting
    │     │     ├── data.py, config.py  Locked data + config
    │     │     ├── model.py, fast.py   numpy MLP + fast train loop
    │     │     ├── train.py            Reference SGD trainer
    │     │     ├── train_fused.py      Fused-op variant
    │     │     ├── train_perlayer.py   Per-layer accounting variant
    │     │     ├── run.py              Single-experiment entry
    │     │     ├── experiments/        Per-experiment scripts (one per method)
    │     │     ├── eval/               Gymnasium env (see below)
    │     │     ├── reference/          Reference implementations
    │     │     └── telegram_sync/      Python sync helpers
    │     ├── telegram/                 TS sync (db.ts, env.ts, sync.ts)
    │     ├── harness.py                Top-level harness shim
    │     ├── sync_google_docs.py       Pull Google Docs to markdown
    │     └── plot_dmc.py               DMC plot helpers
    │
    ├── src/sparse_parity/eval/ ────── Eval environment
    │     ├── env.py                    Gymnasium: SparseParity-v0
    │     ├── grader.py                 12 categories, 72 points
    │     ├── registry.py               Add methods without editing env.py
    │     ├── backends.py               Local / Modal / Remote
    │     ├── answer_key.json           37 experiments as ground truth
    │     └── adapters/                 Anthropic, PrimeIntellect, HuggingFace
    │
    ├── research/ ──────────────────── Autonomous research state
    │     ├── log.jsonl                 Append-only experiment log (37 entries)
    │     ├── questions.yaml            Open-question dependency graph
    │     ├── search_space.yaml         Bounded mutation space per challenge
    │     ├── sparse-parity-literature.md
    │     └── README.md
    │
    ├── findings/ ──────────────────── 38 finding files (exp_*.md), one per
    │                                   experiment. Templates start with `_`.
    │
    ├── contributions/ ─────────────── Drop-zone for raw external contributions.
    │                                   No template required.
    │
    ├── tests/ ─────────────────────── Unit tests (ByteDMD + sparse_parity)
    │
    ├── bin/ ───────────────────────── Operational scripts
    │     ├── reproduce-all             Re-run all canonical experiments
    │     ├── run-agent                 Launch autonomous agent cycle
    │     ├── analyze-log               Summarize log.jsonl
    │     ├── merge-findings            Import contributor log entries via PR
    │     ├── tg-sync, tg-post, tg-auth Telegram CLI
    │     └── gpu_egd.py, gpu_energy.py GPU energy probes
    │
    ├── checks/ ────────────────────── Pre-flight verification
    │     ├── env_check.py              Environment sanity check
    │     └── baseline_check.py         Re-establish baselines on this machine
    │
    ├── docs/ ──────────────────────── MkDocs site source
    │     ├── index.md, context.md, changelog.md, getting-started.md,
    │     │   goals.md, learning-guide.md, branch-workflow.md, references.md
    │     ├── research/                 This page lives here
    │     ├── findings/                 Curated finding write-ups (site)
    │     ├── tasks/                    Current task tracker (INDEX.md)
    │     ├── catchups/                 Weekly catch-ups
    │     ├── meeting-notes/, meetings/ Meeting records
    │     ├── sessions/                 Video transcripts and chapters
    │     ├── lectures/, homework/      Learning material
    │     ├── agent-prompts/            Reusable prompts
    │     ├── plans/                    Design docs
    │     ├── google-docs/              Mirror of synced Google Docs
    │     ├── tooling/                  Automation runbooks
    │     ├── results/                  Result write-ups
    │     ├── diagrams/, stylesheets/   Assets
    │     └── overrides/ (repo root)    MkDocs theme overrides
    │
    └── Automation ─────────────────── Sync and reporting
          ├── sync_telegram.ts          Pull Telegram messages
          ├── index.ts                  TS entry shared by sync scripts
          ├── package.json              bun deps for TS tooling
          ├── pyproject.toml            Python package metadata
          ├── flake.nix                 Reproducible Nix dev shell
          ├── mkdocs.yml                Site build config
          ├── src/sync_google_docs.py   Pull Google Docs to markdown
          ├── docs/catchups/            Weekly summaries
          ├── docs/sessions/            Video transcripts and chapters
          └── telegram.db               SQLite mirror (gitignored)
```

## The two costs

Yaroslav's constraint: experiments should run within 1980s compute budgets. Under 1 second, ideally under 10ms.

The workspace measures two costs:

**Experiment cost.** How much data does the method move? DMC (data movement complexity) captures this. GF(2) at DMC 8,607 is 9 million times cheaper than Fourier at 78 billion. 4 of 16 methods run under 10ms. The eval environment tracks this per step.

**Agent cost.** How many experiments did the agent need? How many tokens of reasoning? The eval environment's efficiency category awards 5 points for finding the best method in 3 steps. But we don't yet track token cost or the agent's thinking time between steps. The goal is to measure both: can an agent find GF(2) using few tokens and small wall clock time, without trying all 16 methods?

## Who does what

**The coding agent** reads CLAUDE.md, runs experiments, writes findings. It doesn't need to know about the eval environment or the hooks. Those are layers on top.

**The hooks** (Claude Code only) fire automatically. Session-start shows status so the agent doesn't ask "catch me up." The security guard blocks edits to measurement code. Session-end shows what changed. Other coding agents (Gemini, Codex) skip hooks and read CLAUDE.md directly.

**The eval environment** is a different mode. Instead of the agent writing experiment code, it picks from 16 existing methods and observes results. The grading checks whether it made specific discoveries. This can run locally, through Anthropic tool calls, on PrimeIntellect, or as a HuggingFace leaderboard.

**The weekly catch-up** syncs Telegram, Google Docs, and GitHub into a summary page. The prepare-meeting skill compiles recent experiments into a report.

**Contributors** fork the repo, run experiments, submit PRs. The reviewing agent checks the PR against the experiment protocol. See the [changelog](../changelog.md) for merged contributions.

## What each file is for

| File | Who reads it | What it does |
|------|-------------|-------------|
| CLAUDE.md | All agents | Problem context, methods table, best results |
| AGENTS.md | Multi-agent runs | Coordination entry point for parallel agents |
| CODEX.md | Codex sessions | Codex-specific context shim |
| CONTEXT.md | All agents | Shared problem context referenced by entry files |
| LAB.md | Agents running experiments | Protocol, rules, templates |
| AGENT.md | Autonomous agent loop | Pick hypothesis, run, classify, log |
| AGENT_EVAL.md | Agents using the eval env | How to add methods, run evals, read grading |
| CONTRIBUTING.md | External contributors | PR workflow, fork-and-branch, locked files |
| DISCOVERIES.md | Everyone, before every experiment | Proven facts, open questions |
| TODO.md | Agents looking for work | Hypothesis queue |
| README.md | First-time visitors | Repo orientation |
| .claude/settings.json | Claude Code | Hook configuration |
| .claude/rules/*.md | Claude Code | Reproducibility, coordination constraints |
| .claude/skills/*/SKILL.md | Claude Code | Workflow definitions |
| src/bytedmd/ | Agents measuring cost | Vendored ByteDMD tracer (primary metric) |
| src/sparse_parity/harness.py | All experiments | Locked evaluation harness (do not edit in PRs) |
| src/sparse_parity/tracker.py | Tracker internals | Element-level memory tracker (legacy) |
| src/sparse_parity/cache_tracker.py | Energy estimates | LRU cache + Bill Dally pJ model |
| src/sparse_parity/tracked_numpy.py | Auto-instrumented runs | TrackedArray wrapper for numpy (legacy DMD) |
| src/sparse_parity/lru_tracker.py | Stack-distance backend | Backs ARD/DMC accounting |
| src/sparse_parity/metrics.py | Reporting | ARD / DMC reporting helpers |
| src/sparse_parity/data.py, config.py | All experiments | Locked benchmark spec |
| src/sparse_parity/experiments/ | Experiment authors | One script per method (n=20, k=3) |
| src/sparse_parity/eval/ | Eval-mode agents | Gymnasium env, grader, registry, adapters |
| src/telegram/ | Telegram sync | TS modules: db.ts, env.ts, sync.ts |
| research/log.jsonl | Autonomous loop | Append-only experiment log (37 entries) |
| research/questions.yaml | Autonomous loop | Open-question dependency graph |
| research/search_space.yaml | Autonomous loop | Bounded mutation space per challenge |
| findings/exp_*.md | Reviewers, future agents | Per-experiment finding write-ups |
| contributions/ | External contributors | Drop-zone for raw results, no template required |
| tests/ | CI and agents | Unit tests for ByteDMD and sparse_parity |
| bin/reproduce-all | Anyone validating results | Re-run all canonical experiments |
| bin/run-agent | Autonomous loop | Launch one agent cycle |
| bin/analyze-log | Reviewers | Summarize log.jsonl |
| bin/merge-findings | Reviewers | Import contributor log entries via PR |
| bin/tg-sync, tg-post, tg-auth | Telegram automation | Pull messages, post to topics, auth flow |
| bin/gpu_egd.py, gpu_energy.py | Future GPU work | GPU energy probes |
| checks/env_check.py | Pre-flight | Environment sanity check |
| checks/baseline_check.py | Pre-flight | Re-establish baselines on this machine |
| docs/tasks/INDEX.md | Anyone picking work | Current task tracker |
| docs/research/survey.md | Method browsers | Practitioner field guide ranking 37 experiments |
| docs/findings/ | Site readers | Curated finding write-ups |
| docs/catchups/ | Weekly readers | Weekly catch-up summaries |
| docs/google-docs/ | Sync consumers | Mirror of synced Google Docs |
| mkdocs.yml | Site build | Navigation and theme config |
| flake.nix | NixOS users | Reproducible dev shell (python3 + numpy) |
| pyproject.toml | Python users | Package metadata and deps |
| package.json, index.ts, sync_telegram.ts | Telegram tooling | bun runtime entry for TS sync |
| telegram.db | Local query | SQLite mirror of Telegram (gitignored) |

## What still needs to happen

This workspace handles sparse parity. The path to nanoGPT requires:

1. Actual GPU energy measurement (Issue #6: compare DMC vs ARD vs real joules on an H100)
2. A harder problem (nanoGPT character-level training, Issue #9)
3. Agent cost tracking (tokens per discovery, thinking time between steps)
4. PR review automation (Issue #54: webhook + Claude Code + Telegram approval)

The workspace itself is the product. Improving how agents find better algorithms is progress along Yaroslav's "process" axis, independent of which problem they're solving.

## Docs map

| Page | What it covers |
|------|---------------|
| [Eval environment](eval-environment.md) | Gymnasium env, grading, adapters, baselines |
| [Agent infrastructure](agent-infrastructure.md) | Hooks, rules, skills, V2 diagram |
| [Adding a challenge](adding-a-challenge.md) | Step-by-step for new problems |
| [Adding an eval challenge](adding-an-eval-challenge.md) | Registry, answer key, baselines |
| [Survey](survey.md) | All 37 experiments ranked |
| [Context](../context.md) | Group history, timeline, people |
| [Peer research protocol](peer-research-protocol.md) | Multi-researcher workflow |
