# SutroYaro

Research workspace for the [Sutro Group](https://t.me/sutro_group) — energy-efficient AI training, meeting weekly at South Park Commons (SF).

**Docs site**: https://0bserver07.github.io/SutroYaro/

## What This Is

An autonomous research lab for the sparse parity challenge: learn XOR/parity from random numbers, scale to 20+ bits with noise, and measure/improve energy efficiency via Average Reuse Distance (ARD).

We use Claude Code with parallel agent teams to run experiments, record findings, and accumulate knowledge automatically.

## Key Results

**20-bit sparse parity (k=3) solved** — 100% accuracy in 0.12 seconds (numpy) across 5 seeds.

- LR=0.1, batch=32, hidden=200, n_train=1000
- Classic grokking pattern: hidden progress invisible to loss metrics, then sharp phase transition
- Per-layer forward-backward gives 3.8% ARD improvement for free
- Forward-Forward algorithm has 25x worse ARD than backprop (opposite of hypothesis)
- Curriculum learning (n=10→30→50) gives 14.6x speedup, cracks n=50 which direct training can't
- Sign SGD solves k=5 2x faster; standard SGD also works with enough data (n_train=5000)
- Weight decay 0.01 optimal, narrow working range [0.01, 0.05]

See [DISCOVERIES.md](DISCOVERIES.md) for the full knowledge base and [changelog](https://0bserver07.github.io/SutroYaro/changelog/) for version history.

## Quick Start

```bash
# Run the fast solver (numpy, <0.2s)
PYTHONPATH=src python3 -m sparse_parity.fast

# Run the full pipeline (pure Python, 3 training variants, ARD comparison)
PYTHONPATH=src python3 -m sparse_parity.run

# Run tests
python3 -m pytest tests/ -v

# Run a specific experiment
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_sign_sgd.py
```

## Running Your Own Experiments

This repo is set up for autonomous AI-assisted research. See [LAB.md](LAB.md) for the full protocol.

1. Read [DISCOVERIES.md](DISCOVERIES.md) — what's proven, what's open
2. Pick an open question
3. Copy `src/sparse_parity/experiments/_template.py`
4. Run, measure, write findings, commit
5. Update DISCOVERIES.md with your result

Each experiment produces:
- `src/sparse_parity/experiments/exp_{name}.py` — the code
- `findings/exp_{name}.md` — analysis (strict template)
- `results/exp_{name}/results.json` — machine-readable metrics

## Using Claude Code for Research

This repo demonstrates a workflow for AI-assisted research:

1. **Literature review** — search arxiv, identify the gap between your config and published baselines
2. **Diagnose** — compare your hyperparameters against the literature (this alone solved our 20-bit problem)
3. **Experiment** — one hypothesis per experiment, always compare against baseline
4. **Record** — structured findings so future sessions don't repeat work
5. **Iterate** — each experiment leaves open questions for the next

The key files that make this work:
- `LAB.md` — protocol for autonomous experiment sessions
- `DISCOVERIES.md` — accumulated knowledge (read before every experiment)
- `_template.py` — experiment code starter
- `findings/*.md` — structured experiment reports

For parallel execution, we use Claude Code's team agents feature: a lead agent reads DISCOVERIES.md, creates tasks, and dispatches worker agents that each run an independent experiment.

See [findings/prompting-strategies.md](findings/prompting-strategies.md) for detailed prompting lessons.

## Structure

```
LAB.md              # Lab protocol for autonomous sessions
DISCOVERIES.md      # Accumulated knowledge base
TODO.md             # Open research tasks

src/sparse_parity/
  fast.py           # Numpy-accelerated solver (0.12s)
  config.py         # Experiment configuration
  data.py           # Dataset generation
  model.py          # MLP model + forward pass
  tracker.py        # ARD measurement (MemTracker)
  cache_tracker.py  # Cache-aware ARD (CacheTracker)
  train.py          # Standard backprop
  train_fused.py    # Fused layer-wise updates
  train_perlayer.py # Per-layer forward-backward
  run.py            # Full pipeline runner
  experiments/      # All experiment scripts

findings/           # Experiment reports (13 so far)
results/            # JSON metrics per experiment
research/           # Literature review
docs/               # MkDocs site source
```

## Links

- Docs site: https://0bserver07.github.io/SutroYaro/
- Telegram: https://t.me/sutro_group
- Main code repo: https://github.com/cybertronai/sutro
- Meetings: Mondays 18:00 at South Park Commons (380 Brannan St, SF)
