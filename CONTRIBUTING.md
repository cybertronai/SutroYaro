# Contributing

Anyone in the Sutro Group (or interested outsiders) can contribute experiments, findings, or raw results.

## The short version

1. Fork the repo
2. Run an experiment on sparse parity (with any tool: Claude Code, Replit, Gemini, plain Python, pencil and paper)
3. Submit a PR with your findings

We accept contributions at three levels of formality. Pick the one that fits your workflow.

## Level 1: Drop raw results (lowest effort)

Put whatever you have in `contributions/`. No format required.

```
contributions/
  germain-depth1-results.md    # copy-paste from your agent's output
  michael-claude-log.txt       # raw Claude conversation
  notes-from-monday.md         # meeting notes, ideas, observations
```

Someone (usually Claude Code) will reformat it later. The point is to get the information into the repo, not to make it pretty.

## Level 2: Write a findings doc (medium effort)

Copy the template and fill it in:

```bash
cp findings/_template.md findings/exp_your_name.md
```

The template asks for: hypothesis, config, results table, analysis, open questions. See any file in `findings/` for examples. This is the format that feeds into the [survey](https://cybertronai.github.io/SutroYaro/research/survey/).

## Level 3: Full experiment with code (highest effort)

Follow the lab protocol:

```bash
# 1. Read what's known
cat DISCOVERIES.md

# 2. Copy the code template
cp src/sparse_parity/experiments/_template.py src/sparse_parity/experiments/exp_yours.py

# 3. Edit, run, measure
PYTHONPATH=src python3 src/sparse_parity/experiments/exp_yours.py

# 4. Write findings
cp findings/_template.md findings/exp_yours.md

# 5. Submit PR with all three: experiment code, results JSON, findings doc
```

## Setup

### Option A: Nix (recommended)

One command gets you everything: Python, numpy, bun, pandoc, mkdocs.

```bash
# 1. Install nix (if you don't have it)
curl --proto '=https' --tlsv1.2 -sSf https://install.determinate.systems/nix | sh -s -- install

# 2. Clone and enter dev shell
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro
nix develop

# 3. Verify it works
python3 -m sparse_parity.fast
# Should print: 100% accuracy in ~0.12s

# 4. Run environment check
python3 checks/env_check.py
# Should print: All checks passed.
```

The dev shell sets `PYTHONPATH=src` automatically, so imports work without manual setup.

**What's included:**
- Python 3 + numpy (core experiments)
- bun (Telegram sync)
- pandoc (Google Docs sync)
- mkdocs-material + plugins (docs site)

**Direnv (optional):** If you use direnv, `direnv allow` will auto-load the dev shell when you cd into the repo.

### Option B: pip/uv (fallback)

If you don't want to install nix:

```bash
git clone https://github.com/cybertronai/SutroYaro.git
cd SutroYaro

# Core deps only (experiments)
pip install numpy

# Or with uv
uv pip install numpy

# Set PYTHONPATH and verify
export PYTHONPATH=$PWD/src:$PYTHONPATH
python3 -m sparse_parity.fast
```

Optional deps for sync/docs:
```bash
pip install mkdocs-material mkdocs-mermaid2-plugin pymdown-extensions
# pandoc: brew install pandoc (macOS) or apt install pandoc (Linux)
# bun: curl -fsSL https://bun.sh/install | bash
```

## What to work on

Open questions live in [DISCOVERIES.md](DISCOVERIES.md) under "Open Questions." The [task tracker](docs/tasks/INDEX.md) has current priorities. Some starting points:

- **Reproduce a result**: pick any experiment from the [survey](https://cybertronai.github.io/SutroYaro/research/survey/) and verify the numbers
- **Try a new algorithm**: check [proposed-approaches.md](docs/research/proposed-approaches.md) for untested ideas
- **Improve the metric**: we just added DMC (Data Movement Complexity) alongside ARD. Both could use testing on more configs.
- **Scale testing**: does your method work at n=50/k=5? n=100/k=10?

## Rules

1. **Read DISCOVERIES.md before experimenting** so you don't repeat what's known
2. **One hypothesis per experiment** so results are interpretable
3. **Always compare against a baseline** so numbers have context
4. **Don't modify measurement code** (tracker.py, cache_tracker.py, data.py, config.py). If you need to change the metric, that's a separate PR.
5. **Record negative results** because knowing what doesn't work prevents others from trying it

## PR process

- Fork and branch
- Push your branch, open a PR against `main`
- Describe what you tried and what happened
- Someone (human or AI) will review and merge

`main` is now branch-protected: PRs require **1 approval** before merge. Repo admins can override for hotfixes (`gh pr merge --admin`). Force-push and branch deletion are blocked.

The only CI gate today: `diagram-staleness.yml` (see "Updating the repo diagrams" below). No test coverage requirements. Hard rule: don't break existing experiments.

## Updating the repo diagrams

The two diagrams on the docs site —
[Repo Layout](https://cybertronai.github.io/SutroYaro/research/repo-layout/) (Mermaid)
and [Interactive Repo Tree](https://cybertronai.github.io/SutroYaro/research/repo-tree/) (D3) —
are generated from a single YAML.

**Don't edit the `.md` files directly.** They have `BEGIN_AUTOGEN` / `END_AUTOGEN` markers around the data blocks; anything between is rewritten by the generator.

```bash
# 1. Edit the source of truth
$EDITOR docs/research/_diagrams.yaml

# 2. Regenerate both diagrams
bin/regen-diagrams

# 3. Commit YAML + regenerated .md files together
git add docs/research/_diagrams.yaml docs/research/repo-{tree,layout}.md
git commit -m "diagrams: ..."
```

CI (`diagram-staleness.yml`) fails any PR where the `.md` files have drifted from a fresh regen. The error message tells you to run `bin/regen-diagrams` locally and commit.

**What's auto-counted** (don't hardcode these in the YAML):
- `{findings_count}` — number of `findings/exp_*.md` files
- `{experiments_jsonl_count}` — line count of `research/log.jsonl`
- `{task_count}` — number of `docs/tasks/[0-9]*-*.md` files

## Releases and version tags

Versions are recorded in [`docs/changelog.md`](docs/changelog.md) following [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/) format.

Each released version has a matching annotated git tag (`v0.29.0`, `v0.28.0`, etc.) pointing at the changelog commit that shipped it. Browse them via `git tag -l 'v*'` or on the [GitHub releases page](https://github.com/cybertronai/SutroYaro/releases).

When you add a new release entry to `docs/changelog.md`, the workflow is:

```bash
# After the changelog PR merges to main
git fetch origin --tags
git tag -a v0.X.0 <merge-commit-sha> -m "v0.X.0"
git push origin v0.X.0
```

Tag the **merge commit** that landed the changelog entry, not your feature branch.

**Tag coverage:** v0.16.0, v0.17.0, v0.23.0, v0.24.0, v0.25.0, v0.27.0, v0.28.0, v0.29.0. Earlier releases (v0.5.0–v0.15.0, v0.18.0–v0.22.0, v0.26.0) are documented in `docs/changelog.md` but have no git tags — the corresponding commits couldn't be unambiguously identified retroactively. Use the changelog as the authoritative source for those versions.

## Questions

Ask in the [Telegram group](https://t.me/sutro_group) or open a GitHub issue.
