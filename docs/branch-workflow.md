# Branch Workflow

How to create a branch, run experiments, and submit a PR.

## For contributors

### 1. Fork and clone

```bash
gh repo fork cybertronai/SutroYaro --clone
cd SutroYaro
```

Or if you already have the repo:

```bash
git checkout -b exp/your-experiment-name
```

### 2. Branch naming

Use a prefix that describes the type of work:

| Prefix | When to use | Example |
|--------|-------------|---------|
| `exp/` | New experiment | `exp/egd-sparse-parity` |
| `fix/` | Bug fix or correction | `fix/sgd-hinge-loss` |
| `docs/` | Documentation only | `docs/getting-started-guide` |
| `challenge/` | Adding a new challenge | `challenge/sparse-majority` |

### 3. What to include in your PR

Minimum:
- Findings doc in `findings/` using the template (`findings/_experiment_template.md`)
- Update to DISCOVERIES.md if your experiment answers an open question

If you wrote code:
- Experiment script in `src/sparse_parity/experiments/`
- Results JSON in `results/`

If you added a challenge:
- Follow all 9 steps in [adding-a-challenge.md](research/adding-a-challenge.md)

### 4. What not to modify

These files are locked (LAB.md rule #9):
- `src/sparse_parity/tracker.py`
- `src/sparse_parity/cache_tracker.py`
- `src/sparse_parity/data.py`
- `src/sparse_parity/config.py`
- `src/harness.py` (unless adding a new challenge)
- `checks/*.py`

If you think these files have a bug, note it in your findings doc. Don't fix it in your PR.

### 5. Submit

```bash
git add -A
git commit -m "exp: description of what you tried"
git push origin exp/your-experiment-name
gh pr create --title "Your experiment title" --body "Summary of what you tried and found"
```

### 6. Review

Someone (human or agent) will review. We check:
- Results are reproducible (commands in the findings doc work)
- DISCOVERIES.md is updated if applicable
- Locked files are not modified
- Findings follow the template

No CI gates. No test coverage requirements. The only hard rule: don't break existing experiments.

## For agents

If you're a coding agent running autonomously via `bin/run-agent`:

- Commit to the current branch after each experiment
- Do NOT push to main directly
- Do NOT create PRs without human review
- Log results to `research/log.jsonl`
- The human decides when to push and merge

## Repo protection

- `main` branch: direct push allowed for repo owners (Yad, Yaroslav). Contributors submit PRs.
- No required status checks (experiments are too varied for CI).
- Merging: repo owners review and merge. Andy's PRs (#2, #3) are examples of the process.
