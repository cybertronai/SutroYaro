---
name: sutro-context
description: Use before any research task, experiment, or PR review. Loads current project state from DISCOVERIES.md, open questions, and recent Telegram discussion.
---

# Sutro Research Context

Load this before doing any research work, running experiments, or reviewing PRs.

## Step 1: Read core files

Read these in order. Stop and report if anything is unexpected.

1. **CLAUDE.md** -- project context, current best methods, constraints
2. **DISCOVERIES.md** -- what's proven, what failed, open questions (bottom of file)
3. **AGENT.md** -- machine-executable experiment loop (if running autonomous)
4. **LAB.md** -- experiment protocol, rules (especially rule #9: metric isolation)

## Step 2: Check recent activity

Read the last 10 messages from priority Telegram topics:

```python
import json
for f in ['chat-yad.json', 'chat-yaroslav.json', 'challenge-1-sparse-parity.json']:
    path = f'src/sparse_parity/telegram_sync/{f}'
    try:
        msgs = json.load(open(path))
        print(f'\n=== {f} (last 3) ===')
        for m in msgs[:3]:
            print(f"  [{m['date'][:10]}] {m['sender']}: {m['text'][:150]}")
    except FileNotFoundError:
        print(f'{f} not found -- run: bun run sync_telegram.ts')
```

Check GitHub for open work:

```bash
gh pr list --repo cybertronai/SutroYaro --state open
gh issue list --repo cybertronai/SutroYaro --state open
```

## Step 3: Know the current state

| Fact | Value |
|------|-------|
| Best method | GF(2) Gaussian elimination, 509us, ARD ~500 |
| Best energy proxy | DMC (Data Movement Complexity, Ding et al.) |
| Experiments done | 33+ (see `research/log.jsonl`) |
| Open questions | Bottom of DISCOVERIES.md (Q7, Q11-Q13 still open) |
| Next milestone | Energy-efficient nanoGPT training ("final exam") |
| Meeting cadence | Mondays 18:00 at South Park Commons |

## Step 4: Before writing code

- Check `research/search_space.yaml` for allowed parameter ranges
- Check `research/questions.yaml` for the dependency graph of open questions
- Run `checks/env_check.py` to verify environment
- Run `checks/baseline_check.py` if baselines may differ on your machine

## For contributors using other tools

This context applies regardless of which AI tool you use. The key files are plain markdown and YAML. Read them before starting work. The experiment protocol in LAB.md and AGENT.md defines the loop: hypothesis, code, run, measure, record.
