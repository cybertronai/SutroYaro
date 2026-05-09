# Example: context load for a PR review session

Snapshot of what `sutro-context` surfaced at the start of a PR review on
2026-03-28. Keep this short and concrete, mirroring the skill's four steps.

## Step 1: core files (skim, note anything unexpected)

- `CLAUDE.md`: best method is GF(2) Gaussian elimination at 509 us; DMC is
  the primary energy proxy. SGD baseline still LR=0.1, batch=32, hidden=200.
- `DISCOVERIES.md`: 33+ experiments. Open questions Q7 (real joules vs DMC),
  Q11-Q13 (noisy parity, feature count scaling, curriculum at k=5).
- `AGENT.md`: two-phase protocol. Phase 1 = raw results.json, Phase 2 =
  findings doc with Status: SUCCESS/PARTIAL/FAILED.
- `LAB.md`: rule 9 still in force -- no edits to tracker.py, cache_tracker.py,
  harness.py, data.py, config.py in an experiment PR.

## Step 2: recent activity

Telegram (last 3 days, priority topics):

- chat-yaroslav, 2026-03-21: Lukas Kaiser and Alec Radford may visit Mar 30.
- chat-yad, 2026-03-21: SutroYaro designated Public Domain. Yaroslav in NYC
  Mar 24-30.
- challenge-1-sparse-parity, 2026-03-16: Michael forked Autoresearch against
  sparse parity using Opus and "unconventional mathematical theories".

GitHub:

```
gh pr list --state open       # 0
gh issue list --state open    # 8 open (homework #17, #22 highest priority)
```

## Step 3: current state snapshot

| Fact | Value |
|------|-------|
| Best method | GF(2), 509 us, ARD ~420, DMC 8,607 (legacy tracking) |
| New DMC leader (exp_dmc_optimize) | KM-min, DMC 3,578 |
| Primary metric | DMC, moving to ByteDMD (byte-granularity) |
| Next milestone | Energy-efficient nanoGPT training |
| Meeting cadence | Mondays 18:00 at South Park Commons |

## Step 4: before writing code

- `research/search_space.yaml` is the allow-list for parameter mutations.
- `research/questions.yaml` shows which questions block others.
- `checks/env_check.py` and `checks/baseline_check.py` confirm the machine
  reproduces baseline numbers before any new measurement is trusted.
