# Codex Project Instructions

These instructions apply to all Codex CLI sessions in this repo.

## Before Any Research Task

Load current project state before running experiments, reviewing PRs, or writing findings.

1. Read these files in order:
   - **CODEX.md** -- project context, current best methods, constraints
   - **DISCOVERIES.md** -- what's proven, what failed, open questions (bottom of file)
   - **AGENT.md** -- machine-executable experiment loop (if running autonomous)
   - **LAB.md** -- experiment protocol, rules (especially rule #9: metric isolation)

2. Check recent Telegram activity (if synced):

```bash
test -f telegram.db && sqlite3 telegram.db \
  "SELECT date, sender, text FROM messages ORDER BY date DESC LIMIT 10;"
```

If `telegram.db` is missing or stale, run `bin/tg-sync`.

3. Check GitHub for open work:

```bash
gh pr list --repo cybertronai/SutroYaro --state open
gh issue list --repo cybertronai/SutroYaro --state open
```

4. Before writing code, check:
   - `research/search_space.yaml` for allowed parameter ranges
   - `research/questions.yaml` for the dependency graph of open questions

## Current State

| Fact | Value |
|------|-------|
| Fastest historical method | GF(2) Gaussian elimination, 509us, ARD ~420 |
| Primary energy metric | ByteDMD, byte-granularity LRU stack cost |
| Legacy metrics | ARD and DMC values remain useful for historical comparisons |
| Experiments done | 37+ (see `research/log.jsonl` and `docs/research/survey.md`) |
| Open questions | Bottom of `DISCOVERIES.md` |
| Next milestone | ByteDMD-aware sparse parity and nanoGPT experiments |
| Meeting cadence | Mondays 18:00 at South Park Commons |

## Sync Routine

Run at session start and before any push:

```bash
# Telegram (daily, incremental SQLite sync)
bin/tg-sync

# Query recent messages
sqlite3 telegram.db "SELECT date, sender, text FROM messages ORDER BY date DESC LIMIT 10"

# Post status updates when configured
bin/tg-post --topic agent-updates "Status update"

# Google Docs (weekly, after Monday meetings)
python3 src/sync_google_docs.py

# GitHub
gh pr list --repo cybertronai/SutroYaro --state open
gh issue list --repo cybertronai/SutroYaro --state open
```

Before pushing:
1. Update `docs/changelog.md` (bump version)
2. `python3 -m mkdocs build` to verify no broken links
3. Show the diff and wait for approval before `git push`

## Writing Rules (Anti-Slop)

Apply these to all prose (findings docs, DISCOVERIES.md updates, PR descriptions):

1. Cut filler phrases. Say the thing directly.
2. Break formulaic structures. No binary contrasts, no dramatic fragmentation.
3. Vary rhythm. Mix sentence lengths. Two items beat three.
4. Trust readers. State facts directly.
5. Prefer plain verbs. "used" not "leveraged," "showed" not "showcased."
6. Use simple copulatives. Write "X is Y" not "X serves as Y."
7. Kill em dashes. Use commas or periods.
8. Never triple. Two items in a list, not three.
9. Be specific. Replace generic statements with concrete details.
10. No AI vocabulary: delve, tapestry, landscape, pivotal, showcase, testament, underscore, foster, garner, interplay, intricate, vibrant, robust, seamless, paramount, multifaceted, nuanced, groundbreaking, cornerstone, transformative, synergy.

Full guide: `.claude/skills/anti-slop-guide/SKILL.md` (plain markdown, readable by any tool).
