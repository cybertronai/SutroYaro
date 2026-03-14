---
name: sutro-sync
description: Use at the start of any session and before pushing. Syncs Telegram, Google Docs, and GitHub state for the Sutro Group research workspace.
---

# Sutro Sync Routine

Run this at session start and before any push. Covers three sources: Telegram, Google Docs, GitHub.

## 1. Telegram (daily)

```bash
bun run sync_telegram.ts
```

Pulls 6 topics in priority order to `src/sparse_parity/telegram_sync/`:

| Priority | Topic | File |
|----------|-------|------|
| 1 | chat-yad | `chat-yad.json` |
| 2 | chat-yaroslav | `chat-yaroslav.json` |
| 3 | challenge #1: sparse parity | `challenge-1-sparse-parity.json` |
| 4 | General | `general.json` |
| 5 | In-person meetings | `in-person-meetings.json` |
| 6 | Introductions | `introductions.json` |

Read the last 5 messages from priority topics 1-3 to check for new questions, directions, or results. Report anything relevant to the user.

**Prerequisites**: `bun install`, `.env` with `TELEGRAM_API_ID` and `TELEGRAM_API_HASH`, `tg auth login` (one-time).

## 2. Google Docs (weekly, after Monday meetings)

```bash
python3 src/sync_google_docs.py
```

Pulls 15+ Google Docs to `docs/google-docs/`. After syncing:

1. Check `docs/google-docs/sutro-group-main.md` for new meeting doc links
2. If new docs found, add them:
   ```bash
   python3 src/sync_google_docs.py --add "URL" "name" "Description"
   python3 src/sync_google_docs.py   # re-run
   ```
3. Add cross-reference headers (`!!! info` admonition) to new docs
4. Add new pages to `mkdocs.yml` nav under Meetings > Google Docs
5. Update `docs/meetings/index.md` and `docs/meetings/notes.md`

**Prerequisites**: `pandoc` (`brew install pandoc`). Docs must be "Anyone with the link" sharing.

## 3. GitHub (daily)

```bash
gh pr list --repo cybertronai/SutroYaro --state open
gh issue list --repo cybertronai/SutroYaro --state open
```

Check for new PRs from contributors (Andy/zh4ngx, Michael, others). Review code, verify results are reproducible, check that DISCOVERIES.md is updated.

## Before pushing

1. Update `docs/changelog.md` (bump version)
2. `python3 -m mkdocs build` to verify no broken links
3. `git push origin main`
4. Deploy: `python3 -m mkdocs gh-deploy --force`

## Tool-agnostic notes

This skill works with any AI coding tool that reads files:

- **Claude Code**: reads `.claude/skills/` automatically
- **Cursor/Windsurf**: paste the commands or add to `.cursorrules`
- **Codex/other**: include this file in the system prompt or project context
- **Manual**: follow the checklists in `docs/tooling/sync-runbook.md`

All state lives in files (JSON, markdown, YAML). No tool-specific APIs needed.
