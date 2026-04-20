# References: sutro-sync

Canonical docs for the three sync sources (Telegram, Google Docs, GitHub)
and the staleness check that runs before a push.

| Doc | Why |
|-----|-----|
| [../../../../docs/tooling/sync-runbook.md](../../../../docs/tooling/sync-runbook.md) | Full pre-push checklist (primary runbook) |
| [../../../../docs/tooling/telegram-setup.md](../../../../docs/tooling/telegram-setup.md) | First-time Telegram auth, `.env`, Bot API posting |
| [../../../../docs/tooling/automation.md](../../../../docs/tooling/automation.md) | `bin/tg-sync`, `bin/tg-post`, `src/sync_google_docs.py` |
| [../../../../docs/tooling/index.md](../../../../docs/tooling/index.md) | Tooling overview and inventory |
| [../../../../docs/changelog.md](../../../../docs/changelog.md) | Version history; bump before any push |
| [../../../../bin/tg-sync](../../../../bin/tg-sync) | Telegram sync script |
| [../../../../bin/tg-post](../../../../bin/tg-post) | Bot-API posting script |
| [../../../../src/sync_google_docs.py](../../../../src/sync_google_docs.py) | Google Docs pull script |
