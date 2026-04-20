# Example: sutro-sync output

Snapshot of a session-start sync run on 2026-03-22 (Sunday before
Meeting #10). Mirrors the three sources the skill covers.

## 1. Telegram

```
$ bin/tg-sync
Syncing chat-yad            (id 4) ... 12 new messages
Syncing chat-yaroslav       (id 5) ... 8 new messages
Syncing challenge-1-sparse-parity (id 7) ... 4 new messages
Syncing general             (id 2) ... 21 new messages
Total: 861 messages across 6 topics.
```

Priority topics, last 3 days:

| Date       | Sender    | Snippet |
|------------|-----------|---------|
| 2026-03-21 | Yaroslav  | SutroYaro -> Public Domain. Visiting NYC Mar 24-30. |
| 2026-03-21 | Yaroslav  | Lukas Kaiser + Alec Radford may stop by Mar 30 meeting. |
| 2026-03-19 | Yaroslav  | Wolfram trains nets with discrete AND/XOR grids, no floats. |
| 2026-03-16 | Michael   | Forked Karpathy's Autoresearch against sparse parity. |

## 2. Google Docs

```
$ python3 src/sync_google_docs.py
Fetched 17/17 docs. 1 new: Meeting #9 notes.
```

Follow-ups:

1. Added Meeting #9 doc link via `--add`, re-synced.
2. Added cross-reference `!!! info` admonition at top of new doc.
3. Added page to `mkdocs.yml` nav under Meetings -> Google Docs.

## 3. GitHub

```
$ gh pr list --repo cybertronai/SutroYaro --state open
(no open PRs)

$ gh issue list --repo cybertronai/SutroYaro --state open | head
#22 DMC optimization experiment: beat baseline on at least one method
#17 DMC baseline sweep: measure all methods
#15 Add tracker integration to fast.py
#16 Backfill scoreboard.tsv with DMC values
#18 DMC visualization and plotting
(8 open total)
```

## 4. Staleness check (before push)

`docs/index.md` mentioned 30 experiments; `DISCOVERIES.md` had 33. Flagged
and updated homepage to 33 before pushing. mkdocs build: 0 broken links.
