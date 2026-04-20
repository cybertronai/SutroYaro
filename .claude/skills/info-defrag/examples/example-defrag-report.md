# Example: info-defrag report

Output of the skill run on 2026-04-15 after merging PR #80 (ByteDMD became
the primary metric). Lists every stale item with file, location, current
text, and proposed fix. No auto-edits.

## 1. Experiment count drift

| File | Line | Current | Should be |
|------|------|---------|-----------|
| `docs/index.md` | ~40 | "30 experiments" | 37 (match research/log.jsonl) |
| `CLAUDE.md` | Phase 2 header | "17 experiments" | fine (phase local count) |
| `README.md` | intro | "33 experiments" | 37 |

Evidence:

```
$ wc -l research/log.jsonl     # 37
$ grep -c '^| exp_' DISCOVERIES.md   # 37
```

## 2. Best methods table drift (ByteDMD shift)

`CLAUDE.md` still ranks methods by legacy element-level DMC. Post-PR #80
(2026-04-15), ByteDMD is the primary metric and none of the table rows have
been re-measured. Added a warning admonition on top of the table; full
re-rank pending.

Action: keep the legacy table marked "pre-ByteDMD" until methods are
re-measured; do not delete numbers.

## 3. People descriptions

```
$ git log --format="%an" --since="2 weeks ago" | sort -u
Andy Zhang
Claude
Michael Keating
Yad Konrad
```

`docs/context.md` People table lists Andy but not his recent PRs (#65, #73).
Proposed: extend Andy's bio with "submitted GF(2) noise robustness results".

## 4. Timeline coverage

Last gantt date: 2026-03-22. Last commit: 2026-04-15. Gap is 24 days.
Action: extend timeline to include Meeting #10 (Mar 23) and ByteDMD shift
(Apr 15).

## 5. Index pages

```
$ ls docs/findings/exp*.md | wc -l        # 24
$ grep -c 'findings/exp' mkdocs.yml       # 22
```

Missing from nav: `exp_dmc_optimize.md`, `exp_km_noise.md`. Action: add
both to `mkdocs.yml` under Research > Findings.

## 6. Broken links

None this run. (Previous scan caught `docs/findings/exp_gf2_v2.md` linked
from `DISCOVERIES.md` but never created -- fixed 2026-03-28.)

## 7. Stale TODO items

`TODO.md` still lists "Backfill scoreboard.tsv with DMC values (#16)" as
open. GitHub shows #16 closed 2026-03-24. Action: check TODO.md off.

## Summary

8 drift items found. 0 auto-fixed (by design). User decides what to update.
