# Active research threads — 2026-05-09 snapshot

A single-source-of-truth dump of what every active contributor is working on right now, pulled from Telegram (last 3 days), the Google Docs mirror (meeting #16 = 04 May 26), and GitHub state (open PRs/issues across SutroYaro + companion repos).

This is a snapshot, not a tracker. Bumped weekly as a catch-up artifact; for ongoing work see [docs/tasks/INDEX.md](../tasks/INDEX.md).

## Yaroslav

### 1. 2D-grid Dally cost model
- **Repo**: [`cybertronai/simplified-dally-model`](https://github.com/cybertronai/simplified-dally-model)
- **Quote**: *"Looking into extending sparse-parity challenge to use the 2D grid model. Might need adding instructions to ... trying to see how few I can get away [with]"* (Telegram, 2026-05-08T00:56)
- **Status**: iterating on the minimum instruction set
- **SutroYaro implication**: when stable, eval-environment + scoreboard need a `--metric=2d-grid` path
- **Tracked**: [Task #12](../tasks/012-2d-grid-dally-sparse-parity.md)

### 2. Accuracy vs joules graph
- **Quote**: *"One thing that would be cool to get is an explicit accuracy as a function of joules graph"* (Telegram, 2026-05-08T01:02)
- **Why it matters**: this is the headline visualization for the energy-efficient-training argument; useful for the next Sutro Group meeting and any external write-up
- **Tracked**: [Task #13](../tasks/013-accuracy-vs-joules-graph.md)

### 3. External reading
- [How to improve AI energy efficiency by 1000x — unconv.ai](https://unconv.ai/blog/how-to-improve-ai-energy-efficiency-by-1000x/) (shared 2026-05-08) — aligned with the SutroYaro thesis.

## Andy Zhang (zh4ngx)

### 1. Sparse-parity over time ("morse code" framing)
- **Quote**: *"yes! love it. i was looking at sparse parity over time (like morse code)"* (Telegram, 2026-05-08T01:01)
- **What it is**: same parity function, but the n bits arrive one per timestep instead of as a fixed-length vector
- **Connects to**: the schmidhuber-problems `rs-parity` stub already implements this in numpy via random-weight-guessing on a recurrent net
- **Tracked**: [Task #14](../tasks/014-sparse-parity-over-time.md)

### 2. Sutro agent + subagent reading pipeline
- **Quote**: *"i've been having my sutro agent and subagents read + synthesize articles, has been helpful to me"* (Telegram, 2026-05-08T01:04)
- **Status**: independent agent infrastructure; not currently surfaced in SutroYaro

### 3. Permissions
- 2026-05-06: Yaroslav promoted Andy to **owner** on `cybertronai` org so he has write access to `sutro-problems` and `hinton-problems`. (Telegram thread)

## Yad

### 1. Companion problem-set catalogs (shipped)

| Repo | Output | Wall hours | Tokens (real) | Status |
|---|---|---:|---:|---|
| [`cybertronai/hinton-problems`](https://github.com/cybertronai/hinton-problems) | 53 stubs | ~30 | ~661M, 93.5% cache_read | All v1 PRs merged, site live |
| [`cybertronai/schmidhuber-problems`](https://github.com/cybertronai/schmidhuber-problems) | 58 stubs (50 v1 + 8 v1.5) | ~41 | ~1.15B, 91.5% cache_read | All 13 PRs merged, site live, follow-up tracking issues #17 (v2 ByteDMD) and #18 (v1.5 paper-scale) open |

Both ship a `BUILD_NOTES.md` § Token consumption with the JSONL-counted breakdown — the harness "780k" display was context-window utilisation, not cumulative cost. See [hinton-problems #56](https://github.com/cybertronai/hinton-problems/issues/56) and [schmidhuber-problems #19](https://github.com/cybertronai/schmidhuber-problems/issues/19) for the methodology.

### 2. SutroYaro housekeeping
- [Issue #95 (Housekeeping: PR triage, doc staleness)](https://github.com/cybertronai/SutroYaro/issues/95) — meta-tracking for catch-up
- 3 stale PRs awaiting triage (see "GitHub state" below)

## Other contributors

### Seth (SethTS)
- [PR #94 — KM-min + SAT hybrid, first ByteDMD measurement of SAT backtracking](https://github.com/cybertronai/SutroYaro/pull/94) (open since 2026-04-27)
- [PR #87 — ByteDMD floor-gap survey, KM-min 268 vs GF(2) 101,501](https://github.com/cybertronai/SutroYaro/pull/87) (open since 2026-04-21; floor-gap work was extended in merged PR #88)

### philoengineer
- [PR #63 — bin/review-cycle, cross-model research supervisor](https://github.com/cybertronai/SutroYaro/pull/63) (open since 2026-03-28; stale)

### Anastasia (adotzh)
- [`adotzh/SutroAna`](https://github.com/adotzh/SutroAna) — auto-research-loop framework presented at meeting #16 (04 May 26). Not currently surfaced in SutroYaro.

## GitHub state (SutroYaro)

| Type | Count | Notes |
|---|---:|---|
| Open PRs | 3 | All listed above |
| Open issues | 5 | Top: [#95](https://github.com/cybertronai/SutroYaro/issues/95) (housekeeping), [#54](https://github.com/cybertronai/SutroYaro/issues/54) (Telegram approval pipeline), [#14](https://github.com/cybertronai/SutroYaro/issues/14) (notification bridge) |
| Recent merges | 26 | Per origin/main since 2026-04-30 |

## Meeting cadence

- Meeting #16 (04 May 26): Ross Pantone (Rain AI history), Yad's hinton-problems demo, Anastasia's SutroAna auto-research loop. Notes: `docs/google-docs/sutro-group-main.md` (auto-synced).
- Meeting #17 (11 May 26, Monday): scheduled, agenda not yet posted.

## Methodology updates worth tracking

### Agent-team session-cost measurement

The hinton-problems and schmidhuber-problems builds both surfaced the "harness UI display ≠ cumulative consumption" point. For any agent-team session you want to characterize:

```python
# Aggregate token usage from JSONL session logs
import json, os, glob
from datetime import datetime, timezone
window_start = datetime(YYYY, MM, DD, HH, MM, tzinfo=timezone.utc).timestamp()
window_end   = datetime(YYYY, MM, DD, HH, MM, tzinfo=timezone.utc).timestamp()
totals = {'input_tokens': 0, 'output_tokens': 0,
          'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}
for path in glob.glob('~/.claude/projects/**/*.jsonl', recursive=True):
    if 'sutroyaro' not in path.lower(): continue  # adjust per project
    if not (window_start <= os.path.getmtime(path) <= window_end): continue
    for line in open(path):
        usage = json.loads(line).get('message', {}).get('usage', {})
        for k in totals:
            totals[k] += usage.get(k, 0)
print(totals)
```

Output gives the four buckets (input / output / cache_create / cache_read). For long supervised-agent runs, expect cache_read to dominate (>90%).

## Next bump of this doc

- Sometime around the next sutro-sync (weekly cadence)
- File name pattern: `docs/research/active-threads-YYYY-MM-DD.md`
- Or fold into a single rolling doc once the cadence stabilizes
