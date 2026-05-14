# START HERE

Four reasons people end up in this repo. Pick yours.

## 1. "What's Sutro Group working on right now?"

- [docs/research/active-threads-2026-05-09.md](docs/research/active-threads-2026-05-09.md) — who's on what this week
- [DISCOVERIES.md](DISCOVERIES.md) — what's already proven
- [docs/tasks/INDEX.md](docs/tasks/INDEX.md) — open tasks

## 2. "I want to run a sparse-parity experiment"

Wrong repo. The submission pipeline is at [cybertronai/sparse-parity-challenge](https://github.com/cybertronai/sparse-parity-challenge). The benchmark code in `src/sparse_parity/` here is being moved out (see [#96](https://github.com/cybertronai/SutroYaro/issues/96)).

## 3. "I want to build my own multi-agent research catalog"

The pattern shipped twice: [hinton-problems](https://github.com/cybertronai/hinton-problems) (53 stubs) and [schmidhuber-problems](https://github.com/cybertronai/schmidhuber-problems) (58 stubs).

- `BUILD_NOTES.md` in either repo above — token math, agent-team pattern, lessons learned
- [.claude/skills/](.claude/skills/) — the dispatcher skills
- [docs/related-repos.md](docs/related-repos.md) — where each repo in the org fits

## 4. "I want to contribute a finding"

- [findings/_template.md](findings/_template.md) — the template
- [CONTRIBUTING.md](CONTRIBUTING.md) — full guide

## What this repo isn't

- Not the active research front. That's at [ByteDMD/experiments/grid](https://github.com/cybertronai/ByteDMD/tree/dev/experiments/grid).
- Not the home for benchmark problems. Those have dedicated repos.
- Not the cost metric. That's [ByteDMD](https://github.com/cybertronai/ByteDMD) and [simplified-dally-model](https://github.com/cybertronai/simplified-dally-model).

Longer version: [README.md](README.md) and [docs/related-repos.md](docs/related-repos.md). The reshuffle to match the file tree to this scoped role is [#96](https://github.com/cybertronai/SutroYaro/issues/96).
