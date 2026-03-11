# What's New (March 2026)

Quick summary for the group. If you want to run experiments with your own AI tool, start here.

## What changed

We built autonomous research infrastructure that any AI tool can use. Before this, experiments were run manually and results lived in scattered session docs. Now there's a protocol, a locked harness, and a machine-readable log.

## The 30-second version

1. A locked evaluation harness (`src/harness.py`) measures ARD, DMC, and timing the same way for everyone
2. An agent protocol (`AGENT.md`) tells any AI tool how to run experiments
3. A bash launcher (`bin/run-agent`) works with Claude Code, Gemini CLI, Codex CLI, OpenCode, or any CLI
4. All 33 experiments are in a machine-readable log (`research/log.jsonl`)
5. A progress report and chart are one command away (`bin/analyze-log`)

## How to try it

```bash
# Pull the latest
git pull

# Check your environment
PYTHONPATH=src python3 checks/env_check.py
PYTHONPATH=src python3 checks/baseline_check.py

# See the 33-experiment progress report
bin/analyze-log

# Run one experiment with your tool of choice
bin/run-agent --tool gemini --max 1
bin/run-agent --tool codex --max 1
bin/run-agent --tool claude --max 1

# Or overnight (10 cycles, 5 experiments each)
bin/run-agent --tool gemini --loop 10 --max 5
```

No special setup beyond having your AI CLI installed and authenticated.

## What tool should I use?

Any of them. The protocol is the same regardless. Pick whatever you already have:

| Tool | Install | Run |
|------|---------|-----|
| Claude Code | `npm i -g @anthropic-ai/claude-code` | `bin/run-agent --tool claude` |
| Gemini CLI | `npm i -g @google/gemini-cli` | `bin/run-agent --tool gemini` |
| Codex CLI | `npm i -g @openai/codex` | `bin/run-agent --tool codex` |
| OpenCode | `brew install opencode` | `bin/run-agent --tool opencode` |
| Antigravity | Download from antigravity.google | Open project, tell agent "Read AGENT.md" |

Full setup details: [Agent CLI Guide](../tooling/agent-cli-guide.md)

## How results get shared

Each researcher runs experiments locally and produces log entries in `research/log.jsonl`. Submit a PR to merge your results into the shared log. The harness ensures everyone's numbers are comparable.

```bash
# After your run, check your results
bin/analyze-log

# Submit via PR -- your log entries merge into the shared log
```

## Key pages

- [Agent CLI Guide](../tooling/agent-cli-guide.md) -- setup for each tool
- [Peer Research Protocol](peer-research-protocol.md) -- full design, nanoGPT proposal
- [Research as Navigation](navigation-thesis.md) -- why we built it this way
- [Practitioner's Field Guide](survey.md) -- all 33 experiments ranked

## For Friday's demo

The demo flow:

1. `PYTHONPATH=src python3 src/harness.py --method gf2 --json` -- the locked harness in action
2. `bin/analyze-log` -- 33 experiments, 36.4% win rate, best ARD = 1 (RL)
3. `bin/analyze-log --plot` then `open results/progress.png` -- the ARD progress chart
4. `bin/run-agent --help` -- show 5 supported tools
5. Walk through `AGENT.md` -- the protocol any AI follows

The pitch: "The tool doesn't matter. The protocol does. Any AI CLI can follow it. Results accumulate across everyone's runs."
