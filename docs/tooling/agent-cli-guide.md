# Agent CLI Setup Guide

How to set up and run autonomous experiments with each supported AI tool.

All tools use the same protocol (AGENT.md), the same locked harness (src/harness.py), and the same log format (research/log.jsonl). The tool doesn't matter -- the protocol does.

## Prerequisites (all tools)

```bash
git clone https://github.com/0bserver07/SutroYaro.git
cd SutroYaro

# Verify environment
PYTHONPATH=src python3 checks/env_check.py

# Establish baselines on your machine
PYTHONPATH=src python3 checks/baseline_check.py
```

Both checks must pass before running experiments. If they fail, fix the issue before proceeding.

---

## Claude Code

Claude Code is a terminal-native AI agent from Anthropic. It has file read/write, bash execution, and tool permissions built in.

### Install

```bash
# macOS / Linux
npm install -g @anthropic-ai/claude-code

# Verify
claude --version
```

You need an Anthropic API key or a Claude Pro/Team subscription.

### Run experiments

```bash
# Single cycle, 10 experiments
bin/run-agent --tool claude --max 10

# Overnight: 10 cycles, 5 experiments each
bin/run-agent --tool claude --loop 10 --max 5

# With researcher ID
bin/run-agent --tool claude --max 10 --researcher yad-agent
```

### How it launches

The launcher calls `claude -p "$prompt" --allowedTools "Bash,Read,Write,Edit,Glob,Grep" --max-turns 200`. The prompt tells the agent to read AGENT.md and follow its protocol.

### Customization

Claude Code reads `CLAUDE.md` at the project root automatically. This file already contains the project context, rules, and file references. You can also use:

- **Skills/plugins**: custom slash commands (e.g., `/commit`, analysis tools)
- **MCP servers**: extend with external tools (calendar, GitHub, etc.)
- **Hooks**: scripts that run on tool calls (stop hooks, pre-commit checks)
- **AGENTS.md**: sub-agent dispatch instructions

---

## Gemini CLI

Gemini CLI is Google's open-source terminal AI agent. It has a 1M token context window, MCP server support, and a free tier.

### Install

```bash
npm install -g @google/gemini-cli

# Verify
gemini --version
```

Free tier with a personal Google account: 60 requests/min, 1,000 requests/day. Or use a Google AI Studio / Vertex AI key for more.

### Run experiments

```bash
# Single cycle
bin/run-agent --tool gemini --max 10

# Overnight
bin/run-agent --tool gemini --loop 10 --max 5
```

### How it launches

The launcher calls `gemini -p "$prompt" --yolo`. The `-p` flag runs in headless mode (no interactive UI). `--yolo` skips confirmation prompts so the agent can run bash commands without pausing.

### Customization

Gemini CLI reads `GEMINI.md` at the project root if it exists. You could create one that mirrors CLAUDE.md, or just rely on AGENT.md (which the prompt tells the agent to read).

- **MCP servers**: same protocol as Claude Code, connect external tools
- **Extensions**: custom commands via `.gemini/` config directory
- **TOML config**: per-project settings in `.gemini/settings.toml`

### Gotcha

Gemini CLI's custom slash commands don't work in headless mode yet (known issue). The agent can still read files and run bash, which is all our protocol needs.

---

## Antigravity

Google Antigravity is an agent-first IDE (like VS Code but built around AI agents). It has a Manager View for dispatching parallel agents.

### Install

Download from [antigravity.google](https://antigravity.google) or via the command line tool:

```bash
# The CLI command is 'agy'
agy .    # opens the current directory in Antigravity
```

Free for individuals in public preview.

### Run experiments (manual workflow)

Antigravity is an IDE, not a headless CLI. It cannot be driven by `bin/run-agent`. Instead:

1. Open the project: `agy .`
2. In the agent chat, type: "Read AGENT.md. Follow its protocol exactly. Your researcher ID is [your-name]."
3. The agent will read the files, pick hypotheses from TODO.md, run experiments, and log results.

### Parallel agents via Manager View

Antigravity's unique feature is Manager View, where you can dispatch multiple agents in parallel:

1. Open Manager View
2. Create Agent 1: "Read AGENT.md. Test the first unchecked hypothesis in TODO.md."
3. Create Agent 2: "Read AGENT.md. Test the second unchecked hypothesis in TODO.md."
4. Both agents work simultaneously on different hypotheses

This is similar to Claude Code's sub-agent dispatch but with a visual interface.

### Customization

- **Multi-model**: supports Claude Sonnet/Opus, Gemini, and GPT models
- **Artifacts**: agents produce visible task lists, plans, and screenshots
- **No MCP yet**: MCP server support is not available as of early 2026

---

## Replit Agent

Replit's AI agent runs in a web IDE with its own execution environment.

### Run experiments (manual workflow)

1. Fork the repo into Replit
2. In the agent chat: "Read AGENT.md. Follow its protocol. Your researcher ID is [your-name]."
3. The agent reads files, runs experiments, logs results.

### Known issues

Germain's experience (Meeting #8): Replit agents tried to rewrite the ARD measurement code to inflate scores instead of improving the algorithm. This is why metric isolation (LAB.md rule #9) exists. Make sure to tell the agent explicitly: "Do NOT modify harness.py, tracker.py, or any measurement code."

---

## Custom CLI

Any AI tool that can accept a prompt and execute commands can be used.

```bash
# Set AI_CMD to your tool's command
AI_CMD="my-ai-tool --prompt" bin/run-agent --tool custom --max 10
```

The launcher pipes the prompt to `$AI_CMD` via stdin. Your tool needs to:

1. Accept a text prompt
2. Be able to read/write files in the working directory
3. Be able to run `python3` commands
4. Write output to stdout (for logging)

---

## After any run

```bash
# Text report
bin/analyze-log

# Generate progress chart
bin/analyze-log --plot

# Merge your results for PR submission
bin/merge-findings research/log.jsonl --scoreboard
```

## Comparing tools

| Feature | Claude Code | Gemini CLI | Antigravity | Replit |
|---------|------------|-----------|-------------|--------|
| Headless mode | Yes (`-p`) | Yes (`-p --yolo`) | No (IDE) | No (web) |
| `bin/run-agent` | Yes | Yes | No | No |
| Looped overnight | Yes | Yes | No | No |
| MCP servers | Yes | Yes | No | No |
| Custom skills | Yes | Yes (extensions) | No | No |
| Context window | 200K (Opus: 1M) | 1M | Varies by model | Varies |
| Cost | API key or subscription | Free tier available | Free preview | Subscription |
| Metric isolation risk | Low (rules enforced) | Low | Medium | High (Germain's experience) |
