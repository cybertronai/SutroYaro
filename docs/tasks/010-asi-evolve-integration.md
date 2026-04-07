# Task 10: ASI-Evolve Integration into SutroYaro Workflow

**Priority**: HIGH
**Status**: IN PROGRESS
**Agents**: Kimi 2.5 (memory), Qwen 3.6 Plus (execution), Claude/GLM-5.1 (algorithms), Gemini (synthesis)
**Source**: G B's Telegram post (2026-04-06): "Seems relevant to https://arxiv.org/abs/2603.29640"

## Context

ASI-Evolve is a closed-loop autonomous AI-for-AI research framework combining learn-design-experiment-analyze cycles with a cognition base and analyzer. We already have the building blocks: evaluation harness (`harness.py`), Gymnasium env (`SutroYaro/SparseParity-v0`), ByteDMD metric, `DISCOVERIES.md` knowledge base, Telegram/Google Docs sync. The question: can we close the loop?

## Tasks

- [ ] **Agent 1 (Systems Architect — Kimi)**: Map ASI-Evolve's "Cognition Base" and "Analyzer" onto SutroYaro's existing markdown files (`DISCOVERIES.md`, `CLAUDE.md`, Telegram sync). Produce `findings/kimi_asi_memory.md`.
- [ ] **Agent 2 (Implementation Engineer — Qwen)**: Design the orchestration to wrap `harness.py` + Gym env in an ASI-Evolve evolutionary loop. Produce `findings/qwen_asi_execution.md`.
- [ ] **Agent 3 (Algorithms Expert — Claude/GLM)**, Focus on how ASI-Evolve discovered SOTA architectures and RL algorithms. What mutation operators/search constraints would yield ByteDMD-optimal learning rules for sparse parity? Produce `findings/claude_asi_algorithms.md`.
- [ ] **Agent 4 (Synthesis — Gemini)**: Combine all 3 reports into execution plan: `docs/research/asi-evolve-integration.md`.

## References

- Paper: https://arxiv.org/abs/2603.29640
- ByteDMD metric: https://github.com/cybertronai/ByteDMD
- ByteDMD examples: https://github.com/cybertronai/ByteDMD-examples
- Wesley Smith meeting notes: [docs/google-docs/31mar26-meeting-wesley-smith.md](docs/google-docs/) (pebble game, Strassen's)
- Agent prompts: [docs/agent-prompts/asi-evolve-*](../agent-prompts/)
- DISCOVERIES.md: what's already known (Hebbian failure, GF(2), KM-min)
- AGENT.md: current autonomous loop protocol
