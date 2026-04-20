# References: run-experiment

Canonical docs to consult before and during any experiment.

| Doc | Why |
|-----|-----|
| [../../../../LAB.md](../../../../LAB.md) | Two-phase protocol, templates, rules (esp. rule 9 metric isolation) |
| [../../../../AGENT.md](../../../../AGENT.md) | Machine-executable experiment loop for autonomous sessions |
| [../../../../DISCOVERIES.md](../../../../DISCOVERIES.md) | What's proven, what failed, open questions |
| [../../../../CLAUDE.md](../../../../CLAUDE.md) | Current best methods table and SGD config |
| [../../../../research/search_space.yaml](../../../../research/search_space.yaml) | Allowed parameter ranges per challenge |
| [../../../../research/questions.yaml](../../../../research/questions.yaml) | Dependency graph of open research questions |
| [../../../../docs/research/bytedmd.md](../../../../docs/research/bytedmd.md) | ByteDMD metric (primary as of 2026-04-15) |
| [../../../../docs/research/peer-research-protocol.md](../../../../docs/research/peer-research-protocol.md) | Full autonomous-research design doc |
| [../../../rules/experiment-reproducibility.md](../../../rules/experiment-reproducibility.md) | Seeds, config dumps, environment logging |

Locked files (never modify in an experiment PR): `src/sparse_parity/harness.py`,
`tracker.py`, `cache_tracker.py`, `config.py`, `data.py`.
