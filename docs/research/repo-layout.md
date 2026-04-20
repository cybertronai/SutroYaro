# Repo Layout

The SutroYaro workspace splits into four layers: read-first docs that load agent context, source code (with the locked harness and the ByteDMD metric), machine-readable research artifacts, and the published mkdocs site. The diagram below groups directories by role rather than enumerating every leaf.

```mermaid
graph LR
    subgraph DOCS[Top-level Docs - read-first stack]
        README[README.md]
        CLAUDE[CLAUDE.md]
        LAB[LAB.md]
        AGENT[AGENT.md]
        DISC[DISCOVERIES.md]
        TODO[TODO.md]
        CONTRIB[CONTRIBUTING.md]
        TOOLS[AGENTS.md / CODEX.md / AGENT_EVAL.md]
    end

    subgraph CODE[src/ - Code]
        BYTEDMD[bytedmd/<br/>primary metric, vendored]
        SP[sparse_parity/<br/>harness, tracker, training, eval, experiments]
        TG[telegram/<br/>TS sync]
    end

    subgraph OPS[Ops + Tests]
        BIN[bin/<br/>reproduce-all, run-agent, tg-sync]
        CHECKS[checks/<br/>env_check, baseline_check]
        TESTS[tests/]
    end

    subgraph ARTIFACTS[Research Artifacts]
        RESEARCH[research/<br/>log.jsonl, questions.yaml, search_space.yaml]
        FINDINGS[findings/<br/>38 per-experiment reports]
        RESULTS[results/<br/>raw numeric outputs]
        CONTRIBDIR[contributions/<br/>external drop-zone]
    end

    subgraph SITE[docs/ - mkdocs site]
        DRESEARCH[research/<br/>findings, surveys, system overviews]
        DFINDINGS[findings/<br/>per-experiment site pages]
        DCATCH[catchups/<br/>weekly summaries]
        DMEET[meetings/, meeting-notes/, lectures/]
        DGOOGLE[google-docs/<br/>synced]
        DPROMPTS[agent-prompts/]
        DTASKS[tasks/<br/>specs 1-10 + INDEX]
        DTOOL[tooling/<br/>telegram, automation, agent compat]
        DDIAG[diagrams/<br/>SVGs]
    end

    CLAUDE -.points to.-> CODE
    LAB -.governs.-> CODE
    AGENT -.drives.-> ARTIFACTS
    CODE -.writes.-> ARTIFACTS
    ARTIFACTS -.published as.-> SITE
    DOCS -.surfaced on.-> SITE

    style DOCS fill:#fde7c4,stroke:#d4a85a,color:#000
    style CODE fill:#c8e6c9,stroke:#4a8c4f,color:#000
    style OPS fill:#dcedc8,stroke:#7a9a52,color:#000
    style ARTIFACTS fill:#bbdefb,stroke:#4a7fb8,color:#000
    style SITE fill:#e1bee7,stroke:#8e5ba0,color:#000
```

## What each layer does

- **Read-first docs** load agent context. A coding agent opens CLAUDE.md, then LAB.md or AGENT.md, then DISCOVERIES.md before touching code.
- **`src/`** holds the locked harness, the ByteDMD tracer, training code, the Gymnasium eval environment, and per-experiment scripts.
- **Ops + tests** contain reproducibility checks and the CLI scripts that orchestrate experiments and syncs.
- **Research artifacts** are machine-readable. `research/log.jsonl` is the append-only experiment log, `findings/` holds the prose writeups, `results/` holds the numbers.
- **`docs/`** is the mkdocs source for [cybertronai.github.io/SutroYaro](https://cybertronai.github.io/SutroYaro/). It mirrors much of the research output in a navigable form and adds meeting notes, weekly catchups, and reusable agent prompts.

!!! info "Active research front"
    Day-to-day method work happens in the ByteDMD repo, not here. The current experimental front is [`cybertronai/ByteDMD/experiments/grid`](https://github.com/cybertronai/ByteDMD/tree/dev/experiments/grid) (Yaroslav's self-contained experiments). This workspace is the lab notebook, contributor pipeline, and public site around that work.
