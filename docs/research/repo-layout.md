# Repo Layout

The SutroYaro workspace splits into four layers: read-first docs that load agent context, source code (with the locked harness and the ByteDMD metric), machine-readable research artifacts, and the published mkdocs site. The diagram below groups directories by role rather than enumerating every leaf.

<div id="repo-layout-wrap" markdown="1" style="position: relative;">
<div id="repo-layout-controls" style="position: absolute; top: 8px; right: 8px; z-index: 10; display: none; gap: 4px; font-family: var(--md-typeface, sans-serif);">
<button data-mpz="in" title="Zoom in" style="width: 32px; height: 32px; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 4px; background: var(--md-default-bg-color); color: var(--md-default-fg-color); cursor: pointer; font-size: 16px; line-height: 1;">＋</button>
<button data-mpz="out" title="Zoom out" style="width: 32px; height: 32px; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 4px; background: var(--md-default-bg-color); color: var(--md-default-fg-color); cursor: pointer; font-size: 16px; line-height: 1;">−</button>
<button data-mpz="reset" title="Reset zoom" style="width: 32px; height: 32px; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 4px; background: var(--md-default-bg-color); color: var(--md-default-fg-color); cursor: pointer; font-size: 14px; line-height: 1;">⤾</button>
</div>

<!-- BEGIN_AUTOGEN mermaidLayout (regenerate via bin/regen-diagrams) -->
```mermaid
graph LR
    subgraph DOCS["Top-level Docs - read-first stack"]
        README["README.md"]
        CLAUDE["CLAUDE.md"]
        LAB["LAB.md"]
        AGENT["AGENT.md"]
        DISC["DISCOVERIES.md"]
        TODO["TODO.md"]
        CONTRIB["CONTRIBUTING.md"]
        TOOLS["AGENTS.md / CODEX.md / GEMINI.md / AGENT_EVAL.md"]
    end

    subgraph CODE["src/ - Code"]
        BYTEDMD["bytedmd/<br/>primary metric, vendored"]
        SP["sparse_parity/<br/>harness, tracker, training, eval, experiments, challenges"]
        TG["telegram/<br/>TS sync"]
    end

    subgraph OPS["Ops + Tests"]
        BIN["bin/<br/>reproduce-all, run-agent, tg-sync, regen-diagrams"]
        CHECKS["checks/<br/>env_check, baseline_check"]
        TESTS["tests/"]
    end

    subgraph ARTIFACTS["Research Artifacts"]
        RESEARCH["research/<br/>log.jsonl (37 entries), questions.yaml, search_space.yaml"]
        FINDINGS["findings/<br/>41 per-experiment reports"]
        RESULTS["results/<br/>raw numeric outputs"]
        CONTRIBDIR["contributions/<br/>external drop-zone"]
    end

    subgraph SITE["docs/ - mkdocs site"]
        DRESEARCH["research/<br/>findings, surveys, system overviews"]
        DFINDINGS["findings/<br/>per-experiment site pages"]
        DCATCH["catchups/<br/>weekly summaries"]
        DMEET["meetings/, meeting-notes/, lectures/"]
        DGOOGLE["google-docs/<br/>synced"]
        DPROMPTS["agent-prompts/"]
        DTASKS["tasks/<br/>11 task specs + INDEX"]
        DTOOL["tooling/<br/>telegram, automation, agent compat"]
        DDIAG["diagrams/<br/>SVGs"]
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
<!-- END_AUTOGEN mermaidLayout -->

</div>

<script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
<script>
(function attachPanZoom() {
  function init() {
    const wrap = document.getElementById("repo-layout-wrap");
    if (!wrap) return;
    const svg = wrap.querySelector(".mermaid svg");
    if (!svg || typeof svgPanZoom === "undefined") {
      // Mermaid may not have rendered yet; retry a few times.
      return false;
    }
    if (wrap.dataset.pzInit === "1") return true;
    wrap.dataset.pzInit = "1";

    // Mermaid sets max-width which blocks pan-zoom sizing; unlock it.
    svg.style.maxWidth = "100%";
    svg.style.height = "auto";
    svg.setAttribute("width", "100%");

    const pz = svgPanZoom(svg, {
      zoomEnabled: true,
      controlIconsEnabled: false,
      fit: true,
      center: true,
      minZoom: 0.3,
      maxZoom: 4,
      mouseWheelZoomEnabled: true,
      dblClickZoomEnabled: false
    });

    const controls = document.getElementById("repo-layout-controls");
    if (controls) {
      controls.style.display = "flex";
      controls.querySelector('[data-mpz="in"]').addEventListener("click", () => pz.zoomBy(1.3));
      controls.querySelector('[data-mpz="out"]').addEventListener("click", () => pz.zoomBy(1 / 1.3));
      controls.querySelector('[data-mpz="reset"]').addEventListener("click", () => { pz.resetZoom(); pz.center(); pz.fit(); });
    }
    return true;
  }

  // Retry until Mermaid finishes rendering.
  let attempts = 0;
  const id = setInterval(() => {
    attempts++;
    if (init() || attempts > 40) clearInterval(id);
  }, 150);
})();
</script>

## What each layer does

- **Read-first docs** load agent context. A coding agent opens CLAUDE.md, then LAB.md or AGENT.md, then DISCOVERIES.md before touching code.
- **`src/`** holds the locked harness, the ByteDMD tracer, training code, the Gymnasium eval environment, and per-experiment scripts.
- **Ops + tests** contain reproducibility checks and the CLI scripts that orchestrate experiments and syncs.
- **Research artifacts** are machine-readable. `research/log.jsonl` is the append-only experiment log, `findings/` holds the prose writeups, `results/` holds the numbers.
- **`docs/`** is the mkdocs source for [cybertronai.github.io/SutroYaro](https://cybertronai.github.io/SutroYaro/). It mirrors much of the research output in a navigable form and adds meeting notes, weekly catchups, and reusable agent prompts.

!!! info "Active research front"
    Day-to-day method work happens in the ByteDMD repo, not here. The current experimental front is [`cybertronai/ByteDMD/experiments/grid`](https://github.com/cybertronai/ByteDMD/tree/dev/experiments/grid) (Yaroslav's self-contained experiments). This workspace is the lab notebook, contributor pipeline, and public site around that work.
