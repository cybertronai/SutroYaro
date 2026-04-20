# Interactive Repo Tree

Click any node with children to expand or collapse it. Internal (directory-like) nodes are filled; leaves are hollow. Hover for the full path.

<div id="tree-viz" style="width: 100%; min-height: 640px; overflow: auto; border: 1px solid var(--md-default-fg-color--lightest); border-radius: 4px; background: var(--md-default-bg-color);"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
(function () {
  // Hardcoded tree data for SutroYaro repo.
  const treeData = {
    name: "SutroYaro",
    children: [
      {
        name: "Top-level docs",
        children: [
          { name: "README.md" },
          { name: "CLAUDE.md" },
          { name: "LAB.md" },
          { name: "AGENT.md" },
          { name: "DISCOVERIES.md" },
          { name: "TODO.md" },
          { name: "CONTRIBUTING.md" },
          { name: "AGENTS.md" },
          { name: "CODEX.md" },
          { name: "AGENT_EVAL.md" }
        ]
      },
      {
        name: "src/",
        children: [
          { name: "bytedmd/  (primary metric, vendored)" },
          {
            name: "sparse_parity/",
            children: [
              { name: "harness.py" },
              { name: "tracker.py" },
              { name: "cache_tracker.py" },
              { name: "tracked_numpy.py" },
              { name: "train.py" },
              { name: "train_fused.py" },
              { name: "train_perlayer.py" },
              { name: "fast.py" },
              { name: "model.py" },
              { name: "data.py" },
              { name: "config.py" },
              { name: "eval/" },
              { name: "experiments/" },
              { name: "telegram_sync/" }
            ]
          },
          { name: "telegram/" }
        ]
      },
      {
        name: "bin/",
        children: [
          { name: "reproduce-all" },
          { name: "run-agent" },
          { name: "analyze-log" },
          { name: "tg-sync" },
          { name: "tg-post" },
          { name: "tg-auth" },
          { name: "merge-findings" }
        ]
      },
      {
        name: "checks/",
        children: [
          { name: "env_check.py" },
          { name: "baseline_check.py" }
        ]
      },
      { name: "tests/" },
      {
        name: "research/",
        children: [
          { name: "log.jsonl  (37 experiments)" },
          { name: "questions.yaml" },
          { name: "search_space.yaml" }
        ]
      },
      { name: "findings/  (38 exp_*.md files)" },
      { name: "results/" },
      { name: "contributions/" },
      {
        name: "docs/  (mkdocs site)",
        children: [
          { name: "index.md" },
          { name: "context.md" },
          { name: "changelog.md" },
          { name: "research/" },
          { name: "findings/" },
          { name: "catchups/" },
          { name: "meetings/" },
          { name: "google-docs/" },
          { name: "agent-prompts/" },
          { name: "tasks/" },
          { name: "tooling/" },
          { name: "diagrams/" }
        ]
      }
    ]
  };

  function render() {
    const container = document.getElementById("tree-viz");
    if (!container) return;
    // Avoid double-initialization if Material re-runs scripts.
    if (container.dataset.initialized === "1") return;
    container.dataset.initialized = "1";

    const margin = { top: 20, right: 160, bottom: 20, left: 140 };
    const width = 960 - margin.left - margin.right;
    const dx = 22;       // vertical spacing per node
    const dy = 180;      // horizontal spacing per depth level
    const duration = 250;

    const root = d3.hierarchy(treeData);
    root.x0 = 0;
    root.y0 = 0;

    // Collapse everything below the top level on first render.
    root.descendants().forEach((d, i) => {
      d.id = i;
      d._children = d.children;
      if (d.depth >= 1 && d.children) d.children = null;
    });

    const svg = d3.select(container)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", 640)
      .style("font", "13px var(--md-typeface, sans-serif)")
      .style("user-select", "none");

    const gLink = svg.append("g")
      .attr("fill", "none")
      .attr("stroke", "#888")
      .attr("stroke-opacity", 0.5)
      .attr("stroke-width", 1.5);

    const gNode = svg.append("g")
      .attr("cursor", "pointer")
      .attr("pointer-events", "all");

    // Build path string for tooltip.
    function pathOf(d) {
      const parts = [];
      let cur = d;
      while (cur) {
        parts.unshift(cur.data.name);
        cur = cur.parent;
      }
      return parts.join(" / ");
    }

    function update(source) {
      const tree = d3.tree().nodeSize([dx, dy]);
      tree(root);

      const nodes = root.descendants();
      const links = root.links();

      // Compute vertical extent.
      let x0 = Infinity, x1 = -Infinity;
      root.each(d => {
        if (d.x > x1) x1 = d.x;
        if (d.x < x0) x0 = d.x;
      });
      const height = x1 - x0 + margin.top + margin.bottom + 40;

      const transition = svg.transition()
        .duration(duration)
        .attr("height", Math.max(640, height))
        .attr("viewBox", [-margin.left, x0 - margin.top, width + margin.left + margin.right, Math.max(640, height)]);

      // ---- Nodes ----
      const node = gNode.selectAll("g.node").data(nodes, d => d.id);

      const nodeEnter = node.enter().append("g")
        .attr("class", "node")
        .attr("transform", () => `translate(${source.y0},${source.x0})`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0)
        .on("click", (event, d) => {
          d.children = d.children ? null : d._children;
          update(d);
        });

      nodeEnter.append("circle")
        .attr("r", 5)
        .attr("fill", d => d._children ? "#7e57c2" : "#fff")     // purple for collapsible, hollow for leaves
        .attr("stroke", d => d._children ? "#4527a0" : "#7e57c2")
        .attr("stroke-width", 1.5);

      nodeEnter.append("text")
        .attr("dy", "0.32em")
        .attr("x", d => d._children ? -10 : 10)
        .attr("text-anchor", d => d._children ? "end" : "start")
        .attr("fill", "var(--md-default-fg-color)")
        .text(d => d.data.name)
        .clone(true).lower()
        .attr("stroke-linejoin", "round")
        .attr("stroke-width", 3)
        .attr("stroke", "var(--md-default-bg-color)");

      nodeEnter.append("title").text(d => pathOf(d));

      node.merge(nodeEnter).transition(transition)
        .attr("transform", d => `translate(${d.y},${d.x})`)
        .attr("fill-opacity", 1)
        .attr("stroke-opacity", 1);

      node.exit().transition(transition).remove()
        .attr("transform", () => `translate(${source.y},${source.x})`)
        .attr("fill-opacity", 0)
        .attr("stroke-opacity", 0);

      // Refresh circle fill on toggled nodes.
      gNode.selectAll("circle")
        .attr("fill", d => (d.children || d._children) ? (d.children ? "#fff" : "#7e57c2") : "#fff")
        .attr("stroke", d => (d.children || d._children) ? "#4527a0" : "#7e57c2");

      // ---- Links ----
      const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);

      const link = gLink.selectAll("path").data(links, d => d.target.id);

      const linkEnter = link.enter().append("path")
        .attr("d", () => {
          const o = { x: source.x0, y: source.y0 };
          return diagonal({ source: o, target: o });
        });

      link.merge(linkEnter).transition(transition)
        .attr("d", diagonal);

      link.exit().transition(transition).remove()
        .attr("d", () => {
          const o = { x: source.x, y: source.y };
          return diagonal({ source: o, target: o });
        });

      root.eachBefore(d => {
        d.x0 = d.x;
        d.y0 = d.y;
      });
    }

    update(root);
  }

  // Render now if DOM is ready, else wait. Re-runs are guarded by the dataset flag.
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
</script>

## Notes

- **Click** any node with children to expand or collapse it.
- **Filled purple circles** mark collapsed branches; **hollow circles** mark leaves or expanded internal nodes.
- The data here is a hand-curated snapshot of the top-level repo layout, not a live filesystem read. Update it when the structure changes.
