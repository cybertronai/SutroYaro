#!/usr/bin/env python3
"""
DMC visualization plots for Issue #18.

Reads results/dmc_baseline_sweep.json and generates:
  1. DMC vs ARD scatter (log-log) across all challenges
  2. DMC bar chart for sparse-parity methods
  3. Cross-challenge grouped bar comparison

Usage:
    python3 src/plot_dmc.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "results", "dmc_baseline_sweep.json")
OUT_DIR = os.path.join(ROOT_DIR, "results", "plots")


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    # Filter out entries with null ARD/DMC (failed methods)
    return [r for r in raw["results"] if r["ard"] is not None and r["dmc"] is not None]


CHALLENGE_COLORS = {
    "sparse-parity": "#2563eb",  # blue
    "sparse-sum": "#16a34a",     # green
    "sparse-and": "#dc2626",     # red
}

CHALLENGE_LABELS = {
    "sparse-parity": "Parity",
    "sparse-sum": "Sum",
    "sparse-and": "AND",
}


def plot_dmc_vs_ard(results):
    """Plot 1: DMC vs ARD scatter (log-log), colored by challenge."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot each challenge separately for legend
    for challenge, color in CHALLENGE_COLORS.items():
        pts = [r for r in results if r["challenge"] == challenge]
        if not pts:
            continue
        xs = [r["ard"] for r in pts]
        ys = [r["dmc"] for r in pts]
        labels = [r["method"].upper() for r in pts]
        ax.scatter(xs, ys, c=color, s=80, zorder=3,
                   label=CHALLENGE_LABELS[challenge], edgecolors="white", linewidths=0.5)

        # Label each point
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ARD (Average Reuse Distance)", fontsize=12)
    ax.set_ylabel("DMC (Data Movement Complexity)", fontsize=12)
    ax.set_title("DMC vs ARD: Do Rankings Agree?", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, which="major")
    ax.grid(False, which="minor")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "dmc_vs_ard.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dmc_ranking_parity(results):
    """Plot 2: DMC bar chart for sparse-parity only, sorted best at top."""
    parity = [r for r in results if r["challenge"] == "sparse-parity"]
    # Sort by DMC ascending (best = lowest at top)
    parity.sort(key=lambda r: r["dmc"])

    methods = [r["method"].upper() for r in parity]
    dmcs = [r["dmc"] for r in parity]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(methods, dmcs, color="#2563eb", edgecolor="white", height=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("DMC (Data Movement Complexity)", fontsize=12)
    ax.set_title("DMC Rankings: Sparse Parity (n=20, k=3)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # best (lowest DMC) at top

    # Add value labels
    for bar, dmc in zip(bars, dmcs):
        # Place label to the right of the bar
        ax.text(dmc * 1.3, bar.get_y() + bar.get_height() / 2,
                f"{dmc:,.0f}", va="center", fontsize=9, fontweight="bold")

    # Extend x-axis to make room for labels
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 5)

    ax.grid(True, alpha=0.2, axis="x", which="major")
    ax.grid(False, which="minor")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "dmc_ranking_parity.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dmc_cross_challenge(results):
    """Plot 3: Grouped bar chart of DMC across challenges."""
    # Collect all methods that appear in at least one challenge
    challenges = ["sparse-parity", "sparse-sum", "sparse-and"]
    all_methods = sorted({r["method"] for r in results})

    fig, ax = plt.subplots(figsize=(10, 6))

    n_challenges = len(challenges)
    bar_width = 0.22
    offsets = [(i - (n_challenges - 1) / 2) * bar_width for i in range(n_challenges)]

    x_positions = list(range(len(all_methods)))

    for i, challenge in enumerate(challenges):
        color = CHALLENGE_COLORS[challenge]
        label = CHALLENGE_LABELS[challenge]
        challenge_results = {r["method"]: r["dmc"] for r in results if r["challenge"] == challenge}

        positions = []
        heights = []
        for j, method in enumerate(all_methods):
            if method in challenge_results:
                positions.append(x_positions[j] + offsets[i])
                heights.append(challenge_results[method])

        ax.bar(positions, heights, width=bar_width, color=color, label=label,
               edgecolor="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("DMC (Data Movement Complexity)", fontsize=12)
    ax.set_title("DMC Across Challenges", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m.upper() for m in all_methods], fontsize=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis="y", which="major")
    ax.grid(False, which="minor")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "dmc_cross_challenge.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_data()
    print(f"Loaded {len(results)} entries from {DATA_PATH}")
    print()

    print("Generating plots:")
    plot_dmc_vs_ard(results)
    plot_dmc_ranking_parity(results)
    plot_dmc_cross_challenge(results)
    print()
    print("Done. All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
