#!/usr/bin/env python3
"""
Experiment exp_km_sat_hybrid: KM-min + SAT hybrid vs pure SAT backtracking under ByteDMD

Hypothesis: KM-min + SAT hybrid achieves near-KM-min ByteDMD cost by shrinking the SAT
search space from C(n,k) to C(k',k) ≈ 1, while pure SAT backtracking scales badly with n
and k due to non-local reads across the full candidate matrix.

Note on oracle-query model: KM-min here uses pre-paired samples (not a fixed random
dataset), so this experiment is an internal ByteDMD measurement, not a sparse-parity-challenge
submission. The challenge requires a fixed random dataset; converting KM-min to that
interface would require a Walsh-Fourier estimator (see exp_fourier.py). This experiment
lives in the research journal as a study of memory-access patterns under ByteDMD.

Note on SAT at n=40/k=5: C(40,5)=658,008 subsets × 41 samples × 5 muls ≈ 135M Python ops
under bytedmd tracing — intractable in reasonable time. Pure SAT at n=40/k=5 is timed
without bytedmd and marked accordingly. The hybrid at n=40/k=5 is fully traced (Phase 2
checks only C(5,5)=1 subset).

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_km_sat_hybrid.py
"""

import math
import time
import json
import random
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bytedmd import bytedmd

GEOMETRIC_LB_FACTOR = 0.3849
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "exp_km_sat_hybrid"

CONFIGS = [
    {"n_bits": 20, "k_sparse": 3, "n_sat_samples": 21,  "trace_sat": True},
    {"n_bits": 40, "k_sparse": 5, "n_sat_samples": 41,  "trace_sat": False},
]
SEED = 42


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def _generate_secret(n_bits, k_sparse, seed):
    rng = random.Random(seed)
    bits = list(range(n_bits))
    rng.shuffle(bits)
    return sorted(bits[:k_sparse])


def _parity(x, secret):
    result = 1
    for i in secret:
        result *= x[i]
    return result


# =============================================================================
# DATA GENERATION (outside bytedmd)
# =============================================================================

def make_km_labels(n_bits, secret, seed):
    """
    Flat list of 2*n_bits paired labels [y_0, yf_0, ..., y_{n-1}, yf_{n-1}].
    Bit i has influence iff y_i != yf_i (exact for noiseless parity).
    """
    rng = random.Random(seed + 1)
    labels = []
    for i in range(n_bits):
        x = [rng.choice([-1, 1]) for _ in range(n_bits)]
        labels.append(_parity(x, secret))
        x[i] = -x[i]
        labels.append(_parity(x, secret))
    return labels


def make_sat_data(n_bits, secret, n_samples, seed):
    """
    Returns (x_data, y_data):
        x_data: list of n_samples lists of n_bits +1/-1 ints
        y_data: list of n_samples +1/-1 ints
    """
    rng = random.Random(seed + 2)
    x_data, y_data = [], []
    for _ in range(n_samples):
        x = [rng.choice([-1, 1]) for _ in range(n_bits)]
        x_data.append(x)
        y_data.append(_parity(x, secret))
    return x_data, y_data


# =============================================================================
# METHOD 1: KM-MIN (reference, reused from exp_bytedmd_floor_gap)
# =============================================================================

def km_min_solve(labels, n_bits, k_sparse):
    """Sequential paired-label scan. Exits early once k candidates found."""
    secret_pred = []
    for i in range(n_bits):
        if labels[2 * i] != labels[2 * i + 1]:
            secret_pred.append(i)
        if len(secret_pred) == k_sparse:
            break
    return secret_pred


# =============================================================================
# METHOD 2: PURE SAT BACKTRACKING
# =============================================================================
#
# Pure Python port of exp_smt.py's backtrack_solve(). No numpy, no Z3.
# Iterates over k-subsets in lexicographic order, prunes on first inconsistent sample.

def _check_subset(x_data, y_data, subset):
    """Return True if subset is consistent with all (x, y) samples."""
    for x, y in zip(x_data, y_data):
        prod = 1
        for i in subset:
            prod *= x[i]
        if prod != y:
            return False
    return True


def sat_backtrack_solve(x_data, y_data, n_bits, k_sparse):
    """
    Backtracking search over all C(n_bits, k_sparse) subsets.
    Returns first subset consistent with all samples.
    """
    def backtrack(chosen, start):
        if len(chosen) == k_sparse:
            if _check_subset(x_data, y_data, chosen):
                return list(chosen)
            return None
        remaining = k_sparse - len(chosen)
        for idx in range(start, n_bits - remaining + 1):
            chosen.append(idx)
            # Early prune: check partial consistency on first sample only
            prod = 1
            for i in chosen:
                prod *= x_data[0][i]
            if len(chosen) < k_sparse or prod == y_data[0]:
                result = backtrack(chosen, idx + 1)
                if result is not None:
                    return result
            chosen.pop()
        return None

    found = backtrack([], 0)
    return sorted(found) if found else []


# =============================================================================
# METHOD 3: KM-MIN + SAT HYBRID
# =============================================================================
#
# Phase 1: KM influence probe → k' candidate bits (= k for noiseless parity).
# Phase 2: SAT backtracking over C(k', k) subsets of candidates only.
# For noiseless parity with 1 sample/bit, Phase 1 is exact → Phase 2 checks 1 subset.

def km_sat_hybrid_solve(km_labels, x_data, y_data, n_bits, k_sparse):
    """
    Phase 1: KM-min identifies candidate bits.
    Phase 2: SAT verifies the unique consistent subset among candidates.
    """
    # Phase 1: influence probe
    candidates = km_min_solve(km_labels, n_bits, k_sparse)

    # Phase 2: SAT over reduced candidate set
    for subset in combinations(candidates, k_sparse):
        if _check_subset(x_data, y_data, list(subset)):
            return sorted(subset)

    # Fallback: if KM returned wrong candidates, expand to full SAT
    return sat_backtrack_solve(x_data, y_data, n_bits, k_sparse)


# =============================================================================
# MAIN
# =============================================================================

def run_config(cfg, secret):
    n_bits    = cfg["n_bits"]
    k_sparse  = cfg["k_sparse"]
    n_samples = cfg["n_sat_samples"]
    trace_sat = cfg["trace_sat"]
    label     = f"n={n_bits}/k={k_sparse}"

    km_labels          = make_km_labels(n_bits, secret, SEED)
    x_data, y_data     = make_sat_data(n_bits, secret, n_samples, SEED)

    print(f"\n{'='*60}")
    print(f"  Config: {label}  secret={secret}")
    print(f"  SAT search space: C({n_bits},{k_sparse}) = {math.comb(n_bits, k_sparse):,} subsets")
    print(f"{'='*60}")

    results = {"label": label, "n_bits": n_bits, "k_sparse": k_sparse,
               "n_sat_samples": n_samples, "secret": secret}

    # --- KM-min ---
    t0 = time.time()
    km_cost = bytedmd(km_min_solve, (km_labels, n_bits, k_sparse))
    km_elapsed = time.time() - t0
    km_pred = km_min_solve(km_labels, n_bits, k_sparse)
    print(f"\n  KM-min")
    print(f"    Predicted: {km_pred}  Correct: {km_pred == secret}")
    print(f"    ByteDMD: {km_cost:,}  geo_LB: {km_cost * GEOMETRIC_LB_FACTOR:,.0f}")
    results["km_min"] = {"bytedmd": km_cost, "correct": km_pred == secret,
                         "geo_lb": round(km_cost * GEOMETRIC_LB_FACTOR, 1),
                         "elapsed_ms": round((time.time() - t0) * 1000, 1)}

    # --- Pure SAT backtracking ---
    if trace_sat:
        t0 = time.time()
        sat_cost = bytedmd(sat_backtrack_solve, (x_data, y_data, n_bits, k_sparse))
        sat_elapsed = time.time() - t0
        sat_pred = sat_backtrack_solve(x_data, y_data, n_bits, k_sparse)
        print(f"\n  Pure SAT backtracking")
        print(f"    Predicted: {sat_pred}  Correct: {sat_pred == secret}")
        print(f"    ByteDMD: {sat_cost:,}  geo_LB: {sat_cost * GEOMETRIC_LB_FACTOR:,.0f}")
        results["sat"] = {"bytedmd": sat_cost, "correct": sat_pred == secret,
                          "geo_lb": round(sat_cost * GEOMETRIC_LB_FACTOR, 1),
                          "elapsed_ms": round(sat_elapsed * 1000, 1)}
    else:
        t0 = time.time()
        sat_pred = sat_backtrack_solve(x_data, y_data, n_bits, k_sparse)
        sat_elapsed = time.time() - t0
        print(f"\n  Pure SAT backtracking (timed only — trace intractable at this scale)")
        print(f"    Predicted: {sat_pred}  Correct: {sat_pred == secret}")
        print(f"    ByteDMD: untraceable  elapsed: {sat_elapsed*1000:.0f}ms")
        results["sat"] = {"bytedmd": None, "correct": sat_pred == secret,
                          "note": "trace skipped — C(40,5)=658,008 subsets exceeds trace budget",
                          "elapsed_ms": round(sat_elapsed * 1000, 1)}

    # --- KM + SAT hybrid ---
    t0 = time.time()
    hyb_cost = bytedmd(km_sat_hybrid_solve, (km_labels, x_data, y_data, n_bits, k_sparse))
    hyb_elapsed = time.time() - t0
    hyb_pred = km_sat_hybrid_solve(km_labels, x_data, y_data, n_bits, k_sparse)
    print(f"\n  KM + SAT hybrid")
    print(f"    Predicted: {hyb_pred}  Correct: {hyb_pred == secret}")
    print(f"    ByteDMD: {hyb_cost:,}  geo_LB: {hyb_cost * GEOMETRIC_LB_FACTOR:,.0f}")
    results["km_sat_hybrid"] = {"bytedmd": hyb_cost, "correct": hyb_pred == secret,
                                "geo_lb": round(hyb_cost * GEOMETRIC_LB_FACTOR, 1),
                                "elapsed_ms": round(hyb_elapsed * 1000, 1)}

    # --- Summary table ---
    print(f"\n  {'Method':<22} {'ByteDMD':>12} {'geo_LB':>12} {'Correct':>8}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*8}")
    for name, r in results.items():
        if not isinstance(r, dict) or "bytedmd" not in r:
            continue
        bd = r["bytedmd"]
        bd_str = f"{bd:,}" if bd is not None else "n/a"
        glb_str = f"{r['geo_lb']:,.0f}" if bd is not None else "n/a"
        print(f"  {name:<22} {bd_str:>12} {glb_str:>12} {str(r['correct']):>8}")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for cfg in CONFIGS:
        secret = _generate_secret(cfg["n_bits"], cfg["k_sparse"], SEED)
        all_results[cfg["label"] if "label" in cfg else f"n{cfg['n_bits']}_k{cfg['k_sparse']}"] = \
            run_config(cfg, secret)

    out = {
        "experiment": "exp_km_sat_hybrid",
        "note": (
            "Oracle-query model: KM-min uses pre-paired samples, not a fixed random dataset. "
            "Internal ByteDMD measurement only — not a sparse-parity-challenge submission. "
            "See ~/dev/research/sutro/km+smt.md for full design rationale."
        ),
        "geometric_lb_factor": GEOMETRIC_LB_FACTOR,
        "configs": all_results,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
