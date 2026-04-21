#!/usr/bin/env python3
"""
Experiment exp_bytedmd_floor_gap: ByteDMD cost survey — how far are top methods from the floor?

Hypothesis: KM-min and GF(2) sit well above the theoretical ByteDMD minimum because
intermediate computation forces non-local memory reads. Measuring absolute ByteDMD costs
across methods establishes a baseline for the floor-gap question.

Answers: Yaroslav's question (Apr 20 2026): "how far from the floor are current solutions?"
The theoretical lower bound (~0.33, per Yaroslav) is noted for PR review; units TBD.

Note: Both solve() functions are pure Python (no numpy) so ByteDMD tracks every read.
SGD omitted — pure Python training at this scale would take minutes and the result
(worse than GF(2)) is already known from prior experiments.

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_bytedmd_floor_gap.py
"""

import math
import time
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bytedmd import bytedmd

N_BITS = 20
K_SPARSE = 3
SEED = 42

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "exp_bytedmd_floor_gap"


# =============================================================================
# SHARED ORACLE (used only during data generation, outside bytedmd)
# =============================================================================

def _parity(x, secret):
    result = 1
    for i in secret:
        result *= x[i]
    return result


def _generate_secret(n_bits, k_sparse, seed):
    rng = random.Random(seed)
    bits = list(range(n_bits))
    rng.shuffle(bits)
    return sorted(bits[:k_sparse])


# =============================================================================
# METHOD 1: KM-MIN (1 influence sample per bit)
# =============================================================================
#
# For each bit i, generate one paired sample (x, x_with_bit_i_flipped).
# Flipping bit i changes the label iff i is in the secret (exact for noiseless parity).
# Only the labels are passed into bytedmd — the x values are irrelevant to the solve step.

def make_km_labels(n_bits, secret, seed):
    """
    Returns flat list length 2*n_bits:
        [y_0, yf_0, y_1, yf_1, ..., y_{n-1}, yf_{n-1}]
    y_i  = parity of a random sample
    yf_i = parity of that sample with bit i flipped
    """
    rng = random.Random(seed + 1)
    labels = []
    for i in range(n_bits):
        x = [rng.choice([-1, 1]) for _ in range(n_bits)]
        labels.append(_parity(x, secret))
        x[i] = -x[i]
        labels.append(_parity(x, secret))
    return labels


def km_min_solve(labels, n_bits, k_sparse):
    """
    Read paired labels sequentially. Bit i is in secret iff labels[2i] != labels[2i+1].
    Stops as soon as k candidates are found (early exit on ordered scan).
    """
    secret_pred = []
    for i in range(n_bits):
        if labels[2 * i] != labels[2 * i + 1]:
            secret_pred.append(i)
        if len(secret_pred) == k_sparse:
            break
    return secret_pred


# =============================================================================
# METHOD 2: GF(2) GAUSSIAN ELIMINATION
# =============================================================================
#
# Build augmented matrix [A | b] over GF(2) from n_bits+1 samples.
# Row-reduce in place; solution bits are pivot columns where reduced b = 1.

def make_gf2_rows(n_bits, secret, seed):
    """
    Returns list of (n_bits+1) rows, each = n_bits bits (0/1) + GF(2) label.
    Label = XOR of secret input bits = sum(x_bin[secret]) mod 2.
    """
    rng = random.Random(seed + 2)
    rows = []
    for _ in range(n_bits + 1):
        x_bin = [rng.randint(0, 1) for _ in range(n_bits)]
        y = sum(x_bin[i] for i in secret) % 2
        rows.append(x_bin + [y])
    return rows


def gf2_solve(rows, n_bits):
    """
    Gaussian elimination over GF(2) on augmented matrix [A | b].
    Returns sorted list of pivot columns where reduced b = 1 (the secret bits).
    """
    m = [list(r) for r in rows]
    n_rows = len(m)
    pivot_row = 0
    pivot_cols = []

    for col in range(n_bits):
        found = -1
        for r in range(pivot_row, n_rows):
            if m[r][col]:
                found = r
                break
        if found == -1:
            continue
        m[pivot_row], m[found] = m[found], m[pivot_row]
        pivot_cols.append(col)
        for r in range(n_rows):
            if r != pivot_row and m[r][col]:
                for c in range(n_bits + 1):
                    m[r][c] ^= m[pivot_row][c]
        pivot_row += 1

    secret_pred = []
    for idx, col in enumerate(pivot_cols):
        if idx < n_rows and m[idx][n_bits] == 1:
            secret_pred.append(col)

    return sorted(secret_pred)


# =============================================================================
# LOWER BOUND REFERENCE
# =============================================================================

def sequential_read_cost(n):
    """ByteDMD cost of reading n values sequentially from the top of a fresh stack."""
    return sum(math.ceil(math.sqrt(i + 1)) for i in range(n))


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    secret = _generate_secret(N_BITS, K_SPARSE, SEED)
    print(f"n={N_BITS}, k={K_SPARSE}, secret={secret}\n")

    results = {}

    # --- KM-min ---
    km_labels = make_km_labels(N_BITS, secret, SEED)
    km_pred = km_min_solve(km_labels, N_BITS, K_SPARSE)
    t0 = time.time()
    km_cost = bytedmd(km_min_solve, (km_labels, N_BITS, K_SPARSE))
    km_elapsed = time.time() - t0
    km_correct = km_pred == secret
    print(f"KM-min (1 sample/bit)")
    print(f"  Predicted: {km_pred}  Correct: {km_correct}")
    print(f"  ByteDMD: {km_cost:,}  ({km_elapsed*1000:.1f}ms to measure)")
    results["km_min"] = {
        "bytedmd": km_cost,
        "correct": km_correct,
        "n_samples": 2 * N_BITS,
        "elapsed_ms": round(km_elapsed * 1000, 1),
    }

    # --- GF(2) ---
    gf2_rows = make_gf2_rows(N_BITS, secret, SEED)
    gf2_pred = gf2_solve(gf2_rows, N_BITS)
    t0 = time.time()
    gf2_cost = bytedmd(gf2_solve, (gf2_rows, N_BITS))
    gf2_elapsed = time.time() - t0
    gf2_correct = gf2_pred == secret
    print(f"\nGF(2) Gaussian Elimination")
    print(f"  Predicted: {gf2_pred}  Correct: {gf2_correct}")
    print(f"  ByteDMD: {gf2_cost:,}  ({gf2_elapsed*1000:.1f}ms to measure)")
    results["gf2"] = {
        "bytedmd": gf2_cost,
        "correct": gf2_correct,
        "n_samples": N_BITS + 1,
        "elapsed_ms": round(gf2_elapsed * 1000, 1),
    }

    # --- Reference bounds ---
    # Minimum cost to read k secret bits once (trivial inference lower bound)
    floor_k = sequential_read_cost(K_SPARSE)
    # Minimum cost to scan all n input bits once (information-theoretic input lower bound)
    floor_n = sequential_read_cost(N_BITS)
    print(f"\nReference bounds (sequential reads from fresh stack):")
    print(f"  Read k={K_SPARSE} bits:  {floor_k}")
    print(f"  Read n={N_BITS} bits: {floor_n}")
    print(f"  Yaroslav's geometric floor: ~0.33 (units TBD — see PR review)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  {'Method':<20} {'ByteDMD':>10} {'vs read-n':>10} {'Correct':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8}")
    for name, r in results.items():
        ratio = r["bytedmd"] / floor_n
        print(f"  {name:<20} {r['bytedmd']:>10,} {ratio:>9.1f}x {str(r['correct']):>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'read-n floor':<20} {floor_n:>10,} {'1.0x':>10} {'─':>8}")
    print(f"{'='*60}")

    out = {
        "experiment": "exp_bytedmd_floor_gap",
        "config": {"n_bits": N_BITS, "k_sparse": K_SPARSE, "seed": SEED},
        "results": results,
        "floor_read_k": floor_k,
        "floor_read_n": floor_n,
        "note": "Yaroslav's geometric lower bound ~0.33 not yet incorporated — units TBD, see PR review",
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
