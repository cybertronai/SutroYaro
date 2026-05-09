#!/usr/bin/env python3
"""
Experiment exp_bytedmd_floor_gap: ByteDMD cost survey — how far are top methods from the floor?

Hypothesis: KM-min and GF(2) sit well above the theoretical ByteDMD minimum because
intermediate computation forces non-local memory reads. Measuring absolute ByteDMD costs
across methods establishes a baseline for the floor-gap question.

Answers: Yaroslav's question (Apr 20 2026): "how far from the floor are current solutions?"
The theoretical lower bound (~0.33, per Yaroslav) is noted for PR review; units TBD.

Note: All solve() functions are pure Python (no numpy) so ByteDMD tracks every read.
Geometric lower bound (0.3849 × bytedmd) added per Yaroslav's clarification (Apr 23):
the constant relates measured ByteDMD to the lower bound on physical VLSI allocation cost
(Gemini DeepThink proof, reviewed with Toranosuke Ozawa).

SGD is included with a deliberately tiny demo config (hidden=4, batch=4, n_train=8,
epochs=2). It does not converge — pure-Python training at any meaningful scale
exceeds practical ByteDMD trace duration. The number is included as a floor-gap
chart entry so the order-of-magnitude is visible alongside the algebraic methods.

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
FOURIER_N_SAMPLES = 50

# SGD demo config — kept tiny on purpose. Pure-Python training is slow under bytedmd
# tracing, and at this scale the network does not converge. The number is included to
# show SGD's ByteDMD profile order-of-magnitude, not to claim it solves parity.
SGD_HIDDEN = 4
SGD_BATCH = 4
SGD_N_TRAIN = 8
SGD_EPOCHS = 2
SGD_LR = 0.3

# Yaroslav's geometric lower bound: 0.3849 × bytedmd is a lower bound on the actual
# VLSI allocation cost (live-byte counting, current ByteDMD). See:
# https://github.com/cybertronai/ByteDMD/blob/dev/gemini/tarjan-detailed-part1.pdf
GEOMETRIC_LOWER_BOUND_FACTOR = 0.3849

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
# METHOD 3: WALSH-FOURIER ESTIMATOR
# =============================================================================
#
# For each k-subset S ⊆ [n], estimate the Walsh coefficient
#     χ̂_S = (1/N) Σ_j  y_j · Π_{i∈S} x_j[i]
# For noiseless sparse parity, χ̂_S = ±1 when S = secret and concentrates near 0
# for any other subset. Pick the subset with the largest |χ̂_S|.
#
# Specialized for k=3 (three nested loops over indices). For general k a recursive
# / itertools-based enumeration would be needed; the current floor-gap survey is
# fixed at k=3 so the specialization is justified.

def make_fourier_data(n_bits, secret, seed, n_samples):
    """
    Returns (flat_xs, ys):
        flat_xs[s * n_bits + i]  = i-th bit of sample s (in {-1, +1})
        ys[s]                    = parity of sample s under the true secret
    """
    rng = random.Random(seed + 3)
    flat_xs = []
    ys = []
    for _ in range(n_samples):
        x = [rng.choice([-1, 1]) for _ in range(n_bits)]
        flat_xs.extend(x)
        ys.append(_parity(x, secret))
    return flat_xs, ys


def fourier_solve(flat_xs, ys, n_bits, k_sparse):
    """
    Walsh-Fourier subset selection (k=3 specialization).
    Returns the sorted list of bit indices with the largest |χ̂_S|.
    """
    if k_sparse != 3:
        raise NotImplementedError("fourier_solve is specialized for k_sparse=3")

    n_samples = len(ys)
    best_abs = -1
    best_subset = (0, 1, 2)

    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            for k in range(j + 1, n_bits):
                total = 0
                for s in range(n_samples):
                    base = s * n_bits
                    total += ys[s] * flat_xs[base + i] * flat_xs[base + j] * flat_xs[base + k]
                abs_total = total if total >= 0 else -total
                if abs_total > best_abs:
                    best_abs = abs_total
                    best_subset = (i, j, k)

    return list(best_subset)


# =============================================================================
# METHOD 4: SGD (DEMO CONFIG)
# =============================================================================
#
# Two-layer ReLU network with MSE loss, trained pure-Python so every read is
# tracked. Config is deliberately tiny (see top-of-file comment) — accuracy is
# expected at chance level. Purpose is to put SGD on the floor-gap chart at a
# fair order of magnitude. A converged SGD would have ByteDMD several orders
# higher; this is a lower bound on "any meaningful SGD run".

def make_sgd_data(n_bits, secret, seed, n_train):
    """Returns (flat_xs, ys) where flat_xs[s*n_bits + i] is sample s bit i in {-1,+1}."""
    rng = random.Random(seed + 4)
    flat_xs = []
    ys = []
    for _ in range(n_train):
        x = [rng.choice([-1, 1]) for _ in range(n_bits)]
        flat_xs.extend(x)
        ys.append(_parity(x, secret))
    return flat_xs, ys


def make_sgd_init(n_bits, hidden, seed):
    """Tiny init — small uniform weights, zero biases. Returned as flat lists."""
    rng = random.Random(seed + 5)
    w1 = [rng.uniform(-0.3, 0.3) for _ in range(hidden * n_bits)]
    b1 = [0.0] * hidden
    w2 = [rng.uniform(-0.3, 0.3) for _ in range(hidden)]
    b2 = 0.0
    return w1, b1, w2, b2


def sgd_solve(flat_xs, ys, w1, b1, w2, b2, n_bits, hidden, n_train, batch, epochs, lr, k_sparse):
    """
    Pure-Python 2-layer MSE SGD on parity targets.

    Returns the top-k bits by absolute mean |W1[i, :]| as the predicted secret —
    the standard "saliency from first-layer weights" heuristic. Training is too
    short to converge; the prediction is mostly noise but the ByteDMD cost is
    representative of one mini training pass.
    """
    # Mutable copies of weights so we can update in place.
    w1 = list(w1)
    b1 = list(b1)
    w2 = list(w2)

    for _epoch in range(epochs):
        for batch_start in range(0, n_train, batch):
            # Accumulate gradients over the mini-batch.
            gw1 = [0.0] * (hidden * n_bits)
            gb1 = [0.0] * hidden
            gw2 = [0.0] * hidden
            gb2 = 0.0

            batch_count = 0
            for s in range(batch_start, min(batch_start + batch, n_train)):
                base = s * n_bits

                # Forward.
                h_pre = [0.0] * hidden
                for j in range(hidden):
                    acc = b1[j]
                    for i in range(n_bits):
                        acc += w1[j * n_bits + i] * flat_xs[base + i]
                    h_pre[j] = acc

                h = [v if v > 0 else 0.0 for v in h_pre]

                out = b2
                for j in range(hidden):
                    out += w2[j] * h[j]

                # Loss: 0.5 (out - y)^2 → dL/dout = (out - y)
                err = out - ys[s]

                # Backward.
                gb2 += err
                for j in range(hidden):
                    gw2[j] += err * h[j]
                    if h_pre[j] > 0:
                        d_pre = err * w2[j]
                        gb1[j] += d_pre
                        for i in range(n_bits):
                            gw1[j * n_bits + i] += d_pre * flat_xs[base + i]

                batch_count += 1

            # SGD update (mean over batch).
            scale = lr / batch_count
            for k in range(hidden * n_bits):
                w1[k] -= scale * gw1[k]
            for j in range(hidden):
                b1[j] -= scale * gb1[j]
                w2[j] -= scale * gw2[j]
            b2 -= scale * gb2

    # Saliency: rank input bits by mean |W1[:, i]| across hidden units.
    saliency = [0.0] * n_bits
    for i in range(n_bits):
        total = 0.0
        for j in range(hidden):
            v = w1[j * n_bits + i]
            total += v if v >= 0 else -v
        saliency[i] = total

    # Top-k indices by saliency.
    indexed = [(saliency[i], i) for i in range(n_bits)]
    indexed.sort(reverse=True)
    return sorted(idx for _val, idx in indexed[:k_sparse])


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

    # --- Fourier (Walsh subset selection) ---
    flat_xs, ys = make_fourier_data(N_BITS, secret, SEED, FOURIER_N_SAMPLES)
    fourier_pred = fourier_solve(flat_xs, ys, N_BITS, K_SPARSE)
    t0 = time.time()
    fourier_cost = bytedmd(fourier_solve, (flat_xs, ys, N_BITS, K_SPARSE))
    fourier_elapsed = time.time() - t0
    fourier_correct = fourier_pred == secret
    print(f"\nFourier (Walsh estimator, N={FOURIER_N_SAMPLES})")
    print(f"  Predicted: {fourier_pred}  Correct: {fourier_correct}")
    print(f"  ByteDMD: {fourier_cost:,}  ({fourier_elapsed*1000:.1f}ms to measure)")
    results["fourier"] = {
        "bytedmd": fourier_cost,
        "correct": fourier_correct,
        "n_samples": FOURIER_N_SAMPLES,
        "elapsed_ms": round(fourier_elapsed * 1000, 1),
    }

    # --- SGD (demo config — does not converge, see config note) ---
    sgd_xs, sgd_ys = make_sgd_data(N_BITS, secret, SEED, SGD_N_TRAIN)
    w1, b1, w2, b2 = make_sgd_init(N_BITS, SGD_HIDDEN, SEED)
    sgd_pred = sgd_solve(sgd_xs, sgd_ys, w1, b1, w2, b2,
                         N_BITS, SGD_HIDDEN, SGD_N_TRAIN, SGD_BATCH, SGD_EPOCHS, SGD_LR, K_SPARSE)
    t0 = time.time()
    sgd_cost = bytedmd(sgd_solve, (sgd_xs, sgd_ys, w1, b1, w2, b2,
                                   N_BITS, SGD_HIDDEN, SGD_N_TRAIN, SGD_BATCH, SGD_EPOCHS, SGD_LR, K_SPARSE))
    sgd_elapsed = time.time() - t0
    sgd_correct = sgd_pred == secret
    print(f"\nSGD demo (hidden={SGD_HIDDEN}, batch={SGD_BATCH}, n_train={SGD_N_TRAIN}, epochs={SGD_EPOCHS})")
    print(f"  Predicted: {sgd_pred}  Correct: {sgd_correct}  (chance-level expected)")
    print(f"  ByteDMD: {sgd_cost:,}  ({sgd_elapsed*1000:.1f}ms to measure)")
    results["sgd_demo"] = {
        "bytedmd": sgd_cost,
        "correct": sgd_correct,
        "config": {"hidden": SGD_HIDDEN, "batch": SGD_BATCH,
                   "n_train": SGD_N_TRAIN, "epochs": SGD_EPOCHS, "lr": SGD_LR},
        "elapsed_ms": round(sgd_elapsed * 1000, 1),
        "note": "tiny config; accuracy at chance level — number shown for floor-gap chart only",
    }

    # --- Reference bounds ---
    # Minimum cost to read k secret bits once (trivial inference lower bound)
    floor_k = sequential_read_cost(K_SPARSE)
    # Minimum cost to scan all n input bits once (information-theoretic input lower bound)
    floor_n = sequential_read_cost(N_BITS)
    print(f"\nReference bounds (sequential reads from fresh stack):")
    print(f"  Read k={K_SPARSE} bits:  {floor_k}")
    print(f"  Read n={N_BITS} bits: {floor_n}")
    print(f"  Geometric LB factor: {GEOMETRIC_LOWER_BOUND_FACTOR} × measured ByteDMD")

    # --- Summary ---
    print(f"\n{'='*78}")
    print(f"  {'Method':<14} {'ByteDMD':>12} {'vs read-n':>11} {'Geom LB':>11} {'Correct':>9}")
    print(f"  {'─'*14} {'─'*12} {'─'*11} {'─'*11} {'─'*9}")
    for name, r in results.items():
        ratio = r["bytedmd"] / floor_n
        geom_lb = r["bytedmd"] * GEOMETRIC_LOWER_BOUND_FACTOR
        r["geometric_lower_bound"] = round(geom_lb, 1)
        print(f"  {name:<14} {r['bytedmd']:>12,} {ratio:>10.1f}x {geom_lb:>11,.0f} {str(r['correct']):>9}")
    print(f"  {'─'*14} {'─'*12} {'─'*11} {'─'*11} {'─'*9}")
    print(f"  {'read-n floor':<14} {floor_n:>12,} {'1.0x':>11} {'─':>11} {'─':>9}")
    print(f"{'='*78}")

    out = {
        "experiment": "exp_bytedmd_floor_gap",
        "config": {"n_bits": N_BITS, "k_sparse": K_SPARSE, "seed": SEED,
                   "fourier_n_samples": FOURIER_N_SAMPLES},
        "results": results,
        "floor_read_k": floor_k,
        "floor_read_n": floor_n,
        "geometric_lower_bound_factor": GEOMETRIC_LOWER_BOUND_FACTOR,
        "geometric_lower_bound_source": "https://github.com/cybertronai/ByteDMD/blob/dev/gemini/tarjan-detailed-part1.pdf",
        "note": "Geometric LB requires live-byte counting (current ByteDMD post-PR #80). SGD demo is a tiny non-converging config — see results.sgd_demo.note.",
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
