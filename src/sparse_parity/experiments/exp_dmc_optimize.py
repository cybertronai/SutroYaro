#!/usr/bin/env python3
"""
Experiment: DMC Optimization for Sparse Parity

Hypothesis: By reducing total floats accessed and stack distances through
(a) minimal-sample KM influence estimation and (b) shared-buffer memory
layout, we can reduce DMC below the GF2 baseline of 8,607.

Answers: Issue #22 -- Can DMC be reduced below the current best?

The key insight: DMC = sum(size * sqrt(stack_distance)). Two levers:
  1. Reduce total floats (fewer reads, smaller buffers)
  2. Reduce stack distances (read data sooner after writing it)

Approaches tested:
  A. KM-min: Use 1 influence sample (parity influence is exactly 0 or 1)
  B. KM-shared: Single shared buffer instead of per-bit buffers
  C. GF2-minimal: Use n+1=21 samples (minimum for n=20)
  D. KM-min + verify: Minimal KM with inline verification

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_dmc_optimize.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from math import sqrt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.tracker import MemTracker
from sparse_parity.config import Config


# =============================================================================
# DATA GENERATION (same protocol as harness.py)
# =============================================================================

def generate_secret(n_bits, k_sparse, seed=42):
    """Generate the secret bits (same RNG protocol as harness)."""
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())


def oracle(x, secret):
    """Compute parity label for input(s)."""
    return np.prod(x[:, secret], axis=1) if x.ndim == 2 else np.prod(x[secret])


# =============================================================================
# APPROACH A: KM with minimal samples (1 sample per bit)
# =============================================================================

def km_min_samples(n_bits, k_sparse, seed=42):
    """
    KM influence estimation with exactly 1 sample per bit.

    For parity, influence of secret bits is ALWAYS 1.0 (flipping any
    secret bit always flips the product). So 1 sample suffices for
    perfect identification with probability 1.

    Memory layout: one shared x buffer rewritten each iteration.
    """
    secret = generate_secret(n_bits, k_sparse, seed)
    rng_inf = np.random.RandomState(seed + 500)

    tracker = MemTracker()
    influences = np.zeros(n_bits)

    # Single shared buffer for x (rewritten each iteration)
    for i in range(n_bits):
        x = rng_inf.choice([-1.0, 1.0], size=(1, n_bits))

        # Write x, compute original label
        tracker.write("x", n_bits)
        y_orig = oracle(x, secret)
        tracker.read("x")

        # Flip bit i, compute new label
        x_flip = x.copy()
        x_flip[0, i] *= -1
        tracker.write("x_flip", n_bits)
        y_flip = oracle(x_flip, secret)
        tracker.read("x_flip")

        # Record influence (1 float)
        influences[i] = float(y_orig[0] != y_flip[0])

    # Top-k by influence
    top_k = sorted(np.argsort(influences)[-k_sparse:].tolist())

    # Verify on test data
    rng = np.random.RandomState(seed)
    _ = rng.choice(n_bits, k_sparse, replace=False)  # consume same RNG as generate_secret
    x_te = rng.choice([-1.0, 1.0], size=(200, n_bits))
    y_te = oracle(x_te, secret)
    y_pred = oracle(x_te, top_k)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "method": "km_min",
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
        "influence_samples": 1,
        "per_buffer": s.get("per_buffer", {}),
    }


# =============================================================================
# APPROACH B: KM with shared buffer (5 samples but single buffer)
# =============================================================================

def km_shared_buffer(n_bits, k_sparse, influence_samples=5, seed=42):
    """
    KM with shared buffer -- reuse the same buffer names across iterations
    to keep stack distances small. Compare against harness KM which uses
    per-bit buffer names (x_batch_0, x_batch_1, ...).
    """
    secret = generate_secret(n_bits, k_sparse, seed)
    rng_inf = np.random.RandomState(seed + 500)

    tracker = MemTracker()
    influences = np.zeros(n_bits)

    for i in range(n_bits):
        x_batch = rng_inf.choice([-1.0, 1.0], size=(influence_samples, n_bits))

        # Reuse same buffer names each iteration
        tracker.write("x_batch", x_batch.size)
        y_orig = oracle(x_batch, secret)
        tracker.read("x_batch")
        tracker.write("y_orig", y_orig.size)

        x_flipped = x_batch.copy()
        x_flipped[:, i] *= -1
        y_flipped = oracle(x_flipped, secret)
        tracker.write("y_flip", y_flipped.size)

        tracker.read("y_orig")
        tracker.read("y_flip")
        influences[i] = np.mean(y_orig != y_flipped)

    top_k = sorted(np.argsort(influences)[-k_sparse:].tolist())

    # Verify
    rng = np.random.RandomState(seed)
    _ = rng.choice(n_bits, k_sparse, replace=False)
    x_te = rng.choice([-1.0, 1.0], size=(200, n_bits))
    y_te = oracle(x_te, secret)
    y_pred = oracle(x_te, top_k)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "method": "km_shared",
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
        "influence_samples": influence_samples,
        "per_buffer": s.get("per_buffer", {}),
    }


# =============================================================================
# APPROACH C: GF2 with minimal samples (n+1 = 21)
# =============================================================================

def gf2_minimal(n_bits, k_sparse, seed=42):
    """
    GF2 Gaussian elimination with exactly n+1 samples and fine-grained
    tracking of the elimination steps to capture the actual memory access
    pattern.

    The harness GF2 tracking is coarse (just write A, read A, write solution).
    Here we track each row operation to measure true stack distances.
    """
    secret = generate_secret(n_bits, k_sparse, seed)
    rng = np.random.RandomState(seed)

    n_samples = n_bits + 1  # minimum needed
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)

    # Convert to GF(2)
    A = ((x + 1) / 2).astype(np.uint8)
    b = ((y + 1) / 2).astype(np.uint8)

    tracker = MemTracker()

    found_secret = None

    for b_try in [b, (1 - b).astype(np.uint8)]:
        aug = np.hstack([A.copy(), b_try.reshape(-1, 1)]).astype(np.uint8)

        # Track the augmented matrix
        tracker.write("aug", aug.size)

        pivot_cols = []
        row = 0
        for col in range(n_bits):
            tracker.read("aug")  # scan column for pivot

            found = None
            for r in range(row, len(aug)):
                if aug[r, col] == 1:
                    found = r
                    break
            if found is None:
                continue

            # Swap and eliminate
            aug[[row, found]] = aug[[found, row]]
            for r in range(len(aug)):
                if r != row and aug[r, col] == 1:
                    aug[r] = aug[r] ^ aug[row]

            tracker.write("aug", aug.size)  # matrix modified
            pivot_cols.append(col)
            row += 1

        # Extract solution
        solution = np.zeros(n_bits, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            solution[col] = aug[i, -1]
        tracker.write("solution", n_bits)

        candidate = sorted([i for i in range(n_bits) if solution[i] == 1])
        if candidate:
            y_check = np.prod(x[:, candidate], axis=1)
            if np.allclose(y_check, y):
                found_secret = candidate
                break

    if found_secret is None:
        found_secret = []

    # Verify
    x_te = rng.choice([-1.0, 1.0], size=(200, n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    if found_secret:
        y_pred = np.prod(x_te[:, found_secret], axis=1)
        accuracy = float(np.mean(y_pred == y_te))
    else:
        accuracy = 0.0

    s = tracker.summary()
    return {
        "method": "gf2_finegrained",
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": found_secret,
        "n_samples": n_samples,
        "per_buffer": s.get("per_buffer", {}),
    }


# =============================================================================
# APPROACH D: KM-1 with inline verify (absolute minimum floats)
# =============================================================================

def km_min_verify(n_bits, k_sparse, seed=42):
    """
    Minimal KM: 1 influence sample per bit, immediate top-k selection,
    then verify with just k+1 samples (barely enough to confirm).

    Goal: absolute minimum total floats accessed.
    """
    secret = generate_secret(n_bits, k_sparse, seed)
    rng_inf = np.random.RandomState(seed + 500)

    tracker = MemTracker()
    influences = np.zeros(n_bits)

    # Phase 1: influence estimation with shared single-sample buffer
    for i in range(n_bits):
        x = rng_inf.choice([-1.0, 1.0], size=(1, n_bits))

        tracker.write("x", n_bits)
        y_orig = oracle(x, secret)
        tracker.read("x")

        x[0, i] *= -1  # flip in-place
        tracker.write("x", n_bits)  # overwrite same buffer
        y_flip = oracle(x, secret)
        tracker.read("x")

        influences[i] = float(y_orig[0] != y_flip[0])

    top_k = sorted(np.argsort(influences)[-k_sparse:].tolist())

    # Phase 2: verify with minimal samples
    n_verify = k_sparse + 1  # just enough to check
    rng_v = np.random.RandomState(seed + 2000)
    x_v = rng_v.choice([-1.0, 1.0], size=(n_verify, n_bits))

    tracker.write("x_v", x_v.size)
    tracker.read("x_v")
    y_true = oracle(x_v, secret)
    y_pred_v = oracle(x_v, top_k)
    verified = bool(np.all(y_true == y_pred_v))

    # Full test accuracy
    rng = np.random.RandomState(seed)
    _ = rng.choice(n_bits, k_sparse, replace=False)
    x_te = rng.choice([-1.0, 1.0], size=(200, n_bits))
    y_te = oracle(x_te, secret)
    y_pred = oracle(x_te, top_k)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "method": "km_min_verify",
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
        "verified": verified,
        "influence_samples": 1,
        "verify_samples": n_verify,
        "per_buffer": s.get("per_buffer", {}),
    }


# =============================================================================
# APPROACH E: KM-1 in-place flip (minimize writes)
# =============================================================================

def km_inplace(n_bits, k_sparse, seed=42):
    """
    Minimal KM with in-place bit flip -- only ONE buffer, flipped and
    unflipped within the same write. Reduces total buffer writes.

    For each bit i:
      1. Write x (n_bits floats)
      2. Read x, compute y_orig
      3. Read x again (x[i] is flipped in computation only, not stored)
      4. No separate x_flip buffer needed

    Actually we can go further: write x once, read it twice with
    the flip applied logically (multiply x[i] by -1 in the product).
    """
    secret = generate_secret(n_bits, k_sparse, seed)
    rng_inf = np.random.RandomState(seed + 500)

    tracker = MemTracker()
    influences = np.zeros(n_bits)

    for i in range(n_bits):
        x = rng_inf.choice([-1.0, 1.0], size=(1, n_bits))

        tracker.write("x", n_bits)
        tracker.read("x")  # read for y_orig
        y_orig = oracle(x, secret)

        # For y_flip: we just read x again and flip bit i logically
        # (no separate write needed -- same data, different computation)
        tracker.read("x")  # read for y_flip
        x_copy = x.copy()
        x_copy[0, i] *= -1
        y_flip = oracle(x_copy, secret)

        influences[i] = float(y_orig[0] != y_flip[0])

    top_k = sorted(np.argsort(influences)[-k_sparse:].tolist())

    # Verify
    rng = np.random.RandomState(seed)
    _ = rng.choice(n_bits, k_sparse, replace=False)
    x_te = rng.choice([-1.0, 1.0], size=(200, n_bits))
    y_te = oracle(x_te, secret)
    y_pred = oracle(x_te, top_k)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "method": "km_inplace",
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
        "influence_samples": 1,
        "per_buffer": s.get("per_buffer", {}),
    }


# =============================================================================
# BASELINES: Run harness methods for comparison
# =============================================================================

def run_harness_gf2(n_bits=20, k_sparse=3, seed=42):
    """Run harness GF2 for baseline comparison."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
    from harness import measure_sparse_parity
    return measure_sparse_parity("gf2", n_bits=n_bits, k_sparse=k_sparse, seed=seed)


def run_harness_km(n_bits=20, k_sparse=3, seed=42):
    """Run harness KM for baseline comparison."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
    from harness import measure_sparse_parity
    return measure_sparse_parity("km", n_bits=n_bits, k_sparse=k_sparse, seed=seed)


# =============================================================================
# ROBUSTNESS: Test across multiple seeds
# =============================================================================

def robustness_check(method_fn, method_name, n_bits=20, k_sparse=3, seeds=None):
    """Run a method across multiple seeds to check reliability."""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]

    results = []
    for seed in seeds:
        r = method_fn(n_bits, k_sparse, seed=seed)
        results.append(r)

    n_correct = sum(1 for r in results if r["accuracy"] == 1.0)
    avg_dmc = np.mean([r["dmc"] for r in results])
    avg_ard = np.mean([r["ard"] for r in results])
    avg_floats = np.mean([r["total_floats"] for r in results])

    return {
        "method": method_name,
        "seeds_tested": len(seeds),
        "correct": n_correct,
        "avg_dmc": round(avg_dmc, 1),
        "avg_ard": round(avg_ard, 1),
        "avg_total_floats": round(avg_floats, 0),
        "all_correct": n_correct == len(seeds),
        "per_seed": results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  EXPERIMENT: DMC Optimization for Sparse Parity (Issue #22)")
    print("  Baseline: GF2 DMC=8,607 | KM DMC=20,633")
    print("=" * 70)

    n_bits = 20
    k_sparse = 3
    seed = 42
    seeds = [42, 43, 44, 45, 46]

    all_results = {}

    # -------------------------------------------------------------------
    # Baselines from harness
    # -------------------------------------------------------------------
    print("\n  [Baselines]")

    t0 = time.time()
    gf2_base = run_harness_gf2(n_bits, k_sparse, seed)
    print(f"    GF2:  acc={gf2_base['accuracy']}  ARD={gf2_base.get('ard','N/A')}  "
          f"DMC={gf2_base.get('dmc','N/A')}  floats={gf2_base.get('total_floats','N/A')}  "
          f"time={time.time()-t0:.4f}s")
    all_results['baseline_gf2'] = gf2_base

    t0 = time.time()
    km_base = run_harness_km(n_bits, k_sparse, seed)
    print(f"    KM:   acc={km_base['accuracy']}  ARD={km_base.get('ard','N/A')}  "
          f"DMC={km_base.get('dmc','N/A')}  floats={km_base.get('total_floats','N/A')}  "
          f"time={time.time()-t0:.4f}s")
    all_results['baseline_km'] = km_base

    # -------------------------------------------------------------------
    # Approach A: KM with 1 sample
    # -------------------------------------------------------------------
    print("\n  [A] KM-min (1 influence sample per bit)")
    t0 = time.time()
    r_a = km_min_samples(n_bits, k_sparse, seed)
    elapsed_a = time.time() - t0
    print(f"    acc={r_a['accuracy']}  ARD={r_a['ard']}  DMC={r_a['dmc']}  "
          f"floats={r_a['total_floats']}  time={elapsed_a:.4f}s  secret={r_a['found_secret']}")
    all_results['km_min'] = r_a

    # -------------------------------------------------------------------
    # Approach B: KM shared buffer (5 samples)
    # -------------------------------------------------------------------
    print("\n  [B] KM-shared (5 samples, shared buffer names)")
    t0 = time.time()
    r_b = km_shared_buffer(n_bits, k_sparse, influence_samples=5, seed=seed)
    elapsed_b = time.time() - t0
    print(f"    acc={r_b['accuracy']}  ARD={r_b['ard']}  DMC={r_b['dmc']}  "
          f"floats={r_b['total_floats']}  time={elapsed_b:.4f}s  secret={r_b['found_secret']}")
    all_results['km_shared'] = r_b

    # -------------------------------------------------------------------
    # Approach C: GF2 fine-grained tracking
    # -------------------------------------------------------------------
    print("\n  [C] GF2 fine-grained tracking (n+1=21 samples)")
    t0 = time.time()
    r_c = gf2_minimal(n_bits, k_sparse, seed)
    elapsed_c = time.time() - t0
    print(f"    acc={r_c['accuracy']}  ARD={r_c['ard']}  DMC={r_c['dmc']}  "
          f"floats={r_c['total_floats']}  time={elapsed_c:.4f}s  secret={r_c['found_secret']}")
    all_results['gf2_finegrained'] = r_c

    # -------------------------------------------------------------------
    # Approach D: KM-1 with verify
    # -------------------------------------------------------------------
    print("\n  [D] KM-min + verify (1 sample + k+1 verify)")
    t0 = time.time()
    r_d = km_min_verify(n_bits, k_sparse, seed)
    elapsed_d = time.time() - t0
    print(f"    acc={r_d['accuracy']}  ARD={r_d['ard']}  DMC={r_d['dmc']}  "
          f"floats={r_d['total_floats']}  time={elapsed_d:.4f}s  "
          f"verified={r_d['verified']}  secret={r_d['found_secret']}")
    all_results['km_min_verify'] = r_d

    # -------------------------------------------------------------------
    # Approach E: KM in-place (1 write, 2 reads per bit)
    # -------------------------------------------------------------------
    print("\n  [E] KM-inplace (1 write + 2 reads per bit)")
    t0 = time.time()
    r_e = km_inplace(n_bits, k_sparse, seed)
    elapsed_e = time.time() - t0
    print(f"    acc={r_e['accuracy']}  ARD={r_e['ard']}  DMC={r_e['dmc']}  "
          f"floats={r_e['total_floats']}  time={elapsed_e:.4f}s  secret={r_e['found_secret']}")
    all_results['km_inplace'] = r_e

    # -------------------------------------------------------------------
    # Robustness checks (5 seeds for best methods)
    # -------------------------------------------------------------------
    print("\n  [Robustness] Testing best methods across 5 seeds...")

    rob_inplace = robustness_check(km_inplace, "km_inplace", n_bits, k_sparse, seeds)
    print(f"    km_inplace:    {rob_inplace['correct']}/{rob_inplace['seeds_tested']} correct  "
          f"avg_DMC={rob_inplace['avg_dmc']}")
    all_results['robustness_km_inplace'] = rob_inplace

    rob_min = robustness_check(km_min_samples, "km_min", n_bits, k_sparse, seeds)
    print(f"    km_min:        {rob_min['correct']}/{rob_min['seeds_tested']} correct  "
          f"avg_DMC={rob_min['avg_dmc']}")
    all_results['robustness_km_min'] = rob_min

    rob_verify = robustness_check(km_min_verify, "km_min_verify", n_bits, k_sparse, seeds)
    print(f"    km_min_verify: {rob_verify['correct']}/{rob_verify['seeds_tested']} correct  "
          f"avg_DMC={rob_verify['avg_dmc']}")
    all_results['robustness_km_min_verify'] = rob_verify

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY: DMC OPTIMIZATION RESULTS")
    print("=" * 90)
    print(f"  {'Method':<22} | {'Acc':>5} | {'ARD':>8} | {'DMC':>10} | {'Floats':>8} | {'vs GF2':>8}")
    print(f"  {'='*22} | {'='*5} | {'='*8} | {'='*10} | {'='*8} | {'='*8}")

    baseline_dmc = 8607.4  # GF2 baseline from sweep
    methods = [
        ("GF2 baseline", gf2_base),
        ("KM baseline", km_base),
        ("KM-min (1 sample)", r_a),
        ("KM-shared (5 samp)", r_b),
        ("GF2 fine-grained", r_c),
        ("KM-min + verify", r_d),
        ("KM-inplace", r_e),
    ]

    for name, r in methods:
        dmc = r.get('dmc', 'N/A')
        ard = r.get('ard', 'N/A')
        floats = r.get('total_floats', 'N/A')
        acc = r.get('accuracy', 0)

        if isinstance(dmc, (int, float)):
            ratio = f"{dmc/baseline_dmc:.2f}x"
        else:
            ratio = "N/A"

        print(f"  {name:<22} | {acc:>5.2f} | {ard:>8} | {dmc:>10} | {floats:>8} | {ratio:>8}")

    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_dmc_optimize'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Clean up per_buffer from results for JSON (can have complex objects)
    def clean_for_json(d):
        """Remove per_buffer details for cleaner JSON output."""
        cleaned = {}
        for k, v in d.items():
            if k == 'per_buffer':
                # Summarize instead of full details
                cleaned[k] = {buf: {"size": info["size"], "read_count": info["read_count"],
                                    "avg_dist": round(info["avg_dist"], 1)}
                              for buf, info in v.items()} if isinstance(v, dict) else v
            elif isinstance(v, dict) and 'per_seed' in v:
                # For robustness results, clean each per-seed result
                cleaned[k] = v
            else:
                cleaned[k] = v
        return cleaned

    results_path = results_dir / 'results.json'
    output = {
        'experiment': 'exp_dmc_optimize',
        'issue': '#22',
        'description': 'DMC optimization: reducing Data Movement Complexity below GF2 baseline',
        'hypothesis': 'Minimal-sample KM with in-place bit flips can reduce DMC below GF2 baseline of 8,607',
        'config': {
            'n_bits': n_bits,
            'k_sparse': k_sparse,
            'seed': seed,
        },
        'baseline_gf2_dmc': baseline_dmc,
        'results': {},
    }

    for key, val in all_results.items():
        if isinstance(val, dict):
            # Remove per_buffer for cleaner output
            cleaned = {k: v for k, v in val.items() if k != 'per_buffer'}
            output['results'][key] = cleaned

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()
