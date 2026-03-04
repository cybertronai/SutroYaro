#!/usr/bin/env python3
"""
Experiment Curriculum: Curriculum learning for scaling sparse parity.

Hypothesis: Training on smaller n first, then expanding the input dimension,
transfers the learned feature detector and reduces epochs-to-solve at scale.

Answers: Open question #3 from DISCOVERIES.md —
"Can curriculum learning help at scale? Train on easy configs (small n) first,
then increase n. Transfer the learned feature detector."

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_curriculum.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# =============================================================================
# CONFIG
# =============================================================================

EXP_NAME = "exp_curriculum"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / EXP_NAME

# Shared hyperparams (from winning config)
LR = 0.1
WD = 0.01
BATCH_SIZE = 32
HIDDEN = 200
N_TRAIN = 1000
N_TEST = 200

# =============================================================================
# CORE TRAINING (adapted from fast.py, with curriculum support)
# =============================================================================


def generate_data(n_bits, k_sparse, secret, n_samples, rng):
    """Generate sparse parity data with a fixed secret."""
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y


def train_phase(W1, b1, W2, b2, n_bits, k_sparse, secret, config_overrides=None,
                max_epochs=200, target_acc=0.95, rng_seed=42, verbose=True, label=""):
    """
    Train one phase of the curriculum. Returns (W1, b1, W2, b2, result_dict).
    W1 shape: (hidden, n_bits), W2 shape: (1, hidden).
    """
    lr = config_overrides.get('lr', LR) if config_overrides else LR
    wd = config_overrides.get('wd', WD) if config_overrides else WD
    batch_size = config_overrides.get('batch_size', BATCH_SIZE) if config_overrides else BATCH_SIZE
    n_train = config_overrides.get('n_train', N_TRAIN) if config_overrides else N_TRAIN
    n_test = config_overrides.get('n_test', N_TEST) if config_overrides else N_TEST

    rng = np.random.RandomState(rng_seed)
    x_tr, y_tr = generate_data(n_bits, k_sparse, secret, n_train, rng)
    x_te, y_te = generate_data(n_bits, k_sparse, secret, n_test, rng)

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n_train)
        rng.shuffle(idx)

        for b_start in range(0, n_train, batch_size):
            b_end = min(b_start + batch_size, n_train)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            # Forward
            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)
            out = (h @ W2.T + b2).ravel()

            # Hinge loss mask
            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            xm, ym, hm, h_pre_m = xb[mask], yb[mask], h[mask], h_pre[mask]

            dout = -ym
            dW2 = dout[:, None] * hm
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            dW1 = dh_pre.T @ xm
            db1 = dh_pre.sum(axis=0)

            W2 -= lr * (dW2.sum(axis=0, keepdims=True) / bs + wd * W2)
            b2 -= lr * (db2 / bs + wd * b2)
            W1 -= lr * (dW1 / bs + wd * W1)
            b1 -= lr * (db1 / bs + wd * b1)

        # Evaluate
        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = float(np.mean(np.sign(te_out) == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= target_acc and solve_epoch < 0:
            solve_epoch = epoch
        if verbose and (epoch % 20 == 0 or epoch == 1 or te_acc >= target_acc):
            print(f"    {label} epoch {epoch:>4}: test={te_acc:.1%}")
        if te_acc >= target_acc:
            break

    elapsed = time.time() - start
    return W1, b1, W2, b2, {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 3),
    }


def init_weights(hidden, n_bits, seed=42):
    """He-init weights."""
    rng = np.random.RandomState(seed)
    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / hidden)
    W1 = rng.randn(hidden, n_bits) * std1
    b1 = np.zeros(hidden)
    W2 = rng.randn(1, hidden) * std2
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def expand_W1(W1, b1, old_n, new_n, rng_seed=99):
    """
    Expand W1 from (hidden, old_n) to (hidden, new_n).
    First old_n columns keep their trained values.
    New columns get small random initialization.
    """
    hidden = W1.shape[0]
    rng = np.random.RandomState(rng_seed)
    new_std = np.sqrt(2.0 / new_n) * 0.1  # Small init for new columns
    W1_new = np.zeros((hidden, new_n))
    W1_new[:, :old_n] = W1
    W1_new[:, old_n:] = rng.randn(hidden, new_n - old_n) * new_std
    return W1_new


# =============================================================================
# DIRECT TRAINING BASELINE (no curriculum)
# =============================================================================


def run_direct(n_bits, k_sparse, secret, max_epochs=500, seed=42, verbose=True):
    """Train directly on the target config. Baseline for comparison."""
    W1, b1, W2, b2 = init_weights(HIDDEN, n_bits, seed=seed + 1)
    _, _, _, _, result = train_phase(
        W1, b1, W2, b2, n_bits, k_sparse, secret,
        max_epochs=max_epochs, target_acc=0.95, rng_seed=seed,
        verbose=verbose, label=f"direct n={n_bits}/k={k_sparse}"
    )
    return result


# =============================================================================
# CURRICULUM A: n-curriculum (increase input dimension)
# =============================================================================


def run_n_curriculum(stages, k_sparse, secret, max_epochs_per_phase=300,
                     seed=42, verbose=True):
    """
    n-curriculum: train on small n, expand W1, train on larger n, etc.
    stages: list of n values, e.g. [10, 20] or [10, 30, 50].
    secret: indices that must be < stages[0] (the smallest n).
    """
    results = []
    total_start = time.time()

    # Phase 1: init and train on smallest n
    n0 = stages[0]
    W1, b1, W2, b2 = init_weights(HIDDEN, n0, seed=seed + 1)
    if verbose:
        print(f"\n  Phase 1: n={n0}, k={k_sparse}")
    W1, b1, W2, b2, phase_result = train_phase(
        W1, b1, W2, b2, n0, k_sparse, secret,
        max_epochs=max_epochs_per_phase, target_acc=0.95,
        rng_seed=seed, verbose=verbose, label=f"phase1 n={n0}"
    )
    results.append({'phase': 1, 'n': n0, **phase_result})

    # Subsequent phases: expand and continue
    for i, n_next in enumerate(stages[1:], start=2):
        n_prev = stages[i - 2]
        W1 = expand_W1(W1, b1, n_prev, n_next, rng_seed=seed + i * 100)
        if verbose:
            print(f"\n  Phase {i}: expand n={n_prev} -> n={n_next}, k={k_sparse}")
        W1, b1, W2, b2, phase_result = train_phase(
            W1, b1, W2, b2, n_next, k_sparse, secret,
            max_epochs=max_epochs_per_phase, target_acc=0.95,
            rng_seed=seed + i * 1000, verbose=verbose,
            label=f"phase{i} n={n_next}"
        )
        results.append({'phase': i, 'n': n_next, **phase_result})

    total_elapsed = time.time() - total_start
    total_epochs = sum(r['total_epochs'] for r in results)
    final_acc = results[-1]['best_test_acc']

    return {
        'phases': results,
        'total_epochs': total_epochs,
        'total_elapsed_s': round(total_elapsed, 3),
        'final_acc': final_acc,
    }


# =============================================================================
# CURRICULUM B: k-curriculum (increase parity order)
# =============================================================================


def run_k_curriculum(n_bits, k_stages, max_epochs_per_phase=300,
                     seed=42, verbose=True):
    """
    k-curriculum: train on easier k (e.g., k=2), then harder k (k=3, k=5).
    We keep the same n. Each phase gets its own secret (superset of previous).
    """
    results = []
    total_start = time.time()

    rng = np.random.RandomState(seed)

    # Pick the largest k's secret indices; smaller k uses subsets
    max_k = max(k_stages)
    full_secret = sorted(rng.choice(n_bits, max_k, replace=False).tolist())

    # Phase 1
    k0 = k_stages[0]
    secret0 = full_secret[:k0]
    W1, b1, W2, b2 = init_weights(HIDDEN, n_bits, seed=seed + 1)

    if verbose:
        print(f"\n  Phase 1: n={n_bits}, k={k0}, secret={secret0}")
    W1, b1, W2, b2, phase_result = train_phase(
        W1, b1, W2, b2, n_bits, k0, secret0,
        max_epochs=max_epochs_per_phase, target_acc=0.95,
        rng_seed=seed, verbose=verbose, label=f"phase1 k={k0}"
    )
    results.append({'phase': 1, 'k': k0, 'secret': secret0, **phase_result})

    # Subsequent phases with larger k
    for i, k_next in enumerate(k_stages[1:], start=2):
        secret_next = full_secret[:k_next]
        if verbose:
            print(f"\n  Phase {i}: n={n_bits}, k={k_next}, secret={secret_next}")
        W1, b1, W2, b2, phase_result = train_phase(
            W1, b1, W2, b2, n_bits, k_next, secret_next,
            max_epochs=max_epochs_per_phase, target_acc=0.95,
            rng_seed=seed + i * 1000, verbose=verbose,
            label=f"phase{i} k={k_next}"
        )
        results.append({'phase': i, 'k': k_next, 'secret': secret_next, **phase_result})

    total_elapsed = time.time() - total_start
    total_epochs = sum(r['total_epochs'] for r in results)
    final_acc = results[-1]['best_test_acc']

    return {
        'phases': results,
        'total_epochs': total_epochs,
        'total_elapsed_s': round(total_elapsed, 3),
        'final_acc': final_acc,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  EXPERIMENT: Curriculum Learning for Sparse Parity")
    print("=" * 70)

    all_results = {}

    # -------------------------------------------------------------------------
    # Test 1: n=10 -> n=20 curriculum vs direct n=20 (k=3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TEST 1: n-curriculum [10 -> 20], k=3")
    print("=" * 70)

    # Secret must be valid for smallest n (n=10), so all indices < 10
    rng_secret = np.random.RandomState(42)
    secret_10 = sorted(rng_secret.choice(10, 3, replace=False).tolist())
    print(f"  Secret: {secret_10}")

    # Direct baseline
    print("\n  --- Direct training: n=20, k=3 ---")
    direct_20 = run_direct(20, 3, secret_10, max_epochs=300, seed=42)
    print(f"  Direct result: acc={direct_20['best_test_acc']:.1%}, "
          f"epochs={direct_20['total_epochs']}")

    # Curriculum
    print("\n  --- Curriculum: n=10 -> n=20, k=3 ---")
    curr_10_20 = run_n_curriculum([10, 20], 3, secret_10, max_epochs_per_phase=300, seed=42)
    print(f"  Curriculum result: acc={curr_10_20['final_acc']:.1%}, "
          f"total_epochs={curr_10_20['total_epochs']}")

    all_results['test1_n10_to_n20'] = {
        'direct': direct_20,
        'curriculum': curr_10_20,
        'secret': secret_10,
    }

    # -------------------------------------------------------------------------
    # Test 2: n=10 -> n=30 -> n=50 curriculum vs direct n=50 (k=3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TEST 2: n-curriculum [10 -> 30 -> 50], k=3")
    print("=" * 70)
    print(f"  Secret: {secret_10}")

    # Direct baseline — n=50 with more epochs (500)
    print("\n  --- Direct training: n=50, k=3 ---")
    direct_50 = run_direct(50, 3, secret_10, max_epochs=500, seed=42)
    print(f"  Direct result: acc={direct_50['best_test_acc']:.1%}, "
          f"epochs={direct_50['total_epochs']}")

    # Curriculum
    print("\n  --- Curriculum: n=10 -> n=30 -> n=50, k=3 ---")
    curr_10_30_50 = run_n_curriculum(
        [10, 30, 50], 3, secret_10, max_epochs_per_phase=500, seed=42
    )
    print(f"  Curriculum result: acc={curr_10_30_50['final_acc']:.1%}, "
          f"total_epochs={curr_10_30_50['total_epochs']}")

    all_results['test2_n10_to_n50'] = {
        'direct': direct_50,
        'curriculum': curr_10_30_50,
        'secret': secret_10,
    }

    # -------------------------------------------------------------------------
    # Test 3: k-curriculum n=20: k=2 -> k=3 -> k=5
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TEST 3: k-curriculum n=20, [k=2 -> k=3 -> k=5]")
    print("=" * 70)

    # Direct baseline — n=20/k=5
    # Need the same secret for fair comparison
    rng_k = np.random.RandomState(42)
    full_secret_k = sorted(rng_k.choice(20, 5, replace=False).tolist())
    secret_k5 = full_secret_k
    secret_k3 = full_secret_k[:3]

    print(f"  Full secret (k=5): {secret_k5}")
    print(f"  Subset secret (k=3): {secret_k3}")

    print("\n  --- Direct training: n=20, k=3 (subset secret) ---")
    direct_20_k3 = run_direct(20, 3, secret_k3, max_epochs=300, seed=42)
    print(f"  Direct k=3 result: acc={direct_20_k3['best_test_acc']:.1%}, "
          f"epochs={direct_20_k3['total_epochs']}")

    print("\n  --- Direct training: n=20, k=5 ---")
    direct_20_k5 = run_direct(20, 5, secret_k5, max_epochs=1000, seed=42)
    print(f"  Direct k=5 result: acc={direct_20_k5['best_test_acc']:.1%}, "
          f"epochs={direct_20_k5['total_epochs']}")

    # k-curriculum
    print("\n  --- Curriculum: n=20, k=2 -> k=3 -> k=5 ---")
    curr_k = run_k_curriculum(20, [2, 3, 5], max_epochs_per_phase=500, seed=42)
    print(f"  Curriculum result: acc={curr_k['final_acc']:.1%}, "
          f"total_epochs={curr_k['total_epochs']}")

    all_results['test3_k_curriculum'] = {
        'direct_k3': direct_20_k3,
        'direct_k5': direct_20_k5,
        'curriculum': curr_k,
        'full_secret': full_secret_k,
    }

    # -------------------------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    rows = [
        ("Direct n=20/k=3", direct_20['best_test_acc'], direct_20['total_epochs'], direct_20['elapsed_s']),
        ("Curr 10->20/k=3", curr_10_20['final_acc'], curr_10_20['total_epochs'], curr_10_20['total_elapsed_s']),
        ("Direct n=50/k=3", direct_50['best_test_acc'], direct_50['total_epochs'], direct_50['elapsed_s']),
        ("Curr 10->30->50/k=3", curr_10_30_50['final_acc'], curr_10_30_50['total_epochs'], curr_10_30_50['total_elapsed_s']),
        ("Direct n=20/k=3*", direct_20_k3['best_test_acc'], direct_20_k3['total_epochs'], direct_20_k3['elapsed_s']),
        ("Direct n=20/k=5", direct_20_k5['best_test_acc'], direct_20_k5['total_epochs'], direct_20_k5['elapsed_s']),
        ("Curr k=2->3->5", curr_k['final_acc'], curr_k['total_epochs'], curr_k['total_elapsed_s']),
    ]

    print(f"  {'Method':<25} {'Acc':>8} {'Epochs':>8} {'Time(s)':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    for name, acc, epochs, t in rows:
        print(f"  {name:<25} {acc:>7.1%} {epochs:>8} {t:>8.2f}")

    # -------------------------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------------------------
    results_path = RESULTS_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {results_path}")


if __name__ == '__main__':
    main()
