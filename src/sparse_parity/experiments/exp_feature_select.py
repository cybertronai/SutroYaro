#!/usr/bin/env python3
"""
Experiment exp_feature_select: Feature Selection — Blank Slate Sparse Parity

Hypothesis: Separating SEARCH (find which k bits matter) from LEARNING (compute parity)
can solve sparse parity faster than end-to-end SGD by exploiting the structure directly.

Three approaches:
  1. Pairwise interaction detection (correlation of y * x_i * x_j)
  2. Greedy forward selection (add bit that most improves product classifier)
  3. Exhaustive triplet/combo check (brute-force over C(n,k) combos)

Answers: Blank slate — can we beat SGD by not using gradient descent at all?

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_feature_select.py
"""

import time
import json
import numpy as np
from itertools import combinations
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_parity.config import Config
from sparse_parity.fast import generate, train as sgd_train

EXP_NAME = "exp_feature_select"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / EXP_NAME


# =============================================================================
# APPROACH 1: Pairwise interaction detection
# =============================================================================

def pairwise_detection(x, y, n_bits, k_sparse):
    """
    For k-parity on secret {a,b,c}, the product x_a*x_b has nonzero correlation
    with y (since y = x_a*x_b*x_c, so y*x_a*x_b = x_c which has mean 0...
    actually for k=3: E[y * x_i * x_j] = E[x_a*x_b*x_c * x_i * x_j].
    This is nonzero only when {i,j} is a subset of {a,b,c}, giving E[x_c^2]=1
    or similar. So pairs (a,b),(a,c),(b,c) score 1.0 in expectation, others ~0.

    For k=5 with secret {a,b,c,d,e}: pair (a,b) gives E[y*x_a*x_b] = E[x_c*x_d*x_e] = 0.
    So pairwise detection only works for k=3 (where pairs leave 1 bit) or k=2.
    For k=5, we'd need order-4 interactions. We'll try it and see.
    """
    n_samples = x.shape[0]
    scores = np.zeros((n_bits, n_bits))
    ops = 0

    for i in range(n_bits):
        for j in range(i + 1, n_bits):
            # correlation: abs(mean(y * x_i * x_j))
            scores[i, j] = abs(np.mean(y * x[:, i] * x[:, j]))
            ops += n_samples * 3  # multiply, multiply, mean

    # Find top-scoring pairs
    flat_idx = np.argsort(scores.ravel())[::-1]
    top_pairs = []
    for idx in flat_idx[:k_sparse * 2]:  # grab enough pairs
        i, j = divmod(idx, n_bits)
        if scores[i, j] < 0.1:
            break
        top_pairs.append((i, j))

    # Intersect top pairs to find candidate bits
    if len(top_pairs) >= 2:
        # Count frequency of each bit in top pairs
        bit_counts = {}
        for i, j in top_pairs:
            bit_counts[i] = bit_counts.get(i, 0) + 1
            bit_counts[j] = bit_counts.get(j, 0) + 1
        # Top k bits by frequency
        candidates = sorted(bit_counts.keys(), key=lambda b: -bit_counts[b])[:k_sparse]
        candidates = sorted(candidates)
    else:
        candidates = []

    # Verify: does product(x[:, candidates]) == y?
    if candidates:
        pred = np.prod(x[:, candidates], axis=1)
        acc = np.mean(pred == y)
    else:
        acc = 0.0

    return {
        'candidates': candidates,
        'accuracy': float(acc),
        'top_pairs': [(int(i), int(j), float(scores[i, j])) for i, j in top_pairs[:5]],
        'n_pair_evals': n_bits * (n_bits - 1) // 2,
        'ops': ops,
    }


# =============================================================================
# APPROACH 2: Greedy forward selection
# =============================================================================

def greedy_forward(x, y, n_bits, k_sparse):
    """
    Start with empty set S. For each remaining bit, test if adding it
    improves the product classifier. Add the best bit. Repeat k times.
    """
    n_samples = x.shape[0]
    selected = []
    ops = 0

    for step in range(k_sparse):
        best_bit = -1
        best_acc = -1

        for i in range(n_bits):
            if i in selected:
                continue
            trial = selected + [i]
            pred = np.prod(x[:, trial], axis=1)
            acc = np.mean(pred == y)
            ops += n_samples * len(trial)  # product ops
            ops += n_samples  # comparison

            if acc > best_acc:
                best_acc = acc
                best_bit = i

        selected.append(best_bit)

    selected = sorted(selected)

    # Final accuracy
    pred = np.prod(x[:, selected], axis=1)
    acc = np.mean(pred == y)

    return {
        'candidates': selected,
        'accuracy': float(acc),
        'n_bit_evals': sum(n_bits - i for i in range(k_sparse)),
        'ops': ops,
    }


# =============================================================================
# APPROACH 3: Exhaustive combo check
# =============================================================================

def exhaustive_check(x, y, n_bits, k_sparse):
    """
    For each C(n,k) combo, compute accuracy of product(x[:, combo]) as classifier.
    The correct combo gives 100% accuracy.
    """
    n_samples = x.shape[0]
    ops = 0
    n_combos = 0

    best_combo = None
    best_acc = -1

    for combo in combinations(range(n_bits), k_sparse):
        pred = np.prod(x[:, list(combo)], axis=1)
        acc = np.mean(pred == y)
        ops += n_samples * k_sparse + n_samples  # product + comparison
        n_combos += 1

        if acc > best_acc:
            best_acc = acc
            best_combo = combo

        if acc >= 1.0:
            break  # found it

    return {
        'candidates': sorted(best_combo) if best_combo else [],
        'accuracy': float(best_acc),
        'n_combos_checked': n_combos,
        'n_combos_total': int(np.prod([n_bits - i for i in range(k_sparse)]) /
                              np.prod([i + 1 for i in range(k_sparse)])),
        'ops': ops,
    }


# =============================================================================
# SGD BASELINE (using fast.py)
# =============================================================================

def run_sgd_baseline(config):
    """Run SGD baseline and return results with op count estimate."""
    result = sgd_train(config, verbose=False)
    # Estimate ops: per epoch ~ n_train * (hidden*n_bits + hidden + hidden + ...) ~ n_train * hidden * n_bits * 4
    ops_per_epoch = config.n_train * config.hidden * config.n_bits * 4
    total_ops = result['total_epochs'] * ops_per_epoch
    result['ops_estimate'] = total_ops
    return result


# =============================================================================
# MAIN
# =============================================================================

def run_scenario(label, n_bits, k_sparse, n_train, seed=42):
    """Run all approaches on one scenario."""
    config = Config(
        n_bits=n_bits, k_sparse=k_sparse, hidden=200,
        lr=0.1, wd=0.01, max_epochs=200,
        n_train=n_train, n_test=500, seed=seed,
    )
    config.batch_size = 32

    # Generate data
    x_tr, y_tr, x_te, y_te, secret = generate(config)

    print(f"\n  --- {label}: n={n_bits}, k={k_sparse}, n_train={n_train}, secret={secret} ---")

    results = {'label': label, 'n_bits': n_bits, 'k_sparse': k_sparse,
               'n_train': n_train, 'secret': secret}

    # Approach 1: Pairwise
    t0 = time.time()
    r1 = pairwise_detection(x_tr, y_tr, n_bits, k_sparse)
    r1['time_s'] = time.time() - t0
    r1['correct'] = r1['candidates'] == secret
    results['pairwise'] = r1
    print(f"    Pairwise:   found={r1['candidates']}  acc={r1['accuracy']:.0%}  "
          f"correct={r1['correct']}  time={r1['time_s']:.4f}s  ops={r1['ops']:,}")

    # Approach 2: Greedy
    t0 = time.time()
    r2 = greedy_forward(x_tr, y_tr, n_bits, k_sparse)
    r2['time_s'] = time.time() - t0
    r2['correct'] = r2['candidates'] == secret
    results['greedy'] = r2
    print(f"    Greedy:     found={r2['candidates']}  acc={r2['accuracy']:.0%}  "
          f"correct={r2['correct']}  time={r2['time_s']:.4f}s  ops={r2['ops']:,}")

    # Approach 3: Exhaustive
    t0 = time.time()
    r3 = exhaustive_check(x_tr, y_tr, n_bits, k_sparse)
    r3['time_s'] = time.time() - t0
    r3['correct'] = r3['candidates'] == secret
    results['exhaustive'] = r3
    print(f"    Exhaustive: found={r3['candidates']}  acc={r3['accuracy']:.0%}  "
          f"correct={r3['correct']}  time={r3['time_s']:.4f}s  ops={r3['ops']:,}  "
          f"combos={r3['n_combos_checked']}/{r3['n_combos_total']}")

    # SGD baseline
    t0 = time.time()
    r4 = run_sgd_baseline(config)
    r4['time_s'] = time.time() - t0
    r4['correct'] = r4['best_test_acc'] >= 0.95
    results['sgd'] = {
        'accuracy': float(r4['best_test_acc']),
        'time_s': r4['time_s'],
        'epochs': r4['total_epochs'],
        'ops_estimate': r4['ops_estimate'],
        'correct': r4['correct'],
    }
    print(f"    SGD:        acc={r4['best_test_acc']:.0%}  "
          f"correct={r4['correct']}  time={r4['time_s']:.4f}s  "
          f"epochs={r4['total_epochs']}  ops~{r4['ops_estimate']:,}")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'='*70}")
    print(f"  EXPERIMENT: {EXP_NAME}")
    print(f"  Feature Selection — Blank Slate Sparse Parity")
    print(f"{'='*70}")

    all_results = []

    # Scenario 1: n=20, k=3 (standard)
    all_results.append(run_scenario("n20_k3", 20, 3, n_train=1000))

    # Scenario 2: n=50, k=3 (harder — SGD fails without curriculum)
    all_results.append(run_scenario("n50_k3", 50, 3, n_train=1000))

    # Scenario 3: n=20, k=5 (higher order — pairwise should fail)
    all_results.append(run_scenario("n20_k5", 20, 5, n_train=5000))

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<12} {'Method':<14} {'Correct':>8} {'Time(s)':>10} {'Ops':>14}")
    print(f"  {'─'*12} {'─'*14} {'─'*8} {'─'*10} {'─'*14}")

    for r in all_results:
        label = r['label']
        for method in ['pairwise', 'greedy', 'exhaustive', 'sgd']:
            m = r[method]
            ops = m.get('ops', m.get('ops_estimate', 0))
            print(f"  {label:<12} {method:<14} {str(m['correct']):>8} "
                  f"{m['time_s']:>10.4f} {ops:>14,}")

    # Speedup table
    print(f"\n  SPEEDUPS vs SGD:")
    print(f"  {'Scenario':<12} {'Method':<14} {'Time Speedup':>14} {'Ops Speedup':>14}")
    print(f"  {'─'*12} {'─'*14} {'─'*14} {'─'*14}")
    for r in all_results:
        sgd_time = r['sgd']['time_s']
        sgd_ops = r['sgd'].get('ops_estimate', 1)
        for method in ['pairwise', 'greedy', 'exhaustive']:
            m = r[method]
            if m['correct']:
                t_speedup = sgd_time / max(m['time_s'], 1e-9)
                ops = m.get('ops', 1)
                o_speedup = sgd_ops / max(ops, 1)
                print(f"  {r['label']:<12} {method:<14} {t_speedup:>13.1f}x {o_speedup:>13.1f}x")
            else:
                print(f"  {r['label']:<12} {method:<14} {'FAILED':>14} {'FAILED':>14}")

    # Save
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_DIR / 'results.json'}")


if __name__ == '__main__':
    main()
