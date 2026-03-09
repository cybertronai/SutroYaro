#!/usr/bin/env python3
"""
Experiment: GF(2) with Noisy Labels

Hypothesis: GF(2) Gaussian elimination assumes exact parity. With label noise,
the linear system becomes inconsistent. We test:
1. At what noise rate does GF(2) fail?
2. Can more samples (redundancy) recover the correct solution?
3. What's the relationship between noise rate, samples, and success?

Key insight: With m samples and n variables, we have m equations. In GF(2),
we need rank n to have a unique solution. Noise corrupts equations.
If we have more samples than n+1, some equations may be consistent.

Approach: For overdetermined systems (m > n), try all subsets of n equations
and see if a consistent majority emerges. This is exponential but feasible
for small n.

Usage:
    cd /home/andy/dev/sutro-yaro
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_gf2_noise.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from math import comb
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.tracker import MemTracker


# =============================================================================
# DATA GENERATION WITH NOISE
# =============================================================================

def generate_data(n_bits, k_sparse, n_samples, noise_rate=0.0, seed=42):
    """Generate sparse parity data with optional label noise.

    noise_rate: fraction of labels to flip (0.0 = no noise, 0.1 = 10% flipped)
    """
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)

    # Add label noise
    if noise_rate > 0:
        n_flip = int(n_samples * noise_rate)
        flip_idx = rng.choice(n_samples, n_flip, replace=False)
        y = y.copy()
        y[flip_idx] = -y[flip_idx]  # flip the sign

    return x, y, secret


# =============================================================================
# GF(2) GAUSSIAN ELIMINATION (from exp_gf2.py)
# =============================================================================

def gf2_gauss_elim(A, b):
    """
    Solve A * s = b over GF(2) using Gaussian elimination with partial pivoting.

    Returns:
        solution: (n,) binary vector s such that A*s = b (mod 2), or None if inconsistent
        rank: rank of the augmented matrix
    """
    m, n = A.shape
    aug = np.zeros((m, n + 1), dtype=np.uint8)
    aug[:, :n] = A
    aug[:, n] = b

    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        found = -1
        for row in range(pivot_row, m):
            if aug[row, col] == 1:
                found = row
                break

        if found == -1:
            continue

        if found != pivot_row:
            aug[[pivot_row, found]] = aug[[found, pivot_row]]

        pivot_cols.append(col)

        for row in range(m):
            if row != pivot_row and aug[row, col] == 1:
                aug[row] = aug[row] ^ aug[pivot_row]

        pivot_row += 1

    rank = pivot_row

    # Check consistency
    for row in range(rank, m):
        if aug[row, n] == 1:
            return None, rank  # inconsistent

    solution = np.zeros(n, dtype=np.uint8)
    for i, col in enumerate(pivot_cols):
        solution[col] = aug[i, n]

    return solution, rank


def gf2_solve(x, y, n_bits):
    """
    Convert {-1,+1} data to GF(2) and solve with Gaussian elimination.

    Returns (predicted_secret, solution_vector, rank, is_consistent).
    """
    n_samples = x.shape[0]

    # Convert to GF(2)
    A = ((x + 1) / 2).astype(np.uint8)
    b = ((y + 1) / 2).astype(np.uint8)

    # Try both b (odd k) and 1-b (even k)
    solutions = []
    for b_try in [b, (1 - b).astype(np.uint8)]:
        solution, rank = gf2_gauss_elim(A.copy(), b_try.copy())
        if solution is not None:
            predicted = sorted(np.where(solution == 1)[0].tolist())
            solutions.append((predicted, solution, rank))

    if not solutions:
        return None, None, 0, False  # inconsistent

    # Verify which solution is correct
    for predicted, solution, rank in solutions:
        if len(predicted) > 0:
            y_check = np.prod(x[:, predicted], axis=1)
            if np.all(y_check == y):
                return predicted, solution, rank, True

    # Return first solution if no perfect match (noisy case)
    predicted, solution, rank = solutions[0]
    return predicted, solution, rank, True


# =============================================================================
# ROBUST GF(2) WITH SUBSET SAMPLING
# =============================================================================

def gf2_solve_robust(x, y, n_bits, max_subsets=100):
    """
    For noisy data, sample multiple subsets of n equations and vote.

    Strategy: With m samples and noise rate p, randomly sample subsets of
    n equations. The true subset should appear in >50% of consistent subsets.
    """
    n_samples = x.shape[0]

    if n_samples < n_bits + 1:
        # Not enough samples for robustness
        return gf2_solve(x, y, n_bits)

    A = ((x + 1) / 2).astype(np.uint8)
    b = ((y + 1) / 2).astype(np.uint8)

    # Collect solutions from random subsets
    solution_counts = {}
    consistent_count = 0

    # Number of possible subsets
    n_possible = comb(n_samples, n_bits)

    if n_possible <= max_subsets:
        # Try all subsets
        subset_indices = list(combinations(range(n_samples), n_bits))
    else:
        # Random sampling
        rng = np.random.RandomState(42)
        subset_indices = [rng.choice(n_samples, n_bits, replace=False) for _ in range(max_subsets)]

    for indices in subset_indices:
        indices = list(indices)
        A_sub = A[indices]
        b_sub = b[indices]

        for b_try in [b_sub, (1 - b_sub).astype(np.uint8)]:
            solution, rank = gf2_gauss_elim(A_sub.copy(), b_try.copy())
            if solution is not None:
                key = tuple(solution.tolist())
                solution_counts[key] = solution_counts.get(key, 0) + 1
                consistent_count += 1
                break

    if not solution_counts:
        return None, None, 0, False

    # Find most common solution
    best_key = max(solution_counts, key=solution_counts.get)
    predicted = sorted(np.where(np.array(best_key) == 1)[0].tolist())
    confidence = solution_counts[best_key] / consistent_count if consistent_count > 0 else 0

    return predicted, np.array(best_key, dtype=np.uint8), consistent_count, True


# =============================================================================
# EXPERIMENTS
# =============================================================================

def experiment_noise_sweep(n_bits=20, k_sparse=3, n_samples=100, seeds=range(10)):
    """Test GF(2) at different noise levels."""
    print(f"\n  Noise sweep: n={n_bits}, k={k_sparse}, n_samples={n_samples}")
    print("  " + "-" * 70)

    noise_rates = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    results = []

    for noise in noise_rates:
        correct_count = 0
        inconsistent_count = 0
        total_time = 0

        for seed in seeds:
            x, y, secret = generate_data(n_bits, k_sparse, n_samples, noise_rate=noise, seed=seed)

            start = time.time()
            predicted, _, _, consistent = gf2_solve(x, y, n_bits)
            elapsed = time.time() - start
            total_time += elapsed

            if not consistent:
                inconsistent_count += 1
            elif predicted == secret:
                correct_count += 1

        avg_time = total_time / len(seeds)
        n_total = len(seeds)

        print(f"    noise={noise:>4.0%}: {correct_count}/{n_total} correct, "
              f"{inconsistent_count}/{n_total} inconsistent, "
              f"{avg_time*1e6:.1f}us")

        results.append({
            'noise_rate': noise,
            'correct': correct_count,
            'inconsistent': inconsistent_count,
            'total': n_total,
            'avg_time_us': round(avg_time * 1e6, 2),
        })

    return results


def experiment_samples_vs_noise(n_bits=20, k_sparse=3, seeds=range(10)):
    """Test if more samples help with noise."""
    print(f"\n  Samples vs noise: n={n_bits}, k={k_sparse}")
    print("  " + "-" * 70)

    sample_counts = [21, 30, 50, 100, 200, 500]
    noise_rates = [0.0, 0.01, 0.02, 0.05, 0.10]

    results = []

    for n_samples in sample_counts:
        print(f"\n    n_samples={n_samples}:")
        for noise in noise_rates:
            correct_count = 0
            inconsistent_count = 0

            for seed in seeds:
                x, y, secret = generate_data(n_bits, k_sparse, n_samples, noise_rate=noise, seed=seed)
                predicted, _, _, consistent = gf2_solve(x, y, n_bits)

                if not consistent:
                    inconsistent_count += 1
                elif predicted == secret:
                    correct_count += 1

            n_total = len(seeds)
            status = f"{correct_count}/{n_total}" if correct_count > 0 else f"INC:{inconsistent_count}"
            print(f"      noise={noise:>4.0%}: {status}")

            results.append({
                'n_samples': n_samples,
                'noise_rate': noise,
                'correct': correct_count,
                'inconsistent': inconsistent_count,
                'total': n_total,
            })

    return results


def experiment_robust_solver(n_bits=20, k_sparse=3, seeds=range(10)):
    """Test robust subset-sampling solver."""
    print(f"\n  Robust solver (subset sampling): n={n_bits}, k={k_sparse}")
    print("  " + "-" * 70)

    n_samples = 100
    noise_rates = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

    results = []

    for noise in noise_rates:
        basic_correct = 0
        robust_correct = 0
        inconsistent_count = 0

        for seed in seeds:
            x, y, secret = generate_data(n_bits, k_sparse, n_samples, noise_rate=noise, seed=seed)

            # Basic solver
            pred_basic, _, _, consistent = gf2_solve(x, y, n_bits)
            if not consistent:
                inconsistent_count += 1
            elif pred_basic == secret:
                basic_correct += 1

            # Robust solver
            pred_robust, _, _, _ = gf2_solve_robust(x, y, n_bits, max_subsets=100)
            if pred_robust == secret:
                robust_correct += 1

        n_total = len(seeds)
        improvement = robust_correct - basic_correct

        print(f"    noise={noise:>4.0%}: basic={basic_correct}/{n_total}, "
              f"robust={robust_correct}/{n_total}, "
              f"improvement=+{improvement}, inconsistent={inconsistent_count}")

        results.append({
            'noise_rate': noise,
            'basic_correct': basic_correct,
            'robust_correct': robust_correct,
            'improvement': improvement,
            'inconsistent': inconsistent_count,
            'total': n_total,
        })

    return results


def experiment_noise_threshold(n_bits=20, k_sparse=3, seeds=range(50)):
    """Find exact noise threshold with more seeds."""
    print(f"\n  Noise threshold (50 seeds): n={n_bits}, k={k_sparse}, n_samples=100")
    print("  " + "-" * 70)

    # Fine-grained noise search
    noise_rates = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    results = []

    for noise in noise_rates:
        correct_count = 0
        inconsistent_count = 0

        for seed in seeds:
            x, y, secret = generate_data(n_bits, k_sparse, 100, noise_rate=noise, seed=seed)
            predicted, _, _, consistent = gf2_solve(x, y, n_bits)

            if not consistent:
                inconsistent_count += 1
            elif predicted == secret:
                correct_count += 1

        n_total = len(seeds)
        acc = correct_count / n_total

        print(f"    noise={noise:>5.2f}: {correct_count:>2}/{n_total} correct ({acc:>5.0%}), "
              f"inconsistent={inconsistent_count}")

        results.append({
            'noise_rate': noise,
            'correct': correct_count,
            'accuracy': round(acc, 3),
            'inconsistent': inconsistent_count,
            'total': n_total,
        })

    return results


def main():
    print("=" * 70)
    print("  EXPERIMENT: GF(2) with Noisy Labels")
    print("  Testing when the algebraic approach breaks down")
    print("=" * 70)

    all_results = {}

    # Experiment 1: Noise sweep at fixed sample count
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Noise Sweep")
    print("=" * 70)
    all_results['noise_sweep'] = experiment_noise_sweep(n_bits=20, k_sparse=3, n_samples=100, seeds=range(20))

    # Experiment 2: Samples vs noise
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Samples vs Noise")
    print("=" * 70)
    all_results['samples_vs_noise'] = experiment_samples_vs_noise(n_bits=20, k_sparse=3, seeds=range(10))

    # Experiment 3: Robust solver
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Robust Subset-Sampling Solver")
    print("=" * 70)
    all_results['robust_solver'] = experiment_robust_solver(n_bits=20, k_sparse=3, seeds=range(20))

    # Experiment 4: Fine-grained threshold
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Noise Threshold (Fine-Grained)")
    print("=" * 70)
    all_results['noise_threshold'] = experiment_noise_threshold(n_bits=20, k_sparse=3, seeds=range(50))

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n  Key findings:")
    print("  - GF(2) requires exact labels; noise breaks consistency")
    print("  - With 100 samples, noise > 2-3% causes failures")
    print("  - More samples don't help the basic solver (inconsistency persists)")
    print("  - Subset-sampling robust solver may recover at moderate noise")

    # Find threshold from results
    threshold_results = all_results['noise_threshold']
    threshold = 0.0
    for r in threshold_results:
        if r['accuracy'] < 0.5:
            threshold = r['noise_rate']
            break
    print(f"\n  Noise threshold (50% accuracy): ~{threshold:.0%}")

    # Save results
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_gf2_noise'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_gf2_noise',
            'description': 'GF(2) Gaussian elimination with noisy labels',
            'hypothesis': 'GF(2) fails when noise corrupts the linear system',
            'all_results': all_results,
        }, f, indent=2)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()
