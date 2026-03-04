#!/usr/bin/env python3
"""
Experiment: Evolutionary/Random Search for Sparse Parity

Hypothesis: Random and evolutionary search over k-subsets can solve sparse
parity without any neural network or gradient computation. Random search
should need ~C(n,k) tries on average; evolutionary search should beat that.

This is a BLANK SLATE approach — no neural net, no SGD, no gradients.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.experiments.exp_evolutionary
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from math import comb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_data(n_bits, k_sparse, n_samples, seed=42):
    """Generate sparse parity data. Returns x, y, secret."""
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y, secret


# =============================================================================
# APPROACH 1: RANDOM SEARCH
# =============================================================================

def random_search(x, y, n_bits, k_sparse, max_tries=100000, seed=42):
    """
    Randomly sample k-subsets of {0..n-1}.
    For each, check if product(x[:, subset]) matches y for all training samples.
    Returns (found_subset, n_tries, elapsed_s).
    """
    rng = np.random.RandomState(seed)
    start = time.time()

    for t in range(1, max_tries + 1):
        subset = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())
        parity = np.prod(x[:, subset], axis=1)
        if np.all(parity == y):
            elapsed = time.time() - start
            return subset, t, elapsed

    elapsed = time.time() - start
    return None, max_tries, elapsed


# =============================================================================
# APPROACH 2: EVOLUTIONARY SEARCH
# =============================================================================

def fitness(x, y, indices):
    """Fraction of samples where product(x[:, indices]) == y."""
    parity = np.prod(x[:, indices], axis=1)
    return np.mean(parity == y)


def mutate(indices, n_bits, rng):
    """Replace one random index with a different one."""
    indices = list(indices)
    pos = rng.randint(len(indices))
    available = [i for i in range(n_bits) if i not in indices]
    if not available:
        return tuple(sorted(indices))
    indices[pos] = available[rng.randint(len(available))]
    return tuple(sorted(indices))


def crossover(parent_a, parent_b, k_sparse, rng):
    """Take some indices from parent A, rest from parent B. Ensure k unique."""
    pool = list(set(parent_a) | set(parent_b))
    if len(pool) < k_sparse:
        return parent_a  # fallback
    chosen = sorted(rng.choice(pool, k_sparse, replace=False).tolist())
    return tuple(chosen)


def tournament_select(population, fitnesses, tournament_size, rng):
    """Pick tournament_size candidates, return the best."""
    indices = rng.choice(len(population), tournament_size, replace=False)
    best_idx = indices[np.argmax(fitnesses[indices])]
    return population[best_idx]


def evolutionary_search(x, y, n_bits, k_sparse, pop_size=100,
                         max_generations=1000, tournament_size=3,
                         mutation_rate=0.8, crossover_rate=0.3,
                         seed=42):
    """
    Evolutionary search over k-subsets.
    Population of pop_size candidates, each a set of k indices.
    Fitness = fraction of training samples correctly predicted.
    Returns (best_subset, generation, elapsed_s, best_fitness).
    """
    rng = np.random.RandomState(seed)
    start = time.time()

    # Initialize population: random k-subsets
    population = []
    for _ in range(pop_size):
        subset = tuple(sorted(rng.choice(n_bits, k_sparse, replace=False).tolist()))
        population.append(subset)

    for gen in range(1, max_generations + 1):
        # Evaluate fitness
        fitnesses = np.array([fitness(x, y, ind) for ind in population])

        # Check for solution
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] >= 1.0:
            elapsed = time.time() - start
            return list(population[best_idx]), gen, elapsed, float(fitnesses[best_idx])

        # Create next generation
        new_pop = []
        # Elitism: keep best
        new_pop.append(population[best_idx])

        while len(new_pop) < pop_size:
            parent_a = tournament_select(population, fitnesses, tournament_size, rng)

            if rng.random() < crossover_rate:
                parent_b = tournament_select(population, fitnesses, tournament_size, rng)
                child = crossover(parent_a, parent_b, k_sparse, rng)
            else:
                child = parent_a

            if rng.random() < mutation_rate:
                child = mutate(child, n_bits, rng)

            new_pop.append(child)

        population = new_pop

    # Final evaluation
    fitnesses = np.array([fitness(x, y, ind) for ind in population])
    best_idx = np.argmax(fitnesses)
    elapsed = time.time() - start
    return list(population[best_idx]), max_generations, elapsed, float(fitnesses[best_idx])


# =============================================================================
# MAIN
# =============================================================================

def run_config(n_bits, k_sparse, n_train, seeds, max_random_tries=200000,
               pop_size=100, max_generations=1000, verbose=True):
    """Run both approaches on one config across multiple seeds."""
    c_n_k = comb(n_bits, k_sparse)
    if verbose:
        print(f"\n  Config: n={n_bits}, k={k_sparse}, C(n,k)={c_n_k}")
        print(f"  Training samples: {n_train}")
        print(f"  Max random tries: {max_random_tries}, Max generations: {max_generations}")

    random_results = []
    evo_results = []

    for seed in seeds:
        x, y, secret = generate_data(n_bits, k_sparse, n_train, seed=seed)

        # Random search
        found, tries, elapsed = random_search(
            x, y, n_bits, k_sparse, max_tries=max_random_tries, seed=seed + 100
        )
        solved = found is not None
        correct = found == secret if solved else False
        random_results.append({
            'seed': seed,
            'found': found,
            'secret': secret,
            'correct': correct,
            'solved': solved,
            'tries': tries,
            'elapsed_s': round(elapsed, 4),
        })
        if verbose:
            status = f"SOLVED in {tries} tries ({elapsed:.3f}s)" if solved else f"FAILED after {tries} tries"
            print(f"    [Random] seed={seed}: {status}")

        # Evolutionary search
        found_evo, gen, elapsed_evo, best_fit = evolutionary_search(
            x, y, n_bits, k_sparse, pop_size=pop_size,
            max_generations=max_generations, seed=seed + 200
        )
        solved_evo = best_fit >= 1.0
        correct_evo = sorted(found_evo) == secret if solved_evo else False
        evo_results.append({
            'seed': seed,
            'found': found_evo,
            'secret': secret,
            'correct': correct_evo,
            'solved': solved_evo,
            'generations': gen,
            'best_fitness': best_fit,
            'elapsed_s': round(elapsed_evo, 4),
        })
        if verbose:
            status = f"SOLVED in {gen} gens ({elapsed_evo:.3f}s)" if solved_evo else f"FAILED ({best_fit:.1%} best) after {gen} gens"
            print(f"    [Evo]    seed={seed}: {status}")

    return {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'c_n_k': c_n_k,
        'n_train': n_train,
        'random': random_results,
        'evolutionary': evo_results,
    }


def main():
    print("=" * 70)
    print("  EXPERIMENT: Evolutionary/Random Search for Sparse Parity")
    print("  BLANK SLATE: No neural net, no SGD, no gradients")
    print("=" * 70)

    seeds = [42, 43, 44, 45, 46]
    all_results = {}

    # -------------------------------------------------------------------
    # Config 1: n=20, k=3 — C(20,3) = 1140
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 1: n=20, k=3  [C(20,3) = 1,140]")
    print("=" * 70)
    all_results['n20_k3'] = run_config(
        n_bits=20, k_sparse=3, n_train=500, seeds=seeds,
        max_random_tries=50000, pop_size=100, max_generations=500
    )

    # -------------------------------------------------------------------
    # Config 2: n=50, k=3 — C(50,3) = 19600
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 2: n=50, k=3  [C(50,3) = 19,600]")
    print("=" * 70)
    all_results['n50_k3'] = run_config(
        n_bits=50, k_sparse=3, n_train=500, seeds=seeds,
        max_random_tries=200000, pop_size=200, max_generations=2000
    )

    # -------------------------------------------------------------------
    # Config 3: n=20, k=5 — C(20,5) = 15504
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CONFIG 3: n=20, k=5  [C(20,5) = 15,504]")
    print("=" * 70)
    all_results['n20_k5'] = run_config(
        n_bits=20, k_sparse=5, n_train=2000, seeds=seeds,
        max_random_tries=200000, pop_size=200, max_generations=2000
    )

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    header = (f"  {'Config':<15} | {'C(n,k)':>8} | "
              f"{'Random tries':>13} | {'Random time':>11} | "
              f"{'Evo gens':>9} | {'Evo time':>9} | {'Evo solved':>10}")
    print(header)
    print("  " + "-" * 88)

    for key, res in all_results.items():
        # Random stats
        r_solved = sum(1 for r in res['random'] if r['solved'])
        r_tries_avg = np.mean([r['tries'] for r in res['random'] if r['solved']]) if r_solved else float('nan')
        r_time_avg = np.mean([r['elapsed_s'] for r in res['random'] if r['solved']]) if r_solved else float('nan')

        # Evo stats
        e_solved = sum(1 for r in res['evolutionary'] if r['solved'])
        e_gens_avg = np.mean([r['generations'] for r in res['evolutionary'] if r['solved']]) if e_solved else float('nan')
        e_time_avg = np.mean([r['elapsed_s'] for r in res['evolutionary'] if r['solved']]) if e_solved else float('nan')

        r_str = f"{r_tries_avg:.0f}" if not np.isnan(r_tries_avg) else "FAIL"
        r_t_str = f"{r_time_avg:.3f}s" if not np.isnan(r_time_avg) else "---"
        e_str = f"{e_gens_avg:.0f}" if not np.isnan(e_gens_avg) else "FAIL"
        e_t_str = f"{e_time_avg:.3f}s" if not np.isnan(e_time_avg) else "---"

        print(f"  {key:<15} | {res['c_n_k']:>8,} | "
              f"{r_str:>13} | {r_t_str:>11} | "
              f"{e_str:>9} | {e_t_str:>9} | {e_solved}/{len(res['evolutionary']):>9}")

    # -------------------------------------------------------------------
    # SGD baseline comparison
    # -------------------------------------------------------------------
    print("\n  SGD Baselines (from DISCOVERIES.md):")
    print("  " + "-" * 60)
    print(f"  {'n=20,k=3 SGD':<30} | {'~5 epochs / 0.12s':>25}")
    print(f"  {'n=50,k=3 SGD (curriculum)':<30} | {'20 epochs total':>25}")
    print(f"  {'n=50,k=3 SGD (direct)':<30} | {'FAIL (54%)':>25}")
    print(f"  {'n=20,k=5 SGD (n_train=5000)':<30} | {'14 epochs':>25}")

    print("=" * 90)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_evolutionary'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_evolutionary',
            'description': 'Random and evolutionary search over k-subsets for sparse parity',
            'hypothesis': 'Random search needs ~C(n,k) tries; evolutionary search beats that',
            'approach': 'blank_slate — no neural net, no SGD, no gradients',
            'configs': all_results,
        }, f, indent=2, default=str)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()
