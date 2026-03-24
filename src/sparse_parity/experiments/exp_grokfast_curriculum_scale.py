#!/usr/bin/env python3
"""
Experiment: GrokFast + Curriculum scaling — how far does it go?

Hypothesis: GrokFast + Curriculum solved n=50/k=5 in 14 epochs / 77ms where SGD
fails completely. Since curriculum makes n-expansion nearly free, this should
scale to n=100 and beyond. The bottleneck should be the initial small-n/high-k
training, not the expansion phases.

Usage:
    PYTHONPATH=src python3 src/sparse_parity/experiments/exp_grokfast_curriculum_scale.py
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

EXP_NAME = "exp_grokfast_curriculum_scale"
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / EXP_NAME

LR = 0.1
WD = 0.01
BATCH_SIZE = 32
HIDDEN = 200
N_TRAIN = 2000
N_TEST = 200

GF_ALPHA = 0.98
GF_LAM = 2.0

SEEDS = [42, 43, 44, 45, 46]


# =============================================================================
# CORE FUNCTIONS (same as exp_grokfast_curriculum)
# =============================================================================

def generate_data(n_bits, k_sparse, secret, n_samples, rng):
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = np.prod(x[:, secret], axis=1)
    return x, y


def init_weights(hidden, n_bits, seed=42):
    rng = np.random.RandomState(seed)
    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / hidden)
    W1 = rng.randn(hidden, n_bits) * std1
    b1 = np.zeros(hidden)
    W2 = rng.randn(1, hidden) * std2
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def expand_W1(W1, old_n, new_n, rng_seed=99):
    hidden = W1.shape[0]
    rng = np.random.RandomState(rng_seed)
    new_std = np.sqrt(2.0 / new_n) * 0.1
    W1_new = np.zeros((hidden, new_n))
    W1_new[:, :old_n] = W1
    W1_new[:, old_n:] = rng.randn(hidden, new_n - old_n) * new_std
    return W1_new


def train_phase(W1, b1, W2, b2, n_bits, k_sparse, secret,
                max_epochs=500, target_acc=0.95, rng_seed=42,
                grokfast=False, alpha=0.98, lam=2.0):
    rng = np.random.RandomState(rng_seed)
    x_tr, y_tr = generate_data(n_bits, k_sparse, secret, N_TRAIN, rng)
    x_te, y_te = generate_data(n_bits, k_sparse, secret, N_TEST, rng)

    if grokfast:
        ema_W1 = np.zeros_like(W1)
        ema_b1 = np.zeros_like(b1)
        ema_W2 = np.zeros_like(W2)
        ema_b2 = np.zeros_like(b2)

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(N_TRAIN)
        rng.shuffle(idx)

        for b_start in range(0, N_TRAIN, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, N_TRAIN)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)
            out = (h @ W2.T + b2).ravel()

            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            xm, ym, hm, h_pre_m = xb[mask], yb[mask], h[mask], h_pre[mask]

            dout = -ym
            gW2 = (dout[:, None] * hm).sum(axis=0, keepdims=True) / bs
            gb2 = dout.sum() / bs
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            gW1 = (dh_pre.T @ xm) / bs
            gb1 = dh_pre.sum(axis=0) / bs

            if grokfast:
                ema_W1 = alpha * ema_W1 + (1 - alpha) * gW1
                ema_b1 = alpha * ema_b1 + (1 - alpha) * gb1
                ema_W2 = alpha * ema_W2 + (1 - alpha) * gW2
                ema_b2 = alpha * ema_b2 + (1 - alpha) * gb2
                gW1 = gW1 + lam * ema_W1
                gb1 = gb1 + lam * ema_b1
                gW2 = gW2 + lam * ema_W2
                gb2 = gb2 + lam * ema_b2

            W2 -= LR * (gW2 + WD * W2)
            b2 -= LR * (gb2 + WD * b2)
            W1 -= LR * (gW1 + WD * W1)
            b1 -= LR * (gb1 + WD * b1)

        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = float(np.mean(np.sign(te_out) == y_te))

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= target_acc and solve_epoch < 0:
            solve_epoch = epoch
            break

    elapsed = time.time() - start
    return W1, b1, W2, b2, {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 4),
    }


def run_curriculum(stages, k_sparse, secret, seed, max_epochs_per_phase=500,
                   grokfast=False):
    phase_details = []
    total_start = time.time()
    total_epochs = 0

    n0 = stages[0]
    W1, b1, W2, b2 = init_weights(HIDDEN, n0, seed=seed + 1)
    W1, b1, W2, b2, phase_r = train_phase(
        W1, b1, W2, b2, n0, k_sparse, secret,
        max_epochs=max_epochs_per_phase, rng_seed=seed,
        grokfast=grokfast, alpha=GF_ALPHA, lam=GF_LAM,
    )
    total_epochs += phase_r['total_epochs']
    phase_details.append({'n': n0, **phase_r})

    for i, n_next in enumerate(stages[1:], start=2):
        n_prev = stages[i - 2]
        W1 = expand_W1(W1, n_prev, n_next, rng_seed=seed + i * 100)
        W1, b1, W2, b2, phase_r = train_phase(
            W1, b1, W2, b2, n_next, k_sparse, secret,
            max_epochs=max_epochs_per_phase, rng_seed=seed + i * 1000,
            grokfast=grokfast, alpha=GF_ALPHA, lam=GF_LAM,
        )
        total_epochs += phase_r['total_epochs']
        phase_details.append({'n': n_next, **phase_r})

    total_elapsed = time.time() - total_start
    return {
        'best_test_acc': phase_r['best_test_acc'],
        'total_epochs': total_epochs,
        'elapsed_s': round(total_elapsed, 4),
        'phases': phase_details,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  EXPERIMENT: GrokFast + Curriculum Scaling")
    print("  How far can we push n with GrokFast + Curriculum?")
    print("=" * 78)

    all_results = {}

    # Test configs: scaling n with k=3 and k=5
    test_cases = [
        # (label, stages, k, max_epochs_per_phase)
        ("n=50/k=3",   [10, 30, 50],          3, 500),
        ("n=100/k=3",  [10, 30, 50, 100],     3, 500),
        ("n=200/k=3",  [10, 30, 50, 100, 200], 3, 500),
        ("n=50/k=5",   [10, 30, 50],          5, 500),
        ("n=100/k=5",  [10, 30, 50, 100],     5, 500),
        ("n=200/k=5",  [10, 30, 50, 100, 200], 5, 500),
    ]

    rng = np.random.RandomState(42)
    # Secrets with all indices < 10 (valid at smallest curriculum stage)
    secret_k3 = sorted(rng.choice(10, 3, replace=False).tolist())
    secret_k5 = sorted(rng.choice(10, 5, replace=False).tolist())

    for label, stages, k, max_ep in test_cases:
        secret = secret_k3 if k == 3 else secret_k5
        print(f"\n{'─'*78}")
        print(f"  {label}  stages={stages}  secret={secret}")
        print(f"{'─'*78}")

        methods = {}

        # GrokFast + Curriculum
        accs, epochs, times, all_phases = [], [], [], []
        for seed in SEEDS:
            r = run_curriculum(stages, k, secret, seed, max_ep, grokfast=True)
            accs.append(r['best_test_acc'])
            epochs.append(r['total_epochs'])
            times.append(r['elapsed_s'])
            all_phases.append(r['phases'])

        solve_rate = sum(1 for a in accs if a >= 0.95) / len(SEEDS)
        avg_ep = sum(epochs) / len(epochs)
        avg_t = sum(times) / len(times)
        print(f"  GrokFast+Curr: solve={solve_rate:.0%}  epochs={avg_ep:.0f}  "
              f"time={avg_t:.3f}s  accs={[f'{a:.0%}' for a in accs]}")

        # Show phase breakdown for first seed
        if all_phases[0]:
            for p in all_phases[0]:
                pstatus = "SOLVED" if p['best_test_acc'] >= 0.95 else f"{p['best_test_acc']:.0%}"
                print(f"    phase n={p['n']}: {pstatus} in {p['total_epochs']} epochs, {p['elapsed_s']:.3f}s")

        methods['grokfast_curriculum'] = {
            'solve_rate': solve_rate, 'avg_epochs': avg_ep, 'avg_time': avg_t,
            'accs': accs, 'epochs': epochs, 'times': times,
            'phases_seed42': all_phases[0],
        }

        # Curriculum only (for comparison)
        accs2, epochs2, times2 = [], [], []
        for seed in SEEDS:
            r = run_curriculum(stages, k, secret, seed, max_ep, grokfast=False)
            accs2.append(r['best_test_acc'])
            epochs2.append(r['total_epochs'])
            times2.append(r['elapsed_s'])

        solve_rate2 = sum(1 for a in accs2 if a >= 0.95) / len(SEEDS)
        avg_ep2 = sum(epochs2) / len(epochs2)
        avg_t2 = sum(times2) / len(times2)
        print(f"  Curriculum:    solve={solve_rate2:.0%}  epochs={avg_ep2:.0f}  "
              f"time={avg_t2:.3f}s  accs={[f'{a:.0%}' for a in accs2]}")

        methods['curriculum'] = {
            'solve_rate': solve_rate2, 'avg_epochs': avg_ep2, 'avg_time': avg_t2,
            'accs': accs2, 'epochs': epochs2, 'times': times2,
        }

        all_results[label] = methods

    # --- Summary ---
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  {'Config':<14} {'Method':<22} {'Solve%':>7} {'Epochs':>8} {'Time':>10}")
    print(f"  {'─'*14} {'─'*22} {'─'*7} {'─'*8} {'─'*10}")

    for label, methods in all_results.items():
        for method_name, r in methods.items():
            short = "GF+Curr" if "grokfast" in method_name else "Curr"
            print(f"  {label:<14} {short:<22} {r['solve_rate']:>6.0%} "
                  f"{r['avg_epochs']:>8.0f} {r['avg_time']:>9.3f}s")
        print()

    # --- Save ---
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {RESULTS_DIR / 'results.json'}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
