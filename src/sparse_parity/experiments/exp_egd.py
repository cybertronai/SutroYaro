#!/usr/bin/env python3
"""
Experiment: Egalitarian Gradient Descent (EGD) on sparse parity.

Hypothesis: EGD eliminates the grokking plateau by equalizing learning rates
across all gradient directions (arXiv:2510.04930). If the plateau shrinks,
fewer epochs are needed, opening the path to sub-10ms.

Answers: TODO.md "SGD Under 10ms" / EGD hypothesis
         DISCOVERIES.md Q6 (tiled W1) tangentially

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.experiments.exp_egd
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_parity.config import Config


def generate(config):
    rng = np.random.RandomState(config.seed)
    secret = sorted(rng.choice(config.n_bits, config.k_sparse, replace=False).tolist())

    def make(n):
        x = rng.choice([-1.0, 1.0], size=(n, config.n_bits))
        y = np.prod(x[:, secret], axis=1)
        return x, y

    x_tr, y_tr = make(config.n_train)
    x_te, y_te = make(config.n_test)
    return x_tr, y_tr, x_te, y_te, secret


def egd_matrix(G, eps=1e-8):
    """EGD transform for matrix gradient: replace singular values with 1."""
    U, S, Vt = np.linalg.svd(G, full_matrices=False)
    return U @ Vt


def egd_vector(g, eps=1e-8):
    """EGD transform for vector gradient: normalize to unit norm."""
    n = np.linalg.norm(g)
    if n < eps:
        return g
    return g / n


def train(config, use_egd=False, verbose=True):
    """Training loop with optional EGD. Returns dict with results."""
    x_tr, y_tr, x_te, y_te, secret = generate(config)

    rng = np.random.RandomState(config.seed + 1)
    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    method = "egd" if use_egd else "sgd"
    if verbose:
        print(f"  [{config.n_bits}-bit, k={config.k_sparse}, {method}] secret={secret}, "
              f"n_train={config.n_train}, lr={config.lr}, hidden={config.hidden}")

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1
    epoch_90 = -1

    for epoch in range(1, config.max_epochs + 1):
        idx = np.arange(config.n_train)
        rng.shuffle(idx)

        for b_start in range(0, config.n_train, config.batch_size):
            b_end = min(b_start + config.batch_size, config.n_train)
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

            # Backward (only violated samples)
            xm = xb[mask]
            ym = yb[mask]
            hm = h[mask]
            h_pre_m = h_pre[mask]

            dout = -ym
            dW2 = dout[:, None] * hm
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre_m > 0)
            dW1 = dh_pre.T @ xm
            db1 = dh_pre.sum(axis=0)

            # Average gradients
            gW1 = dW1 / bs
            gb1 = db1 / bs
            gW2 = dW2.sum(axis=0, keepdims=True) / bs
            gb2 = db2 / bs

            if use_egd:
                # EGD: replace gradient singular values with 1
                gW1 = egd_matrix(gW1)
                gb1 = egd_vector(gb1)
                gW2 = egd_matrix(gW2)
                # gb2 is scalar, just use sign
                gb2 = np.sign(gb2) if abs(gb2) > 1e-8 else 0.0

            # Update with weight decay
            W1 -= config.lr * (gW1 + config.wd * W1)
            b1 -= config.lr * (gb1 + config.wd * b1)
            W2 -= config.lr * (gW2 + config.wd * W2)
            b2 -= config.lr * (gb2 + config.wd * b2)

        # Evaluate
        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = np.mean(np.sign(te_out) == y_te)
        tr_out = (np.maximum(x_tr @ W1.T + b1, 0) @ W2.T + b2).ravel()
        tr_acc = np.mean(np.sign(tr_out) == y_tr)

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 0.90 and epoch_90 < 0:
            epoch_90 = epoch
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch

        if verbose and (epoch % 10 == 0 or epoch == 1 or te_acc >= 0.90):
            print(f"    epoch {epoch:>4}: train={tr_acc:.0%} test={te_acc:.0%}")

        if te_acc >= 1.0:
            break

    elapsed = time.time() - start
    if verbose:
        print(f"  Result: {best_acc:.0%} in {elapsed:.3f}s ({epoch} epochs)")

    return {
        'method': method,
        'best_test_acc': round(float(best_acc), 4),
        'solve_epoch': solve_epoch,
        'epoch_90': epoch_90,
        'total_epochs': epoch,
        'elapsed_s': round(elapsed, 4),
        'secret': secret,
        'n_bits': config.n_bits,
        'k_sparse': config.k_sparse,
        'n_train': config.n_train,
        'hidden': config.hidden,
        'lr': config.lr,
        'wd': config.wd,
        'batch_size': config.batch_size,
        'max_epochs': config.max_epochs,
    }


def run_config(label, n_bits, k_sparse, n_train, hidden, lr, wd, max_epochs,
               batch_size, seeds, use_egd, verbose=True):
    """Run multiple seeds for one config. Returns list of results."""
    results = []
    for seed in seeds:
        config = Config(
            n_bits=n_bits, k_sparse=k_sparse, hidden=hidden,
            lr=lr, wd=wd, max_epochs=max_epochs,
            n_train=n_train, n_test=500, seed=seed,
        )
        config.batch_size = batch_size
        r = train(config, use_egd=use_egd, verbose=(verbose and seed == seeds[0]))
        results.append(r)
        if not verbose or seed != seeds[0]:
            status = "SOLVED" if r['best_test_acc'] >= 0.95 else f"{r['best_test_acc']:.0%}"
            print(f"    seed={seed}: {r['elapsed_s']:.3f}s  {status}  "
                  f"(ep90={r['epoch_90']}, solve={r['solve_epoch']})")
    return results


def main():
    print("=" * 70)
    print("  EXPERIMENT: Egalitarian Gradient Descent (EGD)")
    print("  Hypothesis: EGD eliminates grokking plateau (arXiv:2510.04930)")
    print("=" * 70)

    seeds = [42, 43, 44, 45, 46]
    all_results = {}

    # =================================================================
    # PART 1: Grokking elimination (n=20/k=3, standard config)
    # =================================================================
    print("\n" + "=" * 70)
    print("  PART 1: Does EGD eliminate the grokking plateau?")
    print("  Config: n=20, k=3, hidden=200, n_train=1000, batch=32")
    print("=" * 70)

    print("\n  --- SGD baseline (lr=0.1) ---")
    all_results['sgd_baseline'] = run_config(
        "sgd_baseline", 20, 3, n_train=1000, hidden=200, lr=0.1, wd=0.01,
        max_epochs=200, batch_size=32, seeds=seeds, use_egd=False)

    # EGD may need different lr since gradient magnitudes change
    for lr in [0.1, 0.05, 0.01, 0.005]:
        label = f"egd_lr{lr}"
        print(f"\n  --- EGD (lr={lr}) ---")
        all_results[label] = run_config(
            label, 20, 3, n_train=1000, hidden=200, lr=lr, wd=0.01,
            max_epochs=200, batch_size=32, seeds=seeds, use_egd=True)

    # =================================================================
    # PART 2: Sub-10ms push (small hidden, fewer samples)
    # =================================================================
    print("\n" + "=" * 70)
    print("  PART 2: Can EGD break 10ms? (small configs)")
    print("=" * 70)

    for hidden, n_train, batch_size in [
        (50, 500, 32),
        (50, 200, 32),
        (100, 500, 32),
        (50, 500, 64),
    ]:
        for lr in [0.05, 0.01]:
            label = f"egd_h{hidden}_n{n_train}_b{batch_size}_lr{lr}"
            print(f"\n  --- EGD ({label}) ---")
            all_results[label] = run_config(
                label, 20, 3, n_train=n_train, hidden=hidden, lr=lr, wd=0.01,
                max_epochs=200, batch_size=batch_size, seeds=seeds, use_egd=True)

        # SGD baseline at same config for fair comparison
        label = f"sgd_h{hidden}_n{n_train}_b{batch_size}"
        print(f"\n  --- SGD ({label}) ---")
        all_results[label] = run_config(
            label, 20, 3, n_train=n_train, hidden=hidden, lr=0.1, wd=0.01,
            max_epochs=200, batch_size=batch_size, seeds=seeds, use_egd=False)

    # =================================================================
    # Print comparison table
    # =================================================================
    print("\n\n" + "=" * 90)
    print("  COMPARISON TABLE")
    print("=" * 90)
    header = (f"  {'Config':<40} | {'Acc':>5} | {'Ep90':>5} | "
              f"{'Solve':>5} | {'Time':>8} | {'Ok':>5}")
    print(header)
    print("  " + "-" * 86)

    for key, runs in all_results.items():
        avg_acc = np.mean([r['best_test_acc'] for r in runs])
        ep90s = [r['epoch_90'] for r in runs if r['epoch_90'] > 0]
        avg_ep90 = np.mean(ep90s) if ep90s else float('nan')
        solves = [r['solve_epoch'] for r in runs if r['solve_epoch'] > 0]
        avg_solve = np.mean(solves) if solves else float('nan')
        avg_time = np.mean([r['elapsed_s'] for r in runs])
        n_solved = sum(1 for r in runs if r['best_test_acc'] >= 0.95)
        ep90_str = f"{avg_ep90:.0f}" if not np.isnan(avg_ep90) else "---"
        solve_str = f"{avg_solve:.0f}" if not np.isnan(avg_solve) else "---"
        print(f"  {key:<40} | {avg_acc:>5.1%} | {ep90_str:>5} | "
              f"{solve_str:>5} | {avg_time:>7.3f}s | {n_solved}/{len(runs)}")

    print("=" * 90)

    # =================================================================
    # Save results
    # =================================================================
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_egd'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'experiment': 'exp_egd',
            'description': 'EGD vs SGD on sparse parity: grokking elimination + sub-10ms push',
            'hypothesis': 'EGD equalizes gradient directions, eliminating grokking plateau',
            'reference': 'arXiv:2510.04930',
            'configs': all_results,
        }, f, indent=2)

    print(f"\n  Results saved to: {results_path}")
    return all_results


if __name__ == '__main__':
    main()
