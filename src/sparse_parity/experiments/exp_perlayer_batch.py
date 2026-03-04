#!/usr/bin/env python3
"""
Experiment: Per-layer + batching combined (numpy-accelerated).

Question: Does per-layer forward-backward combine with mini-batch training?
          We know per-layer gives 3.8% ARD improvement (exp_a, exp_c) and
          batching helps convergence speed (exp1). What happens together?

Method:
  Compare 4 variants on n=20, k=3:
    1. standard + single-sample
    2. standard + batch=32
    3. perlayer + single-sample
    4. perlayer + batch=32

  Per-layer + batch means: for each mini-batch of 32 samples,
    - Forward all 32 through layer 1, compute layer 1 gradients, update W1/b1
    - Forward all 32 through layer 2 (using UPDATED W1), compute layer 2 gradients, update W2/b2

  Config: n=20, k=3, hidden=200, n_train=1000, batch=32, 5 seeds
  Report: avg epochs_to_solve, avg wall_time for each variant

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.experiments.exp_perlayer_batch
"""

import time
import json
import numpy as np
from pathlib import Path


def generate(n_bits, k_sparse, n_train, n_test, seed):
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    def make(n):
        x = rng.choice([-1.0, 1.0], size=(n, n_bits))
        y = np.prod(x[:, secret], axis=1)
        return x, y

    x_tr, y_tr = make(n_train)
    x_te, y_te = make(n_test)
    return x_tr, y_tr, x_te, y_te, secret


def init_weights(n_bits, hidden, seed):
    rng = np.random.RandomState(seed + 1)
    std1 = np.sqrt(2.0 / n_bits)
    std2 = np.sqrt(2.0 / hidden)
    W1 = rng.randn(hidden, n_bits) * std1
    b1 = np.zeros(hidden)
    W2 = rng.randn(1, hidden) * std2
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def evaluate(x, y, W1, b1, W2, b2):
    out = (np.maximum(x @ W1.T + b1, 0) @ W2.T + b2).ravel()
    return np.mean(np.sign(out) == y)


# ---------------------------------------------------------------------------
# Variant 1: Standard backprop, single-sample SGD
# ---------------------------------------------------------------------------

def train_standard_single(x_tr, y_tr, x_te, y_te, W1, b1, W2, b2,
                          lr, wd, max_epochs):
    start = time.time()
    n = len(x_tr)
    rng = np.random.RandomState(0)
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n)
        rng.shuffle(idx)

        for i in idx:
            xb = x_tr[i:i+1]
            yb = y_tr[i:i+1]

            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)
            out = (h @ W2.T + b2).ravel()

            margin = out * yb
            if margin[0] >= 1.0:
                continue

            dout = -yb
            dW2 = dout[:, None] * h
            db2 = dout.sum()
            dh = dout[:, None] * W2
            dh_pre = dh * (h_pre > 0)
            dW1 = dh_pre.T @ xb
            db1 = dh_pre.sum(axis=0)

            W2 -= lr * (dW2 + wd * W2)
            b2 -= lr * (db2 + wd * b2)
            W1 -= lr * (dW1 + wd * W1)
            b1 -= lr * (db1 + wd * b1)

        te_acc = evaluate(x_te, y_te, W1, b1, W2, b2)
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch
            break

    return {'solve_epoch': solve_epoch, 'epochs': epoch,
            'wall_time': time.time() - start, 'final_acc': te_acc}


# ---------------------------------------------------------------------------
# Variant 2: Standard backprop, mini-batch SGD
# ---------------------------------------------------------------------------

def train_standard_batch(x_tr, y_tr, x_te, y_te, W1, b1, W2, b2,
                         lr, wd, max_epochs, batch_size):
    start = time.time()
    n = len(x_tr)
    rng = np.random.RandomState(0)
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n)
        rng.shuffle(idx)

        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
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

            W2 -= lr * (dW2.sum(axis=0, keepdims=True) / bs + wd * W2)
            b2 -= lr * (db2 / bs + wd * b2)
            W1 -= lr * (dW1 / bs + wd * W1)
            b1 -= lr * (db1 / bs + wd * b1)

        te_acc = evaluate(x_te, y_te, W1, b1, W2, b2)
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch
            break

    return {'solve_epoch': solve_epoch, 'epochs': epoch,
            'wall_time': time.time() - start, 'final_acc': te_acc}


# ---------------------------------------------------------------------------
# Variant 3: Per-layer, single-sample SGD
# ---------------------------------------------------------------------------

def train_perlayer_single(x_tr, y_tr, x_te, y_te, W1, b1, W2, b2,
                          lr, wd, max_epochs):
    start = time.time()
    n = len(x_tr)
    rng = np.random.RandomState(0)
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n)
        rng.shuffle(idx)

        for i in idx:
            xb = x_tr[i:i+1]
            yb = y_tr[i:i+1]

            # Layer 1 forward
            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)

            # Layer 2 forward
            out = (h @ W2.T + b2).ravel()

            margin = out * yb
            if margin[0] >= 1.0:
                continue

            dout = -yb

            # Layer 2 backward + update
            dW2 = dout[:, None] * h
            db2 = dout.sum()
            dh = dout[:, None] * W2

            W2 -= lr * (dW2 + wd * W2)
            b2 -= lr * (db2 + wd * b2)

            # Layer 1 backward + update
            dh_pre = dh * (h_pre > 0)
            dW1 = dh_pre.T @ xb
            db1 = dh_pre.sum(axis=0)

            W1 -= lr * (dW1 + wd * W1)
            b1 -= lr * (db1 + wd * b1)

        te_acc = evaluate(x_te, y_te, W1, b1, W2, b2)
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch
            break

    return {'solve_epoch': solve_epoch, 'epochs': epoch,
            'wall_time': time.time() - start, 'final_acc': te_acc}


# ---------------------------------------------------------------------------
# Variant 4: Per-layer, mini-batch SGD
# ---------------------------------------------------------------------------

def train_perlayer_batch(x_tr, y_tr, x_te, y_te, W1, b1, W2, b2,
                         lr, wd, max_epochs, batch_size):
    """
    Per-layer + batching:
    For each mini-batch:
      1. Forward all samples through layer 1 (W1, b1, ReLU)
      2. Compute layer 1 gradients from the output error signal
      3. Update W1, b1 immediately
      4. Forward all samples through layer 2 using UPDATED W1
      5. Compute layer 2 gradients
      6. Update W2, b2
    """
    start = time.time()
    n = len(x_tr)
    rng = np.random.RandomState(0)
    solve_epoch = -1

    for epoch in range(1, max_epochs + 1):
        idx = np.arange(n)
        rng.shuffle(idx)

        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            # --- Layer 1 forward ---
            h_pre = xb @ W1.T + b1
            h = np.maximum(h_pre, 0)

            # --- Layer 2 forward (to get error signal) ---
            out = (h @ W2.T + b2).ravel()

            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            xm = xb[mask]
            ym = yb[mask]
            hm = h[mask]
            h_pre_m = h_pre[mask]

            dout = -ym

            # --- Full backward to get Layer 1 gradients ---
            dh = dout[:, None] * W2              # (ms, hidden)
            dh_pre = dh * (h_pre_m > 0)          # ReLU backward
            dW1 = dh_pre.T @ xm                  # (hidden, n_bits)
            db1_grad = dh_pre.sum(axis=0)

            # --- Update W1, b1 FIRST ---
            W1 -= lr * (dW1 / bs + wd * W1)
            b1 -= lr * (db1_grad / bs + wd * b1)

            # --- Re-forward through layer 2 with UPDATED W1 ---
            h_pre_new = xb @ W1.T + b1
            h_new = np.maximum(h_pre_new, 0)
            out_new = (h_new @ W2.T + b2).ravel()

            margin_new = out_new * yb
            mask_new = margin_new < 1.0
            if not np.any(mask_new):
                continue

            hm_new = h_new[mask_new]
            ym_new = yb[mask_new]

            dout_new = -ym_new
            dW2 = dout_new[:, None] * hm_new
            db2_grad = dout_new.sum()

            # --- Update W2, b2 ---
            W2 -= lr * (dW2.sum(axis=0, keepdims=True) / bs + wd * W2)
            b2 -= lr * (db2_grad / bs + wd * b2)

        te_acc = evaluate(x_te, y_te, W1, b1, W2, b2)
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch
            break

    return {'solve_epoch': solve_epoch, 'epochs': epoch,
            'wall_time': time.time() - start, 'final_acc': te_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Config
    N_BITS = 20
    K_SPARSE = 3
    HIDDEN = 200
    LR = 0.1
    WD = 0.01
    MAX_EPOCHS = 200
    N_TRAIN = 1000
    N_TEST = 200
    BATCH_SIZE = 32
    SEEDS = [42, 43, 44, 45, 46]

    variants = [
        ('standard+single', lambda *a: train_standard_single(*a, LR, WD, MAX_EPOCHS)),
        ('standard+batch',  lambda *a: train_standard_batch(*a, LR, WD, MAX_EPOCHS, BATCH_SIZE)),
        ('perlayer+single', lambda *a: train_perlayer_single(*a, LR, WD, MAX_EPOCHS)),
        ('perlayer+batch',  lambda *a: train_perlayer_batch(*a, LR, WD, MAX_EPOCHS, BATCH_SIZE)),
    ]

    print("=" * 70)
    print("  EXP PERLAYER+BATCH: Per-layer + batching combined")
    print("=" * 70)
    print(f"  n={N_BITS}, k={K_SPARSE}, hidden={HIDDEN}, lr={LR}, wd={WD}")
    print(f"  n_train={N_TRAIN}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    all_results = {}

    for variant_name, train_fn in variants:
        print(f"\n  --- {variant_name} ---")
        seed_results = []

        for seed in SEEDS:
            x_tr, y_tr, x_te, y_te, secret = generate(
                N_BITS, K_SPARSE, N_TRAIN, N_TEST, seed)
            W1, b1, W2, b2 = init_weights(N_BITS, HIDDEN, seed)

            res = train_fn(x_tr, y_tr, x_te, y_te, W1, b1, W2, b2)
            seed_results.append(res)

            status = "SOLVED" if res['solve_epoch'] > 0 else f"{res['final_acc']:.0%}"
            print(f"    seed={seed}: epoch={res['solve_epoch']:>4}  "
                  f"time={res['wall_time']:.3f}s  {status}")

        solved = [r for r in seed_results if r['solve_epoch'] > 0]
        n_solved = len(solved)
        avg_epoch = sum(r['solve_epoch'] for r in solved) / n_solved if solved else -1
        avg_time = sum(r['wall_time'] for r in seed_results) / len(seed_results)

        all_results[variant_name] = {
            'seeds': len(SEEDS),
            'solved': n_solved,
            'avg_solve_epoch': round(avg_epoch, 1) if solved else None,
            'avg_wall_time': round(avg_time, 4),
            'per_seed': [
                {'seed': s, 'solve_epoch': r['solve_epoch'],
                 'wall_time': round(r['wall_time'], 4),
                 'final_acc': round(r['final_acc'], 4)}
                for s, r in zip(SEEDS, seed_results)
            ],
        }

        print(f"    => {n_solved}/{len(SEEDS)} solved, "
              f"avg epoch={avg_epoch:.1f}, avg time={avg_time:.3f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Variant':<22} {'Solved':>7} {'Avg Epoch':>10} {'Avg Time':>10}")
    print(f"  {'─'*22} {'─'*7} {'─'*10} {'─'*10}")
    for name, r in all_results.items():
        ep = f"{r['avg_solve_epoch']:.1f}" if r['avg_solve_epoch'] else "N/A"
        print(f"  {name:<22} {r['solved']}/{r['seeds']:>4} {ep:>10} {r['avg_wall_time']:>9.3f}s")
    print("=" * 70)

    # Save results
    output = {
        'experiment': 'exp_perlayer_batch',
        'question': 'Does per-layer + batching combine for better convergence?',
        'config': {
            'n_bits': N_BITS, 'k_sparse': K_SPARSE, 'hidden': HIDDEN,
            'lr': LR, 'wd': WD, 'max_epochs': MAX_EPOCHS,
            'n_train': N_TRAIN, 'n_test': N_TEST, 'batch_size': BATCH_SIZE,
            'seeds': SEEDS,
        },
        'variants': all_results,
    }

    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_perlayer_batch'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == '__main__':
    main()
