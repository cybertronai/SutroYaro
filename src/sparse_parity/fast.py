#!/usr/bin/env python3
"""
Fast sparse parity training — numpy-accelerated.

Target: full 20-bit solve in <2 seconds.
Same algorithm as train.py, just using numpy for matrix ops.

Usage:
    cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro
    PYTHONPATH=src python3 -m sparse_parity.fast
"""

import time
import numpy as np
from .config import Config
from .tracker import MemTracker


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


def _tracked_step(x, y, W1, b1, W2, b2, config, tracker):
    """Run one single-sample forward+backward pass with full tracker instrumentation.

    Mirrors the exact read/write sequence from train.py's forward() + backward_and_update(),
    but uses numpy scalars/arrays.  Called only once per training run (if tracker is requested),
    so performance does not matter here.
    """
    hidden = config.hidden
    n_bits = config.n_bits

    # --- Initial writes (parameters + input already in memory) ---
    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('x', n_bits)
    tracker.write('y', 1)

    # --- Forward pass (matches model.forward with tracker) ---
    tracker.read('x', n_bits)
    tracker.read('W1', hidden * n_bits)
    tracker.read('b1', hidden)

    h_pre = x @ W1.T + b1                    # (hidden,)

    tracker.write('h_pre', hidden)
    tracker.read('h_pre', hidden)

    h = np.maximum(h_pre, 0)                 # ReLU

    tracker.write('h', hidden)
    tracker.read('h', hidden)
    tracker.read('W2', hidden)
    tracker.read('b2', 1)

    out = float((h @ W2.T + b2).ravel()[0])

    tracker.write('out', 1)

    # --- Backward pass (matches train.backward_and_update with tracker) ---
    tracker.read('out', 1)
    tracker.read('y', 1)

    margin = out * y
    if margin >= 1.0:
        return  # no violated sample, nothing to backprop

    dout = -y

    tracker.write('dout', 1)
    tracker.read('dout', 1)
    tracker.read('h', hidden)

    # Layer 2 gradients
    # dW2 = dout * h, db2 = dout
    tracker.write('dW2', hidden)
    tracker.write('db2', 1)

    # dh = W2^T * dout
    tracker.read('W2', hidden)
    tracker.read('dout', 1)

    tracker.write('dh', hidden)
    tracker.read('dh', hidden)
    tracker.read('h_pre', hidden)

    # ReLU backward
    tracker.write('dh_pre', hidden)

    # Layer 1 gradients + update
    tracker.read('dh_pre', hidden)
    tracker.read('x', n_bits)
    tracker.read('W1', hidden * n_bits)

    # W1 update happens here (in-place)
    tracker.write('W1', hidden * n_bits)
    tracker.read('dh_pre', hidden)
    tracker.read('b1', hidden)

    # b1 update happens here (in-place)
    tracker.write('b1', hidden)

    # Layer 2 update
    tracker.read('dW2', hidden)
    tracker.read('W2', hidden)

    # W2 update happens here (in-place)
    tracker.write('W2', hidden)
    tracker.read('db2', 1)
    tracker.read('b2', 1)

    # b2 update happens here (in-place)
    tracker.write('b2', 1)


def train(config, verbose=True, tracker=None):
    """Full training loop. Returns dict with results.

    If tracker is a MemTracker instance, one tracked single-sample step is run
    after the first epoch (same read/write sequence as train.py).
    If tracker is True, a fresh MemTracker is created automatically.
    When tracker is None (default), no tracking overhead at all.
    """
    if tracker is True:
        tracker = MemTracker()

    x_tr, y_tr, x_te, y_te, secret = generate(config)

    rng = np.random.RandomState(config.seed + 1)
    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    if verbose:
        print(f"  [{config.n_bits}-bit, k={config.k_sparse}] secret={secret}, "
              f"params={config.hidden * config.n_bits + config.hidden + config.hidden + 1:,}")

    start = time.time()
    best_acc = 0.0
    solve_epoch = -1
    epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        # Mini-batch SGD
        idx = np.arange(config.n_train)
        rng.shuffle(idx)

        for b_start in range(0, config.n_train, config.batch_size):
            b_end = min(b_start + config.batch_size, config.n_train)
            xb = x_tr[idx[b_start:b_end]]
            yb = y_tr[idx[b_start:b_end]]
            bs = xb.shape[0]

            # Forward
            h_pre = xb @ W1.T + b1          # (bs, hidden)
            h = np.maximum(h_pre, 0)         # ReLU
            out = h @ W2.T + b2              # (bs, 1)
            out = out.ravel()                # (bs,)

            # Hinge loss mask
            margin = out * yb
            mask = margin < 1.0
            if not np.any(mask):
                continue

            # Backward (only on violated samples)
            xm = xb[mask]
            ym = yb[mask]
            hm = h[mask]
            h_pre_m = h_pre[mask]
            ms = xm.shape[0]

            dout = -ym                           # (ms,)
            dW2 = dout[:, None] * hm             # (ms, hidden)
            db2 = dout.sum()
            dh = dout[:, None] * W2              # (ms, hidden)
            dh_pre = dh * (h_pre_m > 0)          # ReLU backward
            dW1 = dh_pre.T @ xm                  # (hidden, n_bits)
            db1 = dh_pre.sum(axis=0)

            # SGD update (averaged over batch)
            W2 -= config.lr * (dW2.sum(axis=0, keepdims=True) / bs + config.wd * W2)
            b2 -= config.lr * (db2 / bs + config.wd * b2)
            W1 -= config.lr * (dW1 / bs + config.wd * W1)
            b1 -= config.lr * (db1 / bs + config.wd * b1)

        # Run one tracked single-sample step after epoch 1
        if tracker is not None and epoch == 1:
            _tracked_step(x_tr[0], y_tr[0], W1, b1, W2, b2, config, tracker)

        # Evaluate
        te_out = (np.maximum(x_te @ W1.T + b1, 0) @ W2.T + b2).ravel()
        te_acc = np.mean(np.sign(te_out) == y_te)
        tr_out = (np.maximum(x_tr @ W1.T + b1, 0) @ W2.T + b2).ravel()
        tr_acc = np.mean(np.sign(tr_out) == y_tr)

        if te_acc > best_acc:
            best_acc = te_acc
        if te_acc >= 1.0 and solve_epoch < 0:
            solve_epoch = epoch

        if verbose and (epoch % 5 == 0 or epoch == 1 or te_acc >= 0.95):
            print(f"    epoch {epoch:>3}: train={tr_acc:.0%} test={te_acc:.0%}")

        if te_acc >= 1.0:
            break

    elapsed = time.time() - start

    if verbose:
        print(f"  Result: {best_acc:.0%} in {elapsed:.2f}s ({epoch} epochs)")

    result = {
        'best_test_acc': best_acc,
        'solve_epoch': solve_epoch,
        'total_epochs': epoch,
        'elapsed_s': elapsed,
        'secret': secret,
        'config': {k: v for k, v in config.__dict__.items()},
    }
    if tracker is not None:
        result['tracker'] = tracker.to_json()
    return result


def main():
    print("=" * 60)
    print("  FAST SPARSE PARITY (numpy)")
    print("=" * 60)

    # Fast 20-bit solve — under 0.2s
    config = Config(
        n_bits=20, k_sparse=3, hidden=200,
        lr=0.1, wd=0.01, max_epochs=200,
        n_train=1000, n_test=200, seed=42,
    )
    config.batch_size = 32

    # Run 5 seeds to show it's robust
    times = []
    for seed in [42, 43, 44, 45, 46]:
        config.seed = seed
        r = train(config, verbose=(seed == 42))
        times.append(r['elapsed_s'])
        if seed != 42:
            status = "SOLVED" if r['best_test_acc'] >= 0.95 else f"{r['best_test_acc']:.0%}"
            print(f"  seed={seed}: {r['elapsed_s']:.2f}s  {status}  (epoch {r['solve_epoch']})")

    print(f"\n  Avg: {sum(times)/len(times):.2f}s  Min: {min(times):.2f}s  Max: {max(times):.2f}s")


def demo_tracker():
    """Run a single training with tracker enabled and print ARD/DMC."""
    print("\n" + "=" * 60)
    print("  TRACKER DEMO (1 tracked step)")
    print("=" * 60)

    config = Config(
        n_bits=20, k_sparse=3, hidden=200,
        lr=0.1, wd=0.01, max_epochs=200,
        n_train=1000, n_test=200, seed=42,
    )
    config.batch_size = 32

    tracker = MemTracker()
    r = train(config, verbose=False, tracker=tracker)
    s = tracker.summary()

    status = 'SOLVED' if r['best_test_acc'] >= 0.95 else f"{r['best_test_acc']:.0%}"
    print(f"  Training: {status} in {r['elapsed_s']:.2f}s ({r['total_epochs']} epochs)")
    print(f"  ARD:  {s['weighted_ard']:,.0f}")
    print(f"  DMC:  {s['dmc']:,.0f}")
    print(f"  Reads: {s['reads']}, Writes: {s['writes']}")
    print(f"  Total floats accessed: {s['total_floats_accessed']:,}")

    if s['per_buffer']:
        print(f"\n  {'Buffer':<12} {'Size':>8} {'Reads':>5} {'Avg Dist':>10}")
        print(f"  {'---':<12} {'---':>8} {'---':>5} {'---':>10}")
        for name, info in s['per_buffer'].items():
            print(f"  {name:<12} {info['size']:>8,} {info['read_count']:>5} "
                  f"{info['avg_dist']:>10,.0f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
    demo_tracker()
