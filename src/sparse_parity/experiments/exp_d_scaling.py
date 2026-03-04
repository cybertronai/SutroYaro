"""
Experiment D: Scale stress test — find where standard SGD breaks on sparse parity.

We solved n=20, k=3. Theory says SGD needs ~n^O(k) iterations.
This experiment maps the frontier: at what n/k does standard SGD become impractical?

Configs tested (in order, with 3-minute timeout each):
  a) n=30, k=3  — should work
  b) n=50, k=3  — might be slow
  c) n=20, k=5  — harder (C(20,5)=15504 subsets)
  d) n=50, k=5  — probably impossible in pure Python

Hyperparams: LR=0.1, batch_size=32, WD=0.01 (winning config from exp1).
"""

import sys
import os
import time
import json
import random
import math
import signal
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward_batch
from sparse_parity.metrics import hinge_loss, accuracy, save_json, timestamp


# ---------------------------------------------------------------------------
# Timeout mechanism
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Config timed out")


TIMEOUT_SECONDS = 180  # 3 minutes per config


# ---------------------------------------------------------------------------
# Mini-batch SGD (copied from exp1 for self-containment)
# ---------------------------------------------------------------------------

def compute_gradients(x, y, W1, b1, W2, b2):
    """Compute gradients for a single sample via backprop."""
    hidden = len(W1)
    n_bits = len(x)

    # Forward
    h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j]
             for j in range(hidden)]
    h = [max(0.0, v) for v in h_pre]
    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    # Hinge loss: max(0, 1 - out*y)
    margin = out * y
    if margin >= 1.0:
        return None

    dout = -y

    dW2 = [[dout * h[j] for j in range(hidden)]]
    db2 = [dout]

    dh = [W2[0][j] * dout for j in range(hidden)]
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    dW1 = [[dh_pre[j] * x[i] for i in range(n_bits)] for j in range(hidden)]
    db1 = [dh_pre[j] for j in range(hidden)]

    return dW1, db1, dW2, db2


def minibatch_sgd_step(batch_x, batch_y, W1, b1, W2, b2, lr, wd):
    """One mini-batch SGD step with gradient accumulation and weight decay."""
    hidden = len(W1)
    n_bits = len(W1[0])

    acc_dW1 = [[0.0] * n_bits for _ in range(hidden)]
    acc_db1 = [0.0] * hidden
    acc_dW2 = [[0.0] * hidden]
    acc_db2 = [0.0]
    n_contributing = 0

    for xi, yi in zip(batch_x, batch_y):
        grads = compute_gradients(xi, yi, W1, b1, W2, b2)
        if grads is None:
            continue
        dW1, db1_g, dW2, db2_g = grads
        n_contributing += 1

        for j in range(hidden):
            for i in range(n_bits):
                acc_dW1[j][i] += dW1[j][i]
            acc_db1[j] += db1_g[j]
            acc_dW2[0][j] += dW2[0][j]
        acc_db2[0] += db2_g[0]

    if n_contributing == 0:
        return

    inv_n = 1.0 / n_contributing

    for j in range(hidden):
        for i in range(n_bits):
            W1[j][i] -= lr * (acc_dW1[j][i] * inv_n + wd * W1[j][i])
        b1[j] -= lr * (acc_db1[j] * inv_n + wd * b1[j])
        W2[0][j] -= lr * (acc_dW2[0][j] * inv_n + wd * W2[0][j])
    b2[0] -= lr * (acc_db2[0] * inv_n + wd * b2[0])


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------

def run_one_config(n_bits, k_sparse, max_epochs=200, seed=42):
    """
    Train sparse parity with given n/k.
    Returns dict with epochs_to_90pct, total_steps, wall_time, final_accuracy, status.
    """
    hidden = min(2 * n_bits, 1000)
    n_train = max(500, 10 * n_bits)
    n_test = 200
    batch_size = 32
    lr = 0.1
    wd = 0.01

    config = Config(
        n_bits=n_bits,
        k_sparse=k_sparse,
        n_train=n_train,
        n_test=n_test,
        hidden=hidden,
        lr=lr,
        wd=wd,
        max_epochs=max_epochs,
        seed=seed,
    )

    label = f"n={n_bits}, k={k_sparse}"
    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  hidden={hidden}, n_train={n_train}, max_epochs={max_epochs}")
    print(f"  Theoretical complexity: ~n^k = {n_bits}^{k_sparse} = {n_bits**k_sparse:,}")
    print(f"  C(n,k) = {math.comb(n_bits, k_sparse):,} possible subsets")
    print(f"{'='*60}")

    # Generate data
    x_train, y_train, x_test, y_test, secret = generate(config)
    print(f"  Secret indices: {secret}")

    # Initialize model
    W1, b1, W2, b2 = init_params(config)

    total_steps = 0
    epochs_to_90pct = -1
    best_test_acc = 0.0
    status = "completed"

    start_time = time.time()

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)

    try:
        for epoch in range(1, max_epochs + 1):
            # Shuffle training data
            indices = list(range(n_train))
            rng = random.Random(seed + epoch)
            rng.shuffle(indices)

            # Mini-batch training
            for batch_start in range(0, n_train, batch_size):
                batch_end = min(batch_start + batch_size, n_train)
                batch_idx = indices[batch_start:batch_end]
                batch_x = [x_train[i] for i in batch_idx]
                batch_y = [y_train[i] for i in batch_idx]

                minibatch_sgd_step(batch_x, batch_y, W1, b1, W2, b2, lr, wd)
                total_steps += 1

            # Evaluate every epoch (cheap relative to training for large n)
            te_outs = forward_batch(x_test, W1, b1, W2, b2)
            te_acc = accuracy(te_outs, y_test)

            if te_acc > best_test_acc:
                best_test_acc = te_acc

            if te_acc >= 0.90 and epochs_to_90pct == -1:
                epochs_to_90pct = epoch

            # Print progress periodically
            elapsed = time.time() - start_time
            if epoch % 20 == 0 or epoch == 1 or te_acc >= 0.90:
                tr_outs = forward_batch(x_train, W1, b1, W2, b2)
                tr_acc = accuracy(tr_outs, y_train)
                print(f"  Epoch {epoch:4d} | train_acc={tr_acc:.3f} test_acc={te_acc:.3f} | "
                      f"steps={total_steps} | {elapsed:.1f}s")

            # Early stop if solved
            if best_test_acc >= 0.99:
                print(f"  *** SOLVED at epoch {epoch}! ***")
                break

    except TimeoutError:
        elapsed = time.time() - start_time
        status = "timeout"
        print(f"  *** TIMEOUT after {elapsed:.1f}s (limit={TIMEOUT_SECONDS}s) ***")
    finally:
        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)

    wall_time = time.time() - start_time

    # Final eval
    te_outs = forward_batch(x_test, W1, b1, W2, b2)
    final_acc = accuracy(te_outs, y_test)
    if final_acc > best_test_acc:
        best_test_acc = final_acc

    result = {
        'n_bits': n_bits,
        'k_sparse': k_sparse,
        'hidden': hidden,
        'n_train': n_train,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'wd': wd,
        'epochs_to_90pct': epochs_to_90pct,
        'total_steps': total_steps,
        'wall_time_s': round(wall_time, 2),
        'final_accuracy': round(final_acc, 4),
        'best_test_acc': round(best_test_acc, 4),
        'status': status,
        'theoretical_nk': n_bits ** k_sparse,
        'c_n_k': math.comb(n_bits, k_sparse),
        'secret_indices': secret,
    }

    print(f"\n  Result: acc={final_acc:.4f}, epochs_to_90%={epochs_to_90pct}, "
          f"steps={total_steps}, time={wall_time:.1f}s, status={status}")

    return result


# ---------------------------------------------------------------------------
# Main: run all configs and print scaling table
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EXPERIMENT D: Scale Stress Test — SGD on Sparse Parity")
    print("  Finding the frontier where standard SGD breaks")
    print(f"  Timeout per config: {TIMEOUT_SECONDS}s")
    print("=" * 60)

    configs = [
        # (n_bits, k_sparse, max_epochs)
        (30, 3, 200),
        (50, 3, 200),
        (20, 5, 200),
        (50, 5, 200),
    ]

    results = []
    for n_bits, k_sparse, max_epochs in configs:
        result = run_one_config(n_bits, k_sparse, max_epochs=max_epochs, seed=42)
        results.append(result)

    # Print scaling table
    print("\n\n")
    print("=" * 90)
    print("  SCALING TABLE: Standard SGD on Sparse Parity")
    print("=" * 90)
    header = f"{'Config':>12} | {'n^k':>10} | {'C(n,k)':>10} | {'Status':>10} | {'Ep->90%':>8} | {'Steps':>8} | {'Time(s)':>8} | {'Best Acc':>8}"
    print(header)
    print("-" * 90)

    for r in results:
        label = f"n={r['n_bits']},k={r['k_sparse']}"
        ep90 = str(r['epochs_to_90pct']) if r['epochs_to_90pct'] != -1 else "---"
        print(f"{label:>12} | {r['theoretical_nk']:>10,} | {r['c_n_k']:>10,} | "
              f"{r['status']:>10} | {ep90:>8} | {r['total_steps']:>8,} | "
              f"{r['wall_time_s']:>8.1f} | {r['best_test_acc']:>8.4f}")

    print("=" * 90)

    # Analysis
    print("\n  ANALYSIS:")
    for r in results:
        label = f"n={r['n_bits']},k={r['k_sparse']}"
        if r['best_test_acc'] >= 0.90:
            print(f"  {label}: SOLVED (or near-solved). SGD works here.")
        elif r['status'] == 'timeout':
            print(f"  {label}: TIMEOUT. n^k={r['theoretical_nk']:,} too expensive for {TIMEOUT_SECONDS}s.")
        else:
            print(f"  {label}: DID NOT CONVERGE in {r['max_epochs']} epochs. "
                  f"May need more epochs or fundamentally different approach.")

    # Save results
    ts = timestamp()
    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_d_scaling'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f'results_{ts}.json'
    save_json({
        'experiment': 'exp_d_scaling',
        'description': 'Scale stress test for standard SGD on sparse parity',
        'timeout_per_config_s': TIMEOUT_SECONDS,
        'configs_tested': len(configs),
        'results': results,
    }, results_path)

    # Also save a latest copy for easy reference
    latest_path = results_dir / 'results_latest.json'
    save_json({
        'experiment': 'exp_d_scaling',
        'description': 'Scale stress test for standard SGD on sparse parity',
        'timeout_per_config_s': TIMEOUT_SECONDS,
        'configs_tested': len(configs),
        'results': results,
    }, latest_path)

    print(f"\n  Results saved to: {results_path}")
    print(f"  Latest copy: {latest_path}")

    return results


if __name__ == '__main__':
    results = main()
