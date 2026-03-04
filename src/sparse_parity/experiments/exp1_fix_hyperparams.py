"""
Experiment 1: Fix hyperparameters to match Barak et al. 2022.

Hypothesis: Matching Barak et al.'s hyperparams (LR=0.1, batch_size=32,
more epochs) will trigger the phase transition on 20-bit sparse parity (k=3).

Key changes from baseline:
  - Mini-batch SGD (batch_size=32) with gradient accumulation
  - LR = 0.1 (was 0.5)
  - n_train = 500, n_test = 200
  - 200-500 epochs (was 50)
  - Track hidden progress: ||w_t - w_0||_1
"""

import sys
import os
import time
import json
import random
import copy
import math
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward, forward_batch
from sparse_parity.metrics import hinge_loss, accuracy, save_json, timestamp


# ---------------------------------------------------------------------------
# Mini-batch gradient computation (no in-place weight updates)
# ---------------------------------------------------------------------------

def compute_gradients(x, y, W1, b1, W2, b2):
    """
    Compute gradients for a single sample via backprop.
    Returns (dW1, db1, dW2, db2) without modifying weights.
    Returns None if the hinge margin >= 1 (no gradient).
    """
    hidden = len(W1)
    n_bits = len(x)

    # Forward
    h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j]
             for j in range(hidden)]
    h = [max(0.0, v) for v in h_pre]
    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    # Hinge loss: loss = max(0, 1 - out*y)
    margin = out * y
    if margin >= 1.0:
        return None  # no gradient contribution

    dout = -y

    # Layer 2 gradients
    dW2 = [[dout * h[j] for j in range(hidden)]]
    db2 = [dout]

    # Backprop through layer 2
    dh = [W2[0][j] * dout for j in range(hidden)]

    # ReLU backward
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    # Layer 1 gradients
    dW1 = [[dh_pre[j] * x[i] for i in range(n_bits)] for j in range(hidden)]
    db1 = [dh_pre[j] for j in range(hidden)]

    return dW1, db1, dW2, db2


def minibatch_sgd_step(batch_x, batch_y, W1, b1, W2, b2, lr, wd):
    """
    Perform one mini-batch SGD step:
    1. Compute gradients for each sample in the batch
    2. Average gradients over the batch
    3. Apply update with weight decay
    """
    hidden = len(W1)
    n_bits = len(W1[0])
    batch_size = len(batch_x)

    # Accumulators (initialize to zero)
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
        return  # all samples in batch already classified with margin >= 1

    # Average over contributing samples
    inv_n = 1.0 / n_contributing

    # Apply update: W -= lr * (avg_grad + wd * W)
    for j in range(hidden):
        for i in range(n_bits):
            W1[j][i] -= lr * (acc_dW1[j][i] * inv_n + wd * W1[j][i])
        b1[j] -= lr * (acc_db1[j] * inv_n + wd * b1[j])
        W2[0][j] -= lr * (acc_dW2[0][j] * inv_n + wd * W2[0][j])
    b2[0] -= lr * (acc_db2[0] * inv_n + wd * b2[0])


# ---------------------------------------------------------------------------
# Weight norm tracking (hidden progress measure)
# ---------------------------------------------------------------------------

def weight_l1_movement(W1, b1, W2, b2, W1_0, b1_0, W2_0, b2_0):
    """Compute ||theta_t - theta_0||_1 for all parameters."""
    hidden = len(W1)
    n_bits = len(W1[0])
    total = 0.0
    for j in range(hidden):
        for i in range(n_bits):
            total += abs(W1[j][i] - W1_0[j][i])
        total += abs(b1[j] - b1_0[j])
        total += abs(W2[0][j] - W2_0[0][j])
    total += abs(b2[0] - b2_0[0])
    return total


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(max_epochs=200, hidden=1000):
    """Run Experiment 1 with Barak et al. hyperparameters."""

    config = Config(
        n_bits=20,
        k_sparse=3,
        n_train=500,
        n_test=200,
        hidden=hidden,
        lr=0.1,
        wd=0.01,
        max_epochs=max_epochs,
        seed=42,
    )
    batch_size = 32

    print("=" * 70)
    print("  EXPERIMENT 1: Fix Hyperparameters (Barak et al. 2022)")
    print("=" * 70)
    print(f"  n_bits={config.n_bits}, k_sparse={config.k_sparse}, hidden={config.hidden}")
    print(f"  n_train={config.n_train}, n_test={config.n_test}")
    print(f"  lr={config.lr}, wd={config.wd}, batch_size={batch_size}")
    print(f"  max_epochs={max_epochs}")
    print(f"  seed={config.seed}")
    print("=" * 70)

    # Generate data
    x_train, y_train, x_test, y_test, secret = generate(config)
    print(f"  Secret indices: {secret}")
    print(f"  Train label balance: {sum(1 for y in y_train if y > 0)}/{len(y_train)} positive")
    print()

    # Initialize model
    W1, b1, W2, b2 = init_params(config)

    # Save initial weights for hidden progress tracking
    W1_0 = [row[:] for row in W1]
    b1_0 = b1[:]
    W2_0 = [row[:] for row in W2]
    b2_0 = b2[:]

    # Training loop
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    weight_movements = []
    best_test_acc = 0.0
    total_steps = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Shuffle training data each epoch
        indices = list(range(config.n_train))
        rng = random.Random(config.seed + epoch)
        rng.shuffle(indices)

        # Mini-batch training
        for batch_start in range(0, config.n_train, batch_size):
            batch_end = min(batch_start + batch_size, config.n_train)
            batch_idx = indices[batch_start:batch_end]
            batch_x = [x_train[i] for i in batch_idx]
            batch_y = [y_train[i] for i in batch_idx]

            minibatch_sgd_step(batch_x, batch_y, W1, b1, W2, b2,
                               config.lr, config.wd)
            total_steps += 1

        # Evaluate
        tr_outs = forward_batch(x_train, W1, b1, W2, b2)
        te_outs = forward_batch(x_test, W1, b1, W2, b2)

        tr_loss = hinge_loss(tr_outs, y_train)
        te_loss = hinge_loss(te_outs, y_test)
        tr_acc = accuracy(tr_outs, y_train)
        te_acc = accuracy(te_outs, y_test)
        wt_move = weight_l1_movement(W1, b1, W2, b2, W1_0, b1_0, W2_0, b2_0)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        weight_movements.append(wt_move)

        if te_acc > best_test_acc:
            best_test_acc = te_acc

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or te_acc > 0.9:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:4d} | train_acc={tr_acc:.3f} test_acc={te_acc:.3f} | "
                  f"loss={tr_loss:.4f} | ||w-w0||_1={wt_move:.2f} | "
                  f"steps={total_steps} | {elapsed:.1f}s")

        # Early stopping if solved
        if best_test_acc >= 0.99:
            print(f"\n  *** SOLVED at epoch {epoch}! test_acc={te_acc:.3f} ***")
            break

    elapsed_total = time.time() - start_time

    print()
    print("=" * 70)
    print(f"  RESULT: best_test_acc = {best_test_acc:.4f}")
    print(f"  Total steps: {total_steps}, Time: {elapsed_total:.1f}s")
    print(f"  Final weight movement: {weight_movements[-1]:.2f}")
    print("=" * 70)

    # Save results
    results = {
        'experiment': 'exp1_fix_hyperparams',
        'hypothesis': 'Matching Barak et al. hyperparams triggers phase transition',
        'config': {
            'n_bits': config.n_bits,
            'k_sparse': config.k_sparse,
            'n_train': config.n_train,
            'n_test': config.n_test,
            'hidden': config.hidden,
            'lr': config.lr,
            'wd': config.wd,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'seed': config.seed,
        },
        'secret_indices': secret,
        'best_test_acc': best_test_acc,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1],
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'total_steps': total_steps,
        'elapsed_s': elapsed_total,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'weight_movements': weight_movements,
    }

    ts = timestamp()
    results_dir = Path(__file__).resolve().parents[3] / 'results' / f'exp1_{ts}'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'results.json'
    save_json(results, results_path)
    print(f"\n  Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    # Phase 1: Try 200 epochs first
    results = run_experiment(max_epochs=200, hidden=1000)

    # Phase 2: If still at chance, try 500 epochs
    if results['best_test_acc'] < 0.6:
        print("\n\n" + "=" * 70)
        print("  Accuracy still near chance. Trying 500 epochs...")
        print("=" * 70 + "\n")
        results = run_experiment(max_epochs=500, hidden=1000)

    # Phase 3: If still failing, try hidden=500 with 1000 epochs
    if results['best_test_acc'] < 0.6:
        print("\n\n" + "=" * 70)
        print("  Still failing. Trying hidden=500, 1000 epochs...")
        print("=" * 70 + "\n")
        results = run_experiment(max_epochs=1000, hidden=500)
