"""Experiment 4: GrokFast — Low-Pass Gradient Filter for Accelerated Grokking.

Implements the GrokFast technique from Lee et al. 2024:
  g_slow = alpha * g_slow + (1-alpha) * grad
  grad_modified = grad + lambda * g_slow

The key insight: grokking is driven by slowly-evolving gradient components.
By maintaining an EMA of gradients and amplifying that slow component,
we can accelerate the phase transition from memorization to generalization.

This requires separating gradient computation from weight updates so we can
filter gradients in between.
"""

import time
import math
import random
import json
import copy
from pathlib import Path

# Add parent paths for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params, forward, forward_batch
from sparse_parity.metrics import hinge_loss, accuracy, save_json, timestamp


# ---------------------------------------------------------------------------
# GrokFast-specific backward: compute gradients WITHOUT updating weights
# ---------------------------------------------------------------------------

def compute_gradients(x, y, out, h_pre, h, W1, b1, W2, b2):
    """
    Compute gradients for all parameters via backprop.
    Does NOT modify weights. Returns (dW1, db1, dW2, db2) or None if margin >= 1.
    """
    hidden = len(W1)
    n_bits = len(x)

    margin = out * y
    if margin >= 1.0:
        return None  # hinge loss is zero, no gradient

    dout = -y

    # Layer 2 gradients
    dW2 = [[dout * h[j] for j in range(hidden)]]
    db2 = [dout]

    # dh = W2^T * dout
    dh = [W2[0][j] * dout for j in range(hidden)]

    # ReLU backward
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    # Layer 1 gradients
    dW1 = [[dh_pre[j] * x[i] for i in range(n_bits)] for j in range(hidden)]
    db1 = [dh_pre[j] for j in range(hidden)]

    return dW1, db1, dW2, db2


def apply_grokfast_and_update(W1, b1, W2, b2,
                               dW1, db1, dW2, db2,
                               ema_W1, ema_b1, ema_W2, ema_b2,
                               lr, wd, alpha, lam):
    """
    Apply GrokFast filter to gradients, then SGD update with weight decay.

    GrokFast:
      ema = alpha * ema + (1-alpha) * grad
      grad_filtered = grad + lam * ema

    SGD update:
      param -= lr * (grad_filtered + wd * param)
    """
    hidden = len(W1)
    n_bits = len(W1[0])
    one_minus_alpha = 1.0 - alpha

    # --- W1 ---
    for j in range(hidden):
        for i in range(n_bits):
            g = dW1[j][i]
            ema_W1[j][i] = alpha * ema_W1[j][i] + one_minus_alpha * g
            g_filtered = g + lam * ema_W1[j][i]
            W1[j][i] -= lr * (g_filtered + wd * W1[j][i])

    # --- b1 ---
    for j in range(hidden):
        g = db1[j]
        ema_b1[j] = alpha * ema_b1[j] + one_minus_alpha * g
        g_filtered = g + lam * ema_b1[j]
        b1[j] -= lr * (g_filtered + wd * b1[j])

    # --- W2 ---
    for j in range(hidden):
        g = dW2[0][j]
        ema_W2[0][j] = alpha * ema_W2[0][j] + one_minus_alpha * g
        g_filtered = g + lam * ema_W2[0][j]
        W2[0][j] -= lr * (g_filtered + wd * W2[0][j])

    # --- b2 ---
    g = db2[0]
    ema_b2[0] = alpha * ema_b2[0] + one_minus_alpha * g
    g_filtered = g + lam * ema_b2[0]
    b2[0] -= lr * (g_filtered + wd * b2[0])


def apply_sgd_update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr, wd):
    """Standard SGD update with weight decay (no GrokFast), for baseline comparison."""
    hidden = len(W1)
    n_bits = len(W1[0])

    for j in range(hidden):
        for i in range(n_bits):
            W1[j][i] -= lr * (dW1[j][i] + wd * W1[j][i])

    for j in range(hidden):
        b1[j] -= lr * (db1[j] + wd * b1[j])

    for j in range(hidden):
        W2[0][j] -= lr * (dW2[0][j] + wd * W2[0][j])

    b2[0] -= lr * (db2[0] + wd * b2[0])


# ---------------------------------------------------------------------------
# Weight movement tracking: ||w_t - w_0||_1
# ---------------------------------------------------------------------------

def compute_weight_movement(W1, b1, W2, b2, W1_0, b1_0, W2_0, b2_0):
    """Compute L1 norm of weight change from initialization."""
    hidden = len(W1)
    n_bits = len(W1[0])
    total = 0.0

    for j in range(hidden):
        for i in range(n_bits):
            total += abs(W1[j][i] - W1_0[j][i])

    for j in range(hidden):
        total += abs(b1[j] - b1_0[j])

    for j in range(hidden):
        total += abs(W2[0][j] - W2_0[0][j])

    total += abs(b2[0] - b2_0[0])
    return total


def deep_copy_params(W1, b1, W2, b2):
    """Deep copy all parameters."""
    W1_c = [row[:] for row in W1]
    b1_c = b1[:]
    W2_c = [row[:] for row in W2]
    b2_c = b2[:]
    return W1_c, b1_c, W2_c, b2_c


# ---------------------------------------------------------------------------
# Training loop with GrokFast
# ---------------------------------------------------------------------------

def train_grokfast(x_train, y_train, x_test, y_test,
                   W1, b1, W2, b2, config,
                   alpha=0.98, lam=2.0, print_every=10):
    """
    Train with GrokFast gradient filtering.

    GrokFast maintains EMA of gradients and amplifies slow components:
      ema = alpha * ema + (1-alpha) * grad
      grad_modified = grad + lam * ema
    """
    hidden = len(W1)
    n_bits = len(W1[0])

    # Save initial weights for movement tracking
    W1_0, b1_0, W2_0, b2_0 = deep_copy_params(W1, b1, W2, b2)

    # Initialize EMA buffers (all zeros)
    ema_W1 = [[0.0] * n_bits for _ in range(hidden)]
    ema_b1 = [0.0] * hidden
    ema_W2 = [[0.0] * hidden]
    ema_b2 = [0.0]

    # History tracking
    history = {
        'epochs': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'weight_movement': [],
    }

    best_test_acc = 0.0
    step = 0
    start = time.time()
    n_train = len(x_train)

    print(f"\n{'='*78}")
    print(f"  GrokFast Training: n={config.n_bits}, k={config.k_sparse}, "
          f"hidden={config.hidden}, alpha={alpha}, lambda={lam}")
    print(f"  LR={config.lr}, WD={config.wd}, epochs={config.max_epochs}, "
          f"n_train={n_train}")
    print(f"{'='*78}")
    print(f"  {'Epoch':>6}  {'Train%':>7}  {'Test%':>7}  {'Loss':>8}  "
          f"{'|w-w0|':>10}  {'Time':>6}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*6}")

    for epoch in range(1, config.max_epochs + 1):
        # Single-sample cyclic SGD with GrokFast
        for i in range(n_train):
            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
            grads = compute_gradients(x_train[i], y_train[i], out, h_pre, h,
                                       W1, b1, W2, b2)
            if grads is not None:
                dW1, db1_g, dW2, db2_g = grads
                apply_grokfast_and_update(
                    W1, b1, W2, b2,
                    dW1, db1_g, dW2, db2_g,
                    ema_W1, ema_b1, ema_W2, ema_b2,
                    config.lr, config.wd, alpha, lam
                )
            step += 1

        # Evaluate
        tr_outs = forward_batch(x_train, W1, b1, W2, b2)
        te_outs = forward_batch(x_test, W1, b1, W2, b2)
        tr_loss = hinge_loss(tr_outs, y_train)
        te_loss = hinge_loss(te_outs, y_test)
        tr_acc = accuracy(tr_outs, y_train)
        te_acc = accuracy(te_outs, y_test)
        w_move = compute_weight_movement(W1, b1, W2, b2, W1_0, b1_0, W2_0, b2_0)

        history['epochs'].append(epoch)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['weight_movement'].append(w_move)

        if te_acc > best_test_acc:
            best_test_acc = te_acc

        if epoch % print_every == 0 or epoch == 1 or te_acc > 0.6:
            elapsed = time.time() - start
            print(f"  {epoch:>6}  {tr_acc:>6.1%}  {te_acc:>6.1%}  "
                  f"{tr_loss:>8.4f}  {w_move:>10.2f}  {elapsed:>5.1f}s")

        if best_test_acc >= 1.0:
            elapsed = time.time() - start
            print(f"\n  *** PERFECT ACCURACY at epoch {epoch}! ***")
            break

    elapsed = time.time() - start
    print(f"\n  Final: train_acc={tr_acc:.1%}, test_acc={te_acc:.1%}, "
          f"best_test={best_test_acc:.1%}")
    print(f"  Total time: {elapsed:.2f}s, steps: {step}")
    print(f"{'='*78}")

    return {
        'method': 'grokfast',
        'alpha': alpha,
        'lam': lam,
        'best_test_acc': best_test_acc,
        'final_train_acc': tr_acc,
        'final_test_acc': te_acc,
        'total_steps': step,
        'elapsed_s': elapsed,
        'history': history,
        'config': {
            'n_bits': config.n_bits,
            'k_sparse': config.k_sparse,
            'hidden': config.hidden,
            'n_train': config.n_train,
            'n_test': config.n_test,
            'lr': config.lr,
            'wd': config.wd,
            'max_epochs': config.max_epochs,
            'seed': config.seed,
        }
    }


def train_baseline(x_train, y_train, x_test, y_test,
                   W1, b1, W2, b2, config, print_every=10):
    """
    Baseline training (same setup, no GrokFast) for comparison.
    Uses separated gradient computation (same code path) but no EMA filtering.
    """
    hidden = len(W1)
    n_bits = len(W1[0])

    W1_0, b1_0, W2_0, b2_0 = deep_copy_params(W1, b1, W2, b2)

    history = {
        'epochs': [],
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'weight_movement': [],
    }

    best_test_acc = 0.0
    step = 0
    start = time.time()
    n_train = len(x_train)

    print(f"\n{'='*78}")
    print(f"  Baseline Training (no GrokFast): n={config.n_bits}, k={config.k_sparse}, "
          f"hidden={config.hidden}")
    print(f"  LR={config.lr}, WD={config.wd}, epochs={config.max_epochs}, "
          f"n_train={n_train}")
    print(f"{'='*78}")
    print(f"  {'Epoch':>6}  {'Train%':>7}  {'Test%':>7}  {'Loss':>8}  "
          f"{'|w-w0|':>10}  {'Time':>6}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*10}  {'-'*6}")

    for epoch in range(1, config.max_epochs + 1):
        for i in range(n_train):
            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2)
            grads = compute_gradients(x_train[i], y_train[i], out, h_pre, h,
                                       W1, b1, W2, b2)
            if grads is not None:
                dW1, db1_g, dW2, db2_g = grads
                apply_sgd_update(W1, b1, W2, b2,
                                 dW1, db1_g, dW2, db2_g,
                                 config.lr, config.wd)
            step += 1

        # Evaluate
        tr_outs = forward_batch(x_train, W1, b1, W2, b2)
        te_outs = forward_batch(x_test, W1, b1, W2, b2)
        tr_loss = hinge_loss(tr_outs, y_train)
        te_loss = hinge_loss(te_outs, y_test)
        tr_acc = accuracy(tr_outs, y_train)
        te_acc = accuracy(te_outs, y_test)
        w_move = compute_weight_movement(W1, b1, W2, b2, W1_0, b1_0, W2_0, b2_0)

        history['epochs'].append(epoch)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['weight_movement'].append(w_move)

        if te_acc > best_test_acc:
            best_test_acc = te_acc

        if epoch % print_every == 0 or epoch == 1 or te_acc > 0.6:
            elapsed = time.time() - start
            print(f"  {epoch:>6}  {tr_acc:>6.1%}  {te_acc:>6.1%}  "
                  f"{tr_loss:>8.4f}  {w_move:>10.2f}  {elapsed:>5.1f}s")

        if best_test_acc >= 1.0:
            elapsed = time.time() - start
            print(f"\n  *** PERFECT ACCURACY at epoch {epoch}! ***")
            break

    elapsed = time.time() - start
    print(f"\n  Final: train_acc={tr_acc:.1%}, test_acc={te_acc:.1%}, "
          f"best_test={best_test_acc:.1%}")
    print(f"  Total time: {elapsed:.2f}s, steps: {step}")
    print(f"{'='*78}")

    return {
        'method': 'baseline_sgd',
        'best_test_acc': best_test_acc,
        'final_train_acc': tr_acc,
        'final_test_acc': te_acc,
        'total_steps': step,
        'elapsed_s': elapsed,
        'history': history,
        'config': {
            'n_bits': config.n_bits,
            'k_sparse': config.k_sparse,
            'hidden': config.hidden,
            'n_train': config.n_train,
            'n_test': config.n_test,
            'lr': config.lr,
            'wd': config.wd,
            'max_epochs': config.max_epochs,
            'seed': config.seed,
        }
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    # Config matching the research plan
    config = Config(
        n_bits=20,
        k_sparse=3,
        hidden=1000,
        n_train=500,
        n_test=200,
        lr=0.1,
        wd=0.01,
        max_epochs=200,
        seed=42,
    )

    # GrokFast hyperparameters
    alpha = 0.98
    lam = 2.0

    print("=" * 78)
    print("  EXPERIMENT 4: GrokFast (Low-Pass Gradient Filter)")
    print("  Hypothesis: Amplifying slow gradient components accelerates grokking")
    print("  Reference: Lee et al. 2024 — GrokFast: Accelerated Grokking")
    print("=" * 78)

    # Generate data
    x_train, y_train, x_test, y_test, secret = generate(config)
    print(f"\n  Secret parity indices: {secret}")
    print(f"  Total params: {config.total_params:,}")
    print(f"  Train/Test: {config.n_train}/{config.n_test}")

    # ---- Run 1: GrokFast ----
    W1_gf, b1_gf, W2_gf, b2_gf = init_params(config)
    result_gf = train_grokfast(
        x_train, y_train, x_test, y_test,
        W1_gf, b1_gf, W2_gf, b2_gf,
        config, alpha=alpha, lam=lam, print_every=10
    )

    # ---- Run 2: Baseline (same config, no GrokFast) ----
    W1_bl, b1_bl, W2_bl, b2_bl = init_params(config)
    result_bl = train_baseline(
        x_train, y_train, x_test, y_test,
        W1_bl, b1_bl, W2_bl, b2_bl,
        config, print_every=10
    )

    # ---- Save results ----
    ts = timestamp()
    results_dir = Path(__file__).parent.parent.parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'experiment': 'exp4_grokfast',
        'timestamp': ts,
        'secret_indices': secret,
        'grokfast': result_gf,
        'baseline': result_bl,
        'summary': {
            'grokfast_best_test': result_gf['best_test_acc'],
            'baseline_best_test': result_bl['best_test_acc'],
            'grokfast_time_s': result_gf['elapsed_s'],
            'baseline_time_s': result_bl['elapsed_s'],
            'alpha': alpha,
            'lambda': lam,
        }
    }

    json_path = results_dir / f'exp4_grokfast_{ts}.json'
    save_json(results, str(json_path))
    print(f"\n  Results saved: {json_path}")

    # ---- Summary ----
    print(f"\n{'='*78}")
    print(f"  EXPERIMENT 4 SUMMARY")
    print(f"{'='*78}")
    print(f"  GrokFast (alpha={alpha}, lambda={lam}):")
    print(f"    Best test accuracy: {result_gf['best_test_acc']:.1%}")
    print(f"    Final train/test:   {result_gf['final_train_acc']:.1%} / "
          f"{result_gf['final_test_acc']:.1%}")
    print(f"    Time: {result_gf['elapsed_s']:.1f}s")
    print(f"")
    print(f"  Baseline SGD (no GrokFast):")
    print(f"    Best test accuracy: {result_bl['best_test_acc']:.1%}")
    print(f"    Final train/test:   {result_bl['final_train_acc']:.1%} / "
          f"{result_bl['final_test_acc']:.1%}")
    print(f"    Time: {result_bl['elapsed_s']:.1f}s")
    print(f"")

    gf_better = result_gf['best_test_acc'] > result_bl['best_test_acc']
    if result_gf['best_test_acc'] >= 0.9:
        print(f"  *** BREAKTHROUGH: GrokFast achieved >90% test accuracy! ***")
    elif result_gf['best_test_acc'] >= 0.6:
        print(f"  ** SIGNIFICANT: GrokFast broke past 60% test accuracy! **")
    elif gf_better:
        print(f"  GrokFast outperformed baseline "
              f"({result_gf['best_test_acc']:.1%} vs {result_bl['best_test_acc']:.1%})")
    else:
        print(f"  GrokFast did not outperform baseline on this configuration.")

    print(f"{'='*78}\n")

    return results


if __name__ == '__main__':
    main()
