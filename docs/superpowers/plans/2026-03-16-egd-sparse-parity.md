# EGD on Sparse Parity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Egalitarian Gradient Descent (EGD) on sparse parity and sparse sum, measure whether it eliminates the grokking plateau and breaks 10ms on GPU.

**Architecture:** Three deliverables: (1) numpy CPU experiment proving EGD changes convergence dynamics, (2) Modal Labs GPU script measuring wall time for SGD vs EGD on both parity and sum, (3) findings write-up. EGD replaces gradient singular values with 1 (G_egd = U @ V^T from SVD of G), equalizing learning rates across all directions.

**Tech Stack:** numpy (CPU experiments), PyTorch + Modal Labs L4 GPU (GPU timing), existing project harness (Config, generate, fast.py patterns)

**Reference:** arXiv:2510.04930 (Egalitarian Gradient Descent). TODO.md line 127.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/sparse_parity/experiments/exp_egd.py` | Create | CPU experiment: EGD vs SGD convergence on parity (multiple configs, seeds) |
| `bin/gpu_egd.py` | Create | Modal GPU script: EGD vs SGD wall time on parity + sum |
| `results/exp_egd/results.json` | Create | Machine-readable results from CPU experiment |
| `results/exp_egd/gpu_results.json` | Create | Machine-readable results from GPU experiment |
| `findings/exp_egd.md` | Create | Findings write-up following lab template |

Files NOT modified (metric isolation, LAB.md rule #9): `tracker.py`, `cache_tracker.py`, `data.py`, `config.py`, `harness.py`.

---

## Chunk 1: CPU Experiment (EGD vs SGD convergence)

### Task 1: EGD training loop on CPU

**Files:**
- Create: `src/sparse_parity/experiments/exp_egd.py`

This is the core experiment. Pattern follows `exp_sign_sgd.py` closely.

- [ ] **Step 1: Create the experiment file with EGD training function**

The EGD algorithm for weight matrix gradient G:
1. Compute SVD: `U, S, Vt = np.linalg.svd(G, full_matrices=False)`
2. Replace: `G_egd = U @ Vt` (all singular values become 1)
3. For bias (vector): `g_egd = g / (||g|| + eps)` (normalize to unit norm)

```python
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

    # Use best lr from Part 1 (we'll hardcode a few candidates)
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
```

- [ ] **Step 2: Run the CPU experiment**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && PYTHONPATH=src python3 -m sparse_parity.experiments.exp_egd`

Expected: Comparison table showing EGD vs SGD epoch counts. EGD should either solve in fewer epochs or fail (both are findings).

- [ ] **Step 3: Analyze results and pick best EGD config**

Read the comparison table. Note:
- Does EGD solve at all? At what lr?
- How many epochs vs SGD baseline (49 epochs on this machine)?
- Which small config (Part 2) gets closest to 10ms?

- [ ] **Step 4: Commit CPU experiment**

```bash
git add src/sparse_parity/experiments/exp_egd.py results/exp_egd/results.json
git commit -m "Add EGD experiment: grokking elimination + sub-10ms push"
```

---

## Chunk 2: GPU Experiment (Modal Labs timing)

### Task 2: EGD on GPU via Modal

**Files:**
- Create: `bin/gpu_egd.py`

Pattern follows existing `bin/gpu_energy.py`. Adds EGD alongside SGD on both parity and sum tasks.

- [ ] **Step 1: Create the GPU experiment file**

```python
#!/usr/bin/env python3
"""
EGD vs SGD on GPU via Modal Labs.

Runs both methods on parity and sum tasks on an NVIDIA L4 GPU.
Measures wall time, epochs to solve, and accuracy.

Usage:
    modal run bin/gpu_egd.py

Prerequisites:
    pip install modal
    modal token set

Cost: ~$0.005 per run (L4 at $0.84/hr, ~20s container time)
"""

import modal
import time
import json
import os

app = modal.App("sutro-egd")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
)

L4_COST_PER_HOUR = 0.84
L4_COST_PER_SEC = L4_COST_PER_HOUR / 3600


@app.function(gpu="L4", image=image, timeout=300)
def run_gpu_egd(n_bits=20, k_sparse=3, n_train=1000, hidden=200,
                lr_sgd=0.1, lr_egd=0.05, wd=0.01, batch_size=32,
                max_epochs=200, seeds=(42, 43, 44, 45, 46), task="parity"):
    """Run EGD vs SGD on GPU. Returns results dict."""
    import torch
    import numpy as np

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
    print(f"Task: sparse {task}, n={n_bits}, k={k_sparse}, hidden={hidden}")

    def generate(seed):
        rng = np.random.RandomState(seed)
        secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

        def make(n):
            x = rng.choice([-1.0, 1.0], size=(n, n_bits)).astype(np.float32)
            if task == "sum":
                y = np.sum(x[:, secret], axis=1).astype(np.float32)
            else:
                y = np.prod(x[:, secret], axis=1).astype(np.float32)
            return x, y

        x_tr, y_tr = make(n_train)
        x_te, y_te = make(500)
        return (torch.from_numpy(x_tr).to(device),
                torch.from_numpy(y_tr).to(device),
                torch.from_numpy(x_te).to(device),
                torch.from_numpy(y_te).to(device),
                secret)

    def train_one(seed, use_egd, lr):
        x, y, x_te, y_te, secret = generate(seed)

        torch.manual_seed(seed + 1)
        W1 = torch.randn(hidden, n_bits, device=device) * np.sqrt(2.0 / n_bits)
        b1 = torch.zeros(hidden, device=device)
        W2 = torch.randn(1, hidden, device=device) * np.sqrt(2.0 / hidden)
        b2 = torch.zeros(1, device=device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        best_acc = 0.0
        solve_epoch = -1
        epoch_90 = -1

        for epoch in range(1, max_epochs + 1):
            perm = torch.randperm(n_train, device=device)
            for s in range(0, n_train, batch_size):
                idx = perm[s:s+batch_size]
                xb = x[idx]
                yb = y[idx]
                bs = len(idx)

                if task == "sum":
                    # Linear model for sum
                    # (Use 2-layer for fair comparison)
                    pass

                # Forward
                h_pre = xb @ W1.t() + b1
                h = torch.relu(h_pre)
                out = (h @ W2.t() + b2).squeeze(-1)

                if task == "sum":
                    # MSE loss
                    diff = out - yb
                    d_out = (2.0 / bs) * diff
                else:
                    # Hinge loss
                    margin = out * yb
                    mask = (margin < 1.0).float()
                    if mask.sum() == 0:
                        continue
                    d_out = (-yb * mask) / bs

                # Backward
                dW2 = d_out.unsqueeze(1) * h  # (bs, hidden)
                dW2 = dW2.sum(0, keepdim=True)  # (1, hidden)
                db2_g = d_out.sum()
                d_h = d_out.unsqueeze(1) * W2  # (bs, hidden)
                d_h = d_h * (h_pre > 0).float()
                dW1 = d_h.t() @ xb  # (hidden, n_bits)
                db1_g = d_h.sum(0)

                if use_egd:
                    # EGD: SVD on gradient matrices
                    U1, S1, V1t = torch.linalg.svd(dW1, full_matrices=False)
                    dW1 = U1 @ V1t
                    U2, S2, V2t = torch.linalg.svd(dW2, full_matrices=False)
                    dW2 = U2 @ V2t
                    # Bias: normalize
                    n1 = torch.norm(db1_g)
                    if n1 > 1e-8:
                        db1_g = db1_g / n1
                    db2_g = torch.sign(db2_g) if abs(db2_g) > 1e-8 else db2_g

                W1 = W1 - lr * (dW1 + wd * W1)
                b1 = b1 - lr * (db1_g + wd * b1)
                W2 = W2 - lr * (dW2 + wd * W2)
                b2 = b2 - lr * (db2_g + wd * b2)

            # Evaluate
            with torch.no_grad():
                h_te = torch.relu(x_te @ W1.t() + b1)
                out_te = (h_te @ W2.t() + b2).squeeze(-1)
                if task == "sum":
                    pred = torch.round(out_te)
                    acc = float((pred == y_te).float().mean())
                else:
                    pred = torch.sign(out_te)
                    acc = float((pred == y_te).float().mean())

                if acc > best_acc:
                    best_acc = acc
                if acc >= 0.90 and epoch_90 < 0:
                    epoch_90 = epoch
                if acc >= 1.0 and solve_epoch < 0:
                    solve_epoch = epoch

            if best_acc >= 1.0:
                break

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return {
            'method': 'egd' if use_egd else 'sgd',
            'seed': seed,
            'best_acc': round(best_acc, 4),
            'solve_epoch': solve_epoch,
            'epoch_90': epoch_90,
            'total_epochs': epoch,
            'time_s': round(elapsed, 6),
        }

    # --- Run all configs ---
    all_results = []

    for seed in seeds:
        # SGD
        r = train_one(seed, use_egd=False, lr=lr_sgd)
        all_results.append(r)
        print(f"  SGD  seed={seed}: acc={r['best_acc']:.2f} "
              f"ep90={r['epoch_90']} solve={r['solve_epoch']} "
              f"{r['time_s']*1000:.1f}ms")

        # EGD
        r = train_one(seed, use_egd=True, lr=lr_egd)
        all_results.append(r)
        print(f"  EGD  seed={seed}: acc={r['best_acc']:.2f} "
              f"ep90={r['epoch_90']} solve={r['solve_epoch']} "
              f"{r['time_s']*1000:.1f}ms")

    return {
        'gpu': torch.cuda.get_device_name(0),
        'task': task,
        'config': {
            'n_bits': n_bits, 'k_sparse': k_sparse,
            'n_train': n_train, 'hidden': hidden,
            'lr_sgd': lr_sgd, 'lr_egd': lr_egd,
            'wd': wd, 'batch_size': batch_size,
            'max_epochs': max_epochs,
        },
        'results': all_results,
    }


@app.local_entrypoint()
def main():
    import json as json_mod
    from pathlib import Path

    results_dir = Path("results/exp_egd")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_gpu = {}

    # --- Parity (standard config) ---
    print("=" * 70)
    print("  EGD vs SGD on GPU: Sparse Parity (n=20, k=3)")
    print("=" * 70)

    wall_start = time.time()
    parity_result = run_gpu_egd.remote(
        task="parity", hidden=200, lr_sgd=0.1, lr_egd=0.05)
    wall_elapsed = time.time() - wall_start
    all_gpu['parity_h200'] = parity_result
    print(f"\n  Wall time: {wall_elapsed:.1f}s")

    # --- Parity (small config for speed) ---
    print("\n" + "=" * 70)
    print("  EGD vs SGD on GPU: Sparse Parity SPEED (hidden=50)")
    print("=" * 70)

    wall_start = time.time()
    parity_small = run_gpu_egd.remote(
        task="parity", hidden=50, n_train=500, lr_sgd=0.1, lr_egd=0.05)
    wall_elapsed = time.time() - wall_start
    all_gpu['parity_h50'] = parity_small
    print(f"\n  Wall time: {wall_elapsed:.1f}s")

    # --- Sum ---
    print("\n" + "=" * 70)
    print("  EGD vs SGD on GPU: Sparse Sum (n=20, k=3)")
    print("=" * 70)

    wall_start = time.time()
    sum_result = run_gpu_egd.remote(
        task="sum", hidden=200, lr_sgd=0.1, lr_egd=0.05)
    wall_elapsed = time.time() - wall_start
    all_gpu['sum_h200'] = sum_result
    print(f"\n  Wall time: {wall_elapsed:.1f}s")

    # --- Print summary ---
    print("\n\n" + "=" * 90)
    print("  GPU SUMMARY")
    print("=" * 90)

    for config_name, data in all_gpu.items():
        print(f"\n  --- {config_name} (GPU: {data['gpu']}) ---")
        sgd_runs = [r for r in data['results'] if r['method'] == 'sgd']
        egd_runs = [r for r in data['results'] if r['method'] == 'egd']

        for label, runs in [('SGD', sgd_runs), ('EGD', egd_runs)]:
            times = [r['time_s'] * 1000 for r in runs]
            accs = [r['best_acc'] for r in runs]
            solves = [r['solve_epoch'] for r in runs if r['solve_epoch'] > 0]
            avg_solve = sum(solves) / len(solves) if solves else -1
            n_ok = sum(1 for a in accs if a >= 0.95)
            print(f"    {label}: avg={sum(times)/len(times):.1f}ms "
                  f"min={min(times):.1f}ms max={max(times):.1f}ms "
                  f"solve_ep={avg_solve:.0f} ok={n_ok}/{len(runs)}")

    # --- Save ---
    gpu_path = results_dir / 'gpu_results.json'
    with open(gpu_path, 'w') as f:
        json_mod.dump(all_gpu, f, indent=2, default=str)
    print(f"\n  Saved: {gpu_path}")
```

- [ ] **Step 2: Run the GPU experiment on Modal**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && modal run bin/gpu_egd.py`

Expected: Summary table with SGD vs EGD wall times on GPU for parity and sum. Note the lr_egd value may need tuning based on CPU results from Task 1.

- [ ] **Step 3: If EGD lr needs tuning, update lr_egd in gpu_egd.py and re-run**

Check the CPU results from Task 1. If the best EGD lr is not 0.05, update the `lr_egd` default in `gpu_egd.py` and re-run.

- [ ] **Step 4: Commit GPU experiment**

```bash
git add bin/gpu_egd.py results/exp_egd/gpu_results.json
git commit -m "Add EGD GPU experiment via Modal Labs (parity + sum)"
```

---

## Chunk 3: Findings Write-up

### Task 3: Write findings document

**Files:**
- Create: `findings/exp_egd.md`
- Modify: `DISCOVERIES.md` (add EGD bullet)

- [ ] **Step 1: Write findings/exp_egd.md**

Use the lab template from `findings/_experiment_template.md`. Fill in based on actual results from Tasks 1 and 2. The structure:

```markdown
# Experiment exp_egd: Egalitarian Gradient Descent

**Date**: 2026-03-16
**Status**: {SUCCESS | PARTIAL | FAILED}
**Answers**: TODO.md "SGD Under 10ms" / EGD hypothesis

## Hypothesis

If we replace gradient singular values with 1 (EGD), then the grokking
plateau shortens because all gradient directions evolve at equal speed,
and we can push toward sub-10ms wall time.

## Config

| Parameter | SGD Baseline | EGD |
|-----------|-------------|-----|
| n_bits | 20 | 20 |
| k_sparse | 3 | 3 |
| hidden | 200 | 200 |
| lr | 0.1 | {best from experiment} |
| wd | 0.01 | 0.01 |
| batch_size | 32 | 32 |
| max_epochs | 200 | 200 |
| n_train | 1000 | 1000 |
| seeds | 42-46 | 42-46 |

## Results

### Part 1: Grokking Elimination (CPU)
{Table from exp_egd.py output}

### Part 2: Sub-10ms Push (CPU)
{Table from exp_egd.py output, small configs}

### Part 3: GPU Timing (Modal L4)
{Table from gpu_egd.py output}

### Part 4: Sparse Sum Comparison (GPU)
{Table from gpu_egd.py sum output}

## Key Table
{The single most important comparison}

## Analysis

### What worked
- {Fill based on results}

### What didn't work
- {Fill based on results}

### Surprise
- {Fill based on results}

## Open Questions (for next experiment)
- {Based on what we learn}

## Files

- CPU experiment: `src/sparse_parity/experiments/exp_egd.py`
- GPU experiment: `bin/gpu_egd.py`
- CPU results: `results/exp_egd/results.json`
- GPU results: `results/exp_egd/gpu_results.json`
```

- [ ] **Step 2: Update DISCOVERIES.md**

Add EGD bullet to the appropriate section based on results. If it worked:
```
- **EGD solves in N epochs vs M for SGD** (Xms vs Yms on GPU). SVD-normalized gradients eliminate/reduce the grokking plateau. [exp_egd]
```
If it failed:
```
- **EGD does not help sparse parity**: SVD normalization {reason}. [exp_egd]
```

- [ ] **Step 3: Commit findings**

```bash
git add findings/exp_egd.md DISCOVERIES.md
git commit -m "EGD findings: {one-line summary of result}"
```
