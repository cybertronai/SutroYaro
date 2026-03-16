#!/usr/bin/env python3
"""
EGD vs SGD on GPU via Modal Labs.

Runs both methods on parity and sum tasks on an NVIDIA L4 GPU.
Measures wall time, epochs to solve, and accuracy.

CPU finding: EGD halves epoch count (14 vs 33 to 90%) but SVD overhead
makes CPU wall time worse. GPU SVD should be fast enough to flip this.

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
                lr_sgd=0.1, lr_egd=0.1, wd=0.01, batch_size=32,
                max_epochs=200, seeds=(42, 43, 44, 45, 46), task="parity"):
    """Run EGD vs SGD on GPU. Returns results dict."""
    import torch
    import numpy as np

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
    print(f"Task: sparse {task}, n={n_bits}, k={k_sparse}, hidden={hidden}")
    print(f"LR: sgd={lr_sgd}, egd={lr_egd}")

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

        # Warmup CUDA
        _ = torch.randn(10, 10, device=device) @ torch.randn(10, 10, device=device)
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

                # Forward
                h_pre = xb @ W1.t() + b1
                h = torch.relu(h_pre)
                out = (h @ W2.t() + b2).squeeze(-1)

                if task == "sum":
                    # MSE loss gradient
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
                dW2 = (d_out.unsqueeze(1) * h).sum(0, keepdim=True)  # (1, hidden)
                db2_g = d_out.sum()
                d_h = d_out.unsqueeze(1) * W2  # (bs, hidden)
                d_h = d_h * (h_pre > 0).float()
                dW1 = d_h.t() @ xb  # (hidden, n_bits)
                db1_g = d_h.sum(0)

                if use_egd:
                    # EGD: SVD on gradient matrices, replace singular values with 1
                    U1, S1, V1t = torch.linalg.svd(dW1, full_matrices=False)
                    dW1 = U1 @ V1t
                    U2, S2, V2t = torch.linalg.svd(dW2, full_matrices=False)
                    dW2 = U2 @ V2t
                    # Bias: normalize to unit norm
                    n1 = torch.norm(db1_g)
                    if n1 > 1e-8:
                        db1_g = db1_g / n1
                    db2_g = torch.sign(db2_g)

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
            'time_ms': round(elapsed * 1000, 2),
        }

    # --- Run all configs ---
    all_results = []

    for seed in seeds:
        # SGD
        r = train_one(seed, use_egd=False, lr=lr_sgd)
        all_results.append(r)
        print(f"  SGD  seed={seed}: acc={r['best_acc']:.2f} "
              f"ep90={r['epoch_90']} solve={r['solve_epoch']} "
              f"{r['time_ms']:.1f}ms ({r['total_epochs']} ep)")

        # EGD
        r = train_one(seed, use_egd=True, lr=lr_egd)
        all_results.append(r)
        print(f"  EGD  seed={seed}: acc={r['best_acc']:.2f} "
              f"ep90={r['epoch_90']} solve={r['solve_epoch']} "
              f"{r['time_ms']:.1f}ms ({r['total_epochs']} ep)")

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


def print_summary(label, data):
    """Print summary for one config."""
    import numpy as np

    print(f"\n  --- {label} (GPU: {data['gpu']}) ---")
    print(f"  Config: {data['config']}")

    sgd_runs = [r for r in data['results'] if r['method'] == 'sgd']
    egd_runs = [r for r in data['results'] if r['method'] == 'egd']

    header = f"  {'Method':<6} | {'Avg ms':>8} | {'Min ms':>8} | {'Max ms':>8} | {'Ep90':>5} | {'Solve':>5} | {'Ok':>5}"
    print(header)
    print("  " + "-" * 70)

    for method_label, runs in [('SGD', sgd_runs), ('EGD', egd_runs)]:
        times = [r['time_ms'] for r in runs]
        accs = [r['best_acc'] for r in runs]
        ep90s = [r['epoch_90'] for r in runs if r['epoch_90'] > 0]
        solves = [r['solve_epoch'] for r in runs if r['solve_epoch'] > 0]
        avg_ep90 = np.mean(ep90s) if ep90s else float('nan')
        avg_solve = np.mean(solves) if solves else float('nan')
        n_ok = sum(1 for a in accs if a >= 0.95)
        ep90_str = f"{avg_ep90:.0f}" if not np.isnan(avg_ep90) else "---"
        solve_str = f"{avg_solve:.0f}" if not np.isnan(avg_solve) else "---"
        print(f"  {method_label:<6} | {np.mean(times):>7.1f} | {min(times):>7.1f} | "
              f"{max(times):>7.1f} | {ep90_str:>5} | {solve_str:>5} | {n_ok}/{len(runs)}")


@app.local_entrypoint()
def main():
    import json as json_mod
    import numpy as np
    from pathlib import Path

    results_dir = Path("results/exp_egd")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_gpu = {}

    # --- Parity (standard config, lr=0.1 for both) ---
    print("=" * 70)
    print("  EGD vs SGD on GPU: Sparse Parity (n=20, k=3, hidden=200)")
    print("=" * 70)

    wall_start = time.time()
    parity_result = run_gpu_egd.remote(
        task="parity", hidden=200, n_train=1000,
        lr_sgd=0.1, lr_egd=0.1, batch_size=32)
    wall_elapsed = time.time() - wall_start
    all_gpu['parity_h200'] = parity_result
    print_summary("Parity h=200", parity_result)
    print(f"  Wall time: {wall_elapsed:.1f}s")

    # --- Parity (small config for speed push) ---
    print("\n" + "=" * 70)
    print("  EGD vs SGD on GPU: Sparse Parity SPEED (hidden=50, n=500)")
    print("=" * 70)

    wall_start = time.time()
    parity_small = run_gpu_egd.remote(
        task="parity", hidden=50, n_train=500,
        lr_sgd=0.1, lr_egd=0.1, batch_size=32)
    wall_elapsed = time.time() - wall_start
    all_gpu['parity_h50'] = parity_small
    print_summary("Parity h=50", parity_small)
    print(f"  Wall time: {wall_elapsed:.1f}s")

    # --- Sum (standard config) ---
    print("\n" + "=" * 70)
    print("  EGD vs SGD on GPU: Sparse Sum (n=20, k=3, hidden=200)")
    print("=" * 70)

    wall_start = time.time()
    sum_result = run_gpu_egd.remote(
        task="sum", hidden=200, n_train=1000,
        lr_sgd=0.1, lr_egd=0.1, batch_size=32)
    wall_elapsed = time.time() - wall_start
    all_gpu['sum_h200'] = sum_result
    print_summary("Sum h=200", sum_result)
    print(f"  Wall time: {wall_elapsed:.1f}s")

    # --- Grand summary ---
    print("\n\n" + "=" * 70)
    print("  GRAND SUMMARY: EGD vs SGD on GPU")
    print("=" * 70)

    for config_name, data in all_gpu.items():
        sgd_runs = [r for r in data['results'] if r['method'] == 'sgd']
        egd_runs = [r for r in data['results'] if r['method'] == 'egd']
        sgd_avg = np.mean([r['time_ms'] for r in sgd_runs])
        egd_avg = np.mean([r['time_ms'] for r in egd_runs])
        speedup = sgd_avg / egd_avg if egd_avg > 0 else float('inf')
        sgd_ok = sum(1 for r in sgd_runs if r['best_acc'] >= 0.95)
        egd_ok = sum(1 for r in egd_runs if r['best_acc'] >= 0.95)
        print(f"  {config_name:<20}: SGD {sgd_avg:.1f}ms ({sgd_ok}/5) | "
              f"EGD {egd_avg:.1f}ms ({egd_ok}/5) | "
              f"speedup={speedup:.2f}x")

    # --- Save ---
    gpu_path = results_dir / 'gpu_results.json'
    with open(gpu_path, 'w') as f:
        json_mod.dump(all_gpu, f, indent=2, default=str)
    print(f"\n  Saved: {gpu_path}")

    cost = sum(time.time() - wall_start for _ in [1]) * L4_COST_PER_SEC
    print(f"  Estimated total cost: ~${len(all_gpu) * 0.003:.3f}")
