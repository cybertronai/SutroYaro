"""
Experiment B: Instrument mini-batch SGD for ARD measurement.

Compare single-sample vs batch-32 ARD to quantify the memory reuse
benefit of mini-batch training.

Key insight: In mini-batch SGD, parameters are read ONCE per batch
(not batch_size times), and written ONCE after gradient accumulation.
Activations/gradients are per-sample temporaries. This should dramatically
improve ARD because the large parameter buffers stay in cache.

Config: n_bits=20, hidden=1000 (matching exp1 winning config).
"""

import sys
import json
import time
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params
from sparse_parity.tracker import MemTracker
from sparse_parity.metrics import save_json, timestamp


# ---------------------------------------------------------------------------
# Single-sample step with full tracker instrumentation
# (Mirrors train.py forward + backward_and_update, single sample)
# ---------------------------------------------------------------------------

def single_sample_step(x, y, W1, b1, W2, b2, config, tracker):
    """One SGD step on a single sample, fully instrumented."""
    hidden = config.hidden
    n_bits = config.n_bits

    # --- Write initial buffers (params + input) ---
    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)
    tracker.write('x', n_bits)
    tracker.write('y', 1)

    # --- Forward pass ---
    tracker.read('x', n_bits)
    tracker.read('W1', hidden * n_bits)
    tracker.read('b1', hidden)

    h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j]
             for j in range(hidden)]

    tracker.write('h_pre', hidden)
    tracker.read('h_pre', hidden)

    h = [max(0.0, v) for v in h_pre]

    tracker.write('h', hidden)
    tracker.read('h', hidden)
    tracker.read('W2', hidden)
    tracker.read('b2', 1)

    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    tracker.write('out', 1)

    # --- Backward pass ---
    tracker.read('out', 1)
    tracker.read('y', 1)

    margin = out * y
    if margin >= 1.0:
        return  # no gradient

    dout = -y
    tracker.write('dout', 1)

    # Layer 2 gradients
    tracker.read('dout', 1)
    tracker.read('h', hidden)
    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout
    tracker.write('dW2', hidden)
    tracker.write('db2', 1)

    # dh = W2^T * dout
    tracker.read('W2', hidden)
    tracker.read('dout', 1)
    dh = [W2[0][j] * dout for j in range(hidden)]
    tracker.write('dh', hidden)

    # ReLU backward
    tracker.read('dh', hidden)
    tracker.read('h_pre', hidden)
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]
    tracker.write('dh_pre', hidden)

    # Layer 1 update
    tracker.read('dh_pre', hidden)
    tracker.read('x', n_bits)
    tracker.read('W1', hidden * n_bits)

    for j in range(hidden):
        for i in range(n_bits):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])

    tracker.write('W1', hidden * n_bits)

    # b1 update
    tracker.read('dh_pre', hidden)
    tracker.read('b1', hidden)
    for j in range(hidden):
        b1[j] -= config.lr * (dh_pre[j] + config.wd * b1[j])
    tracker.write('b1', hidden)

    # Layer 2 update
    tracker.read('dW2', hidden)
    tracker.read('W2', hidden)
    for j in range(hidden):
        W2[0][j] -= config.lr * (dW2_0[j] + config.wd * W2[0][j])
    tracker.write('W2', hidden)

    tracker.read('db2', 1)
    tracker.read('b2', 1)
    b2[0] -= config.lr * (db2_0 + config.wd * b2[0])
    tracker.write('b2', 1)


# ---------------------------------------------------------------------------
# Batch-32 step with full tracker instrumentation
# ---------------------------------------------------------------------------

def batch_step(batch_x, batch_y, W1, b1, W2, b2, config, tracker):
    """
    One mini-batch SGD step (batch_size samples), fully instrumented.

    Key difference from single-sample:
    - Parameters (W1, b1, W2, b2) are READ once at batch start
    - For each sample: forward/backward uses per-sample activation buffers
    - Gradients are accumulated into shared accumulators
    - Parameters are WRITTEN once at batch end (update step)
    """
    hidden = config.hidden
    n_bits = config.n_bits
    batch_size = len(batch_x)

    # --- Write initial parameter buffers ONCE ---
    tracker.write('W1', hidden * n_bits)
    tracker.write('b1', hidden)
    tracker.write('W2', hidden)
    tracker.write('b2', 1)

    # --- Read parameters ONCE at batch start ---
    tracker.read('W1', hidden * n_bits)
    tracker.read('b1', hidden)
    tracker.read('W2', hidden)
    tracker.read('b2', 1)

    # Initialize gradient accumulators
    acc_dW1 = [[0.0] * n_bits for _ in range(hidden)]
    acc_db1 = [0.0] * hidden
    acc_dW2 = [[0.0] * hidden]
    acc_db2 = [0.0]
    tracker.write('acc_dW1', hidden * n_bits)
    tracker.write('acc_db1', hidden)
    tracker.write('acc_dW2', hidden)
    tracker.write('acc_db2', 1)

    n_contributing = 0

    for s in range(batch_size):
        x = batch_x[s]
        y = batch_y[s]

        # --- Per-sample input ---
        tracker.write(f'x_{s}', n_bits)
        tracker.write(f'y_{s}', 1)

        # --- Forward pass (per-sample activations) ---
        tracker.read(f'x_{s}', n_bits)
        # Note: W1, b1, W2, b2 already in cache from batch-start read.
        # We still track reads for correctness, but they hit cache (short distance).
        tracker.read('W1', hidden * n_bits)
        tracker.read('b1', hidden)

        h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j]
                 for j in range(hidden)]

        tracker.write(f'h_pre_{s}', hidden)
        tracker.read(f'h_pre_{s}', hidden)

        h = [max(0.0, v) for v in h_pre]

        tracker.write(f'h_{s}', hidden)
        tracker.read(f'h_{s}', hidden)
        tracker.read('W2', hidden)
        tracker.read('b2', 1)

        out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

        tracker.write(f'out_{s}', 1)

        # --- Backward pass ---
        tracker.read(f'out_{s}', 1)
        tracker.read(f'y_{s}', 1)

        margin = out * y
        if margin >= 1.0:
            continue  # no gradient for this sample

        dout = -y
        tracker.write(f'dout_{s}', 1)

        # Layer 2 gradients
        tracker.read(f'dout_{s}', 1)
        tracker.read(f'h_{s}', hidden)
        dW2_local = [dout * h[j] for j in range(hidden)]
        db2_local = dout
        tracker.write(f'dW2_{s}', hidden)
        tracker.write(f'db2_{s}', 1)

        # dh = W2^T * dout
        tracker.read('W2', hidden)
        tracker.read(f'dout_{s}', 1)
        dh = [W2[0][j] * dout for j in range(hidden)]
        tracker.write(f'dh_{s}', hidden)

        # ReLU backward
        tracker.read(f'dh_{s}', hidden)
        tracker.read(f'h_pre_{s}', hidden)
        dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]
        tracker.write(f'dh_pre_{s}', hidden)

        # --- Accumulate into shared gradient buffers ---
        tracker.read(f'dh_pre_{s}', hidden)
        tracker.read(f'x_{s}', n_bits)

        # dW1 accumulation
        tracker.read('acc_dW1', hidden * n_bits)
        for j in range(hidden):
            for i in range(n_bits):
                acc_dW1[j][i] += dh_pre[j] * x[i]
        tracker.write('acc_dW1', hidden * n_bits)

        # db1 accumulation
        tracker.read('acc_db1', hidden)
        for j in range(hidden):
            acc_db1[j] += dh_pre[j]
        tracker.write('acc_db1', hidden)

        # dW2 accumulation
        tracker.read(f'dW2_{s}', hidden)
        tracker.read('acc_dW2', hidden)
        for j in range(hidden):
            acc_dW2[0][j] += dW2_local[j]
        tracker.write('acc_dW2', hidden)

        # db2 accumulation
        tracker.read(f'db2_{s}', 1)
        tracker.read('acc_db2', 1)
        acc_db2[0] += db2_local
        tracker.write('acc_db2', 1)

        n_contributing += 1

    if n_contributing == 0:
        return

    inv_n = 1.0 / n_contributing

    # --- Parameter update: READ accumulators + params, WRITE params ONCE ---
    tracker.read('acc_dW1', hidden * n_bits)
    tracker.read('W1', hidden * n_bits)
    for j in range(hidden):
        for i in range(n_bits):
            W1[j][i] -= config.lr * (acc_dW1[j][i] * inv_n + config.wd * W1[j][i])
    tracker.write('W1', hidden * n_bits)

    tracker.read('acc_db1', hidden)
    tracker.read('b1', hidden)
    for j in range(hidden):
        b1[j] -= config.lr * (acc_db1[j] * inv_n + config.wd * b1[j])
    tracker.write('b1', hidden)

    tracker.read('acc_dW2', hidden)
    tracker.read('W2', hidden)
    for j in range(hidden):
        W2[0][j] -= config.lr * (acc_dW2[0][j] * inv_n + config.wd * W2[0][j])
    tracker.write('W2', hidden)

    tracker.read('acc_db2', 1)
    tracker.read('b2', 1)
    b2[0] -= config.lr * (acc_db2[0] * inv_n + config.wd * b2[0])
    tracker.write('b2', 1)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run():
    config = Config(
        n_bits=20,
        k_sparse=3,
        n_train=500,
        n_test=200,
        hidden=1000,
        lr=0.1,
        wd=0.01,
        max_epochs=1,
        seed=42,
    )
    batch_size = 32

    print("=" * 70)
    print("  EXPERIMENT B: Mini-Batch ARD vs Single-Sample ARD")
    print("=" * 70)
    print(f"  n_bits={config.n_bits}, hidden={config.hidden}")
    print(f"  batch_size={batch_size}")
    print()

    # Generate data
    x_train, y_train, x_test, y_test, secret = generate(config)
    print(f"  Secret indices: {secret}")

    # Initialize model
    W1, b1, W2, b2 = init_params(config)

    # -----------------------------------------------------------------------
    # Measure 1: Single-sample ARD (32 consecutive single-sample steps)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  SINGLE-SAMPLE: 32 consecutive single-sample SGD steps")
    print("-" * 70)

    # Deep copy so both experiments start from same weights
    W1_ss = [row[:] for row in W1]
    b1_ss = b1[:]
    W2_ss = [row[:] for row in W2]
    b2_ss = b2[:]

    tracker_ss = MemTracker()
    t0 = time.time()

    for i in range(batch_size):
        single_sample_step(
            x_train[i], y_train[i],
            W1_ss, b1_ss, W2_ss, b2_ss,
            config, tracker_ss
        )

    t_ss = time.time() - t0
    summary_ss = tracker_ss.summary()
    tracker_ss.report()

    # -----------------------------------------------------------------------
    # Measure 2: Batch-32 ARD (one mini-batch step, 32 samples)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  BATCH-32: One mini-batch SGD step (32 samples)")
    print("-" * 70)

    W1_b = [row[:] for row in W1]
    b1_b = b1[:]
    W2_b = [row[:] for row in W2]
    b2_b = b2[:]

    tracker_b = MemTracker()
    t0 = time.time()

    batch_step(
        x_train[:batch_size], y_train[:batch_size],
        W1_b, b1_b, W2_b, b2_b,
        config, tracker_b
    )

    t_b = time.time() - t0
    summary_b = tracker_b.summary()
    tracker_b.report()

    # -----------------------------------------------------------------------
    # Side-by-side comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Single-Sample':>15} {'Batch-32':>15} {'Ratio':>10}")
    print(f"  {'─'*30} {'─'*15} {'─'*15} {'─'*10}")

    ard_ss = summary_ss['weighted_ard']
    ard_b = summary_b['weighted_ard']
    ratio = ard_ss / ard_b if ard_b > 0 else float('inf')

    floats_ss = summary_ss['total_floats_accessed']
    floats_b = summary_b['total_floats_accessed']
    floats_ratio = floats_ss / floats_b if floats_b > 0 else float('inf')

    reads_ss = summary_ss['reads']
    reads_b = summary_b['reads']

    writes_ss = summary_ss['writes']
    writes_b = summary_b['writes']

    print(f"  {'Weighted ARD (floats)':<30} {ard_ss:>15,.0f} {ard_b:>15,.0f} {ratio:>10.2f}x")
    print(f"  {'Total floats accessed':<30} {floats_ss:>15,} {floats_b:>15,} {floats_ratio:>10.2f}x")
    print(f"  {'Read operations':<30} {reads_ss:>15,} {reads_b:>15,}")
    print(f"  {'Write operations':<30} {writes_ss:>15,} {writes_b:>15,}")
    print(f"  {'Wall-clock time (s)':<30} {t_ss:>15.3f} {t_b:>15.3f}")

    # Per-buffer comparison for key buffers
    print(f"\n  Per-buffer ARD for key parameters:")
    print(f"  {'Buffer':<15} {'SS Avg Dist':>12} {'Batch Avg':>12} {'Improvement':>12}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*12}")

    for buf in ['W1', 'b1', 'W2', 'b2']:
        ss_info = summary_ss['per_buffer'].get(buf, {})
        b_info = summary_b['per_buffer'].get(buf, {})
        ss_avg = ss_info.get('avg_dist', 0)
        b_avg = b_info.get('avg_dist', 0)
        improvement = ss_avg / b_avg if b_avg > 0 else float('inf')
        ss_reads = ss_info.get('read_count', 0)
        b_reads = b_info.get('read_count', 0)
        print(f"  {buf:<15} {ss_avg:>12,.0f} {b_avg:>12,.0f} {improvement:>11.1f}x"
              f"  (reads: {ss_reads} vs {b_reads})")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Also run single-batch ARD for batch sizes 1, 4, 8, 16, 32, 64
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ARD vs BATCH SIZE SWEEP")
    print("=" * 70)

    batch_sizes = [1, 4, 8, 16, 32, 64]
    sweep_results = []

    for bs in batch_sizes:
        W1_t = [row[:] for row in W1]
        b1_t = b1[:]
        W2_t = [row[:] for row in W2]
        b2_t = b2[:]

        tracker_t = MemTracker()

        if bs == 1:
            single_sample_step(
                x_train[0], y_train[0],
                W1_t, b1_t, W2_t, b2_t,
                config, tracker_t
            )
        else:
            batch_step(
                x_train[:bs], y_train[:bs],
                W1_t, b1_t, W2_t, b2_t,
                config, tracker_t
            )

        s = tracker_t.summary()
        ard = s['weighted_ard']
        total = s['total_floats_accessed']
        # ARD per sample = total work amortized over batch
        ard_per_sample = ard  # ARD is already an average
        floats_per_sample = total / bs

        sweep_results.append({
            'batch_size': bs,
            'weighted_ard': ard,
            'total_floats': total,
            'floats_per_sample': floats_per_sample,
            'reads': s['reads'],
            'writes': s['writes'],
        })

        print(f"  batch_size={bs:3d} | ARD={ard:>10,.0f} | "
              f"total_floats={total:>12,} | floats/sample={floats_per_sample:>10,.0f} | "
              f"reads={s['reads']:>5} writes={s['writes']:>5}")

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        'experiment': 'exp_b_batch_ard',
        'question': 'How much does mini-batch SGD improve ARD vs single-sample?',
        'config': {
            'n_bits': config.n_bits,
            'k_sparse': config.k_sparse,
            'hidden': config.hidden,
            'lr': config.lr,
            'wd': config.wd,
            'batch_size': batch_size,
            'seed': config.seed,
        },
        'single_sample_32steps': {
            'weighted_ard': ard_ss,
            'total_floats_accessed': floats_ss,
            'reads': reads_ss,
            'writes': writes_ss,
            'wall_clock_s': t_ss,
            'per_buffer': {k: {kk: vv for kk, vv in v.items() if kk != 'distances'}
                           for k, v in summary_ss['per_buffer'].items()},
        },
        'batch_32': {
            'weighted_ard': ard_b,
            'total_floats_accessed': floats_b,
            'reads': reads_b,
            'writes': writes_b,
            'wall_clock_s': t_b,
            'per_buffer': {k: {kk: vv for kk, vv in v.items() if kk != 'distances'}
                           for k, v in summary_b['per_buffer'].items()},
        },
        'ard_improvement_ratio': ratio,
        'total_floats_ratio': floats_ratio,
        'batch_size_sweep': sweep_results,
    }

    results_dir = Path(__file__).resolve().parents[3] / 'results' / 'exp_b_batch_ard'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'results.json'
    save_json(results, results_path)
    print(f"\n  Results saved to: {results_path}")

    return results


if __name__ == '__main__':
    run()
