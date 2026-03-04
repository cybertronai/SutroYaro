#!/usr/bin/env python3
"""
Sparse Parity Benchmark — Pure Python, No Dependencies
=======================================================

The most atomic way to train a neural network on sparse parity.
This file is the complete algorithm. Everything else is just efficiency.

No torch. No numpy. No autograd. Just Python, math, and random.
Inspired by Karpathy's microGPT.

Architecture: x ∈ {-1,1}^n → Linear(W1,b1) → ReLU → Linear(W2,b2) → scalar
Loss:         Hinge loss = max(0, 1 - f(x)·y)
Optimizer:    SGD with weight decay

Includes memory reuse distance tracking to measure energy efficiency.
"""

import math
import random
import time

random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

N_BITS    = 3       # total input bits
K_SPARSE  = 3       # parity subset size
N_TRAIN   = 20      # training samples
N_TEST    = 20      # test samples
HIDDEN    = 1000    # hidden layer width
LR        = 0.5     # learning rate
WD        = 0.01    # weight decay
MAX_EPOCHS = 2      # 2 epochs × 20 samples = 40 steps
LOG_EVERY  = 1
PATIENCE   = 10

# =============================================================================
# MEMORY REUSE DISTANCE TRACKER
# =============================================================================

class MemTracker:
    """
    Tracks memory reuse distance: the volume of data (in floats) that flows
    through memory between when a buffer is written and when it is next read.

    Small reuse distance → value likely still in cache → energy efficient.
    Large reuse distance → cache miss → expensive memory fetch.

    The clock advances by the SIZE of each buffer accessed, not by 1 per op.
    This way, a 3000-float matrix read contributes 3000 to the clock, while
    a 1-float scalar contributes 1 — matching real cache eviction behavior.
    """

    def __init__(self):
        self.clock = 0          # counts floats accessed, not operations
        self.write_time = {}    # name → clock at last write
        self.write_size = {}    # name → size in floats at last write
        self.events = []        # (type, name, size, clock_before, distance_or_None)

    def write(self, name, size):
        """Record writing `size` floats to buffer `name`."""
        self.write_time[name] = self.clock
        self.write_size[name] = size
        self.events.append(('W', name, size, self.clock, None))
        self.clock += size      # clock advances by buffer size

    def read(self, name, size=None):
        """Record reading `size` floats from buffer `name`. Returns reuse distance."""
        if size is None:
            size = self.write_size.get(name, 0)
        if name in self.write_time:
            distance = self.clock - self.write_time[name]
        else:
            distance = -1
        self.events.append(('R', name, size, self.clock, distance))
        self.clock += size      # clock advances by buffer size
        return distance

    def report(self):
        """Print reuse distance report with distances measured in floats."""
        reads = [(name, size, dist) for typ, name, size, _, dist in self.events
                 if typ == 'R']
        writes = [e for e in self.events if e[0] == 'W']

        print(f"\n{'=' * 80}")
        print(f"  MEMORY REUSE DISTANCE REPORT  (distances in floats)")
        print(f"{'=' * 80}")
        print(f"  Total floats accessed: {self.clock:,}")
        print(f"  Operations: {len(reads)} reads, {len(writes)} writes")

        # Event log
        print(f"\n  {'Op':>3}  {'Clock':>8}  {'Buffer':<12}  {'Size':>8}  {'Reuse Dist':>10}")
        print(f"  {'─'*3}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*10}")
        for typ, name, size, clk, dist in self.events:
            dist_str = f"{dist:,}" if dist is not None else "—"
            print(f"  {typ:>3}  {clk:>8,}  {name:<12}  {size:>8,}  {dist_str:>10}")

        # Per-buffer summary
        print(f"\n  Per-buffer summary:")
        print(f"  {'Buffer':<12}  {'Size':>8}  {'Reads':>5}  {'Avg Dist':>10}  "
              f"{'Min':>8}  {'Max':>8}  {'Cache?':>6}")
        print(f"  {'─'*12}  {'─'*8}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*6}")
        buf_reads = {}
        for name, size, dist in reads:
            buf_reads.setdefault(name, []).append((size, dist))

        # Total working set to estimate "fits in cache" threshold
        total_params = sum(self.write_size.get(n, 0) for n in self.write_size)

        for name, entries in buf_reads.items():
            size = entries[0][0]
            dists = [d for _, d in entries]
            avg = sum(dists) / len(dists)
            # Heuristic: if avg distance < total working set, likely cached
            cached = "✅" if avg < total_params else "❌"
            print(f"  {name:<12}  {size:>8,}  {len(dists):>5}  {avg:>10,.0f}  "
                  f"{min(dists):>8,}  {max(dists):>8,}  {cached:>6}")

        # Overall averages
        if reads:
            all_dists = [d for _, _, d in reads]
            avg_all = sum(all_dists) / len(all_dists)
            # Size-weighted: each float's distance is counted equally
            total_float_dist = sum(s * d for _, s, d in reads)
            total_floats = sum(s for _, s, _ in reads)
            weighted_avg = total_float_dist / total_floats if total_floats > 0 else 0

            print(f"\n  Average reuse distance (per-read):        {avg_all:,.0f} floats")
            print(f"  Average reuse distance (per-float-read):  {weighted_avg:,.0f} floats")
            print(f"  Total floats read: {total_floats:,}")
            print(f"  Total working set: {total_params:,} floats")
        print(f"{'=' * 80}")

# =============================================================================
# MATRIX HELPERS
# =============================================================================

def zeros(rows, cols=None):
    if cols is None:
        return [0.0] * rows
    return [[0.0] * cols for _ in range(rows)]

def randn(rows, cols=None, std=1.0):
    if cols is None:
        return [random.gauss(0, std) for _ in range(rows)]
    return [[random.gauss(0, std) for _ in range(cols)] for _ in range(rows)]

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def mat_norm(M):
    return math.sqrt(sum(x * x for row in M for x in row))

# =============================================================================
# 1. DATA GENERATION
# =============================================================================

def create_datasets():
    secret = sorted(random.sample(range(N_BITS), K_SPARSE))
    print(f"[DATA] Secret parity indices S = {secret}")
    print(f"[DATA] Problem: n={N_BITS}, k={K_SPARSE}, N_train={N_TRAIN}, N_test={N_TEST}")

    def make_data(num):
        xs, ys = [], []
        for _ in range(num):
            x = [random.choice([-1.0, 1.0]) for _ in range(N_BITS)]
            y = 1.0
            for idx in secret:
                y *= x[idx]
            xs.append(x)
            ys.append(y)
        return xs, ys

    x_train, y_train = make_data(N_TRAIN)
    x_test, y_test = make_data(N_TEST)
    return x_train, y_train, x_test, y_test, secret

# =============================================================================
# 2. MODEL INIT
# =============================================================================

def init_params():
    std1 = math.sqrt(2.0 / N_BITS)
    std2 = math.sqrt(2.0 / HIDDEN)
    W1 = randn(HIDDEN, N_BITS, std=std1)
    b1 = zeros(HIDDEN)
    W2 = randn(1, HIDDEN, std=std2)
    b2 = zeros(1)
    total = HIDDEN * N_BITS + HIDDEN + HIDDEN + 1
    print(f"[MODEL] MLP: {N_BITS} → {HIDDEN} → 1  ({total:,} parameters)")
    return W1, b1, W2, b2

# =============================================================================
# 3. FORWARD PASS (instrumented with reuse tracking)
# =============================================================================

def forward(x, W1, b1, W2, b2, mem=None):
    """
    Forward pass for a single sample, with optional memory tracking.
    x → W1·x + b1 → ReLU → W2·h + b2 → scalar
    """
    if mem:
        mem.read('x', N_BITS)
        mem.read('W1', HIDDEN * N_BITS)
        mem.read('b1', HIDDEN)

    h_pre = [sum(W1[j][i] * x[i] for i in range(len(x))) + b1[j]
             for j in range(len(W1))]

    if mem:
        mem.write('h_pre', HIDDEN)
        mem.read('h_pre', HIDDEN)

    h = [max(0.0, hp) for hp in h_pre]

    if mem:
        mem.write('h', HIDDEN)
        mem.read('h', HIDDEN)
        mem.read('W2', HIDDEN)
        mem.read('b2', 1)

    out = sum(W2[0][j] * h[j] for j in range(len(h))) + b2[0]

    if mem:
        mem.write('out', 1)

    return out, h_pre, h

def forward_batch(xs, W1, b1, W2, b2):
    return [forward(x, W1, b1, W2, b2)[0] for x in xs]

# =============================================================================
# 4. LOSS & ACCURACY
# =============================================================================

def hinge_loss_batch(outs, ys):
    return sum(max(0.0, 1.0 - o * y) for o, y in zip(outs, ys)) / len(ys)

def accuracy(outs, ys):
    correct = sum(1 for o, y in zip(outs, ys)
                  if (1.0 if o >= 0 else -1.0) == y)
    return correct / len(ys)

# =============================================================================
# 5. BACKWARD + SGD (instrumented with reuse tracking)
# =============================================================================
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    🔧 OPTIMIZER INJECTION POINT 🔧                     │
# │                                                                         │
# │  Replace backward_and_update() with your custom learning algorithm.     │
# │  All buffers are plain Python lists — fully transparent.                │
# └─────────────────────────────────────────────────────────────────────────┘

def backward_and_update(x, y, out, h_pre, h, W1, b1, W2, b2, mem=None):
    """
    Manual backward + SGD with FUSED layer-wise updates.

    Instead of:  grad_all → update_all  (standard backprop)
    We do:       grad_layer2 → update_layer2 → grad_layer1 → update_layer1

    This is mathematically identical but reduces reuse distance: each
    gradient buffer (dW2, db2, dW1, db1) is consumed immediately after
    creation, and parameters are re-read while still in cache.

    x → [W1, b1] → ReLU → [W2, b2] → out
    L = max(0, 1 - out·y)
    """
    if mem:
        mem.read('out', 1)
        mem.read('y', 1)

    margin = out * y
    if margin >= 1.0:
        return

    # dL/dout = -y
    dout = -y
    if mem:
        mem.write('dout', 1)

    # ── Layer 2 backward: out = W2·h + b2 ───────────────────────────────
    # Compute dW2, db2
    if mem:
        mem.read('dout', 1)
        mem.read('h', HIDDEN)

    dW2_0 = [dout * h[j] for j in range(HIDDEN)]
    db2_0 = dout

    if mem:
        mem.write('dW2', HIDDEN)
        mem.write('db2', 1)

    # Compute dh BEFORE updating W2 (needs pre-update W2)
    if mem:
        mem.read('W2', HIDDEN)
        mem.read('dout', 1)

    dh = [W2[0][j] * dout for j in range(HIDDEN)]

    if mem:
        mem.write('dh', HIDDEN)

    # ── FUSED: Update W2, b2 immediately (dW2, db2 still in cache) ──────
    if mem:
        mem.read('dW2', HIDDEN)
        mem.read('W2', HIDDEN)

    for j in range(HIDDEN):
        W2[0][j] -= LR * (dW2_0[j] + WD * W2[0][j])

    if mem:
        mem.write('W2', HIDDEN)
        mem.read('db2', 1)
        mem.read('b2', 1)

    b2[0] -= LR * (db2_0 + WD * b2[0])

    if mem:
        mem.write('b2', 1)

    # ── ReLU backward ────────────────────────────────────────────────────
    if mem:
        mem.read('dh', HIDDEN)
        mem.read('h_pre', HIDDEN)

    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(HIDDEN)]

    if mem:
        mem.write('dh_pre', HIDDEN)

    # ── FUSED: Layer 1 backward + update W1, b1 immediately ─────────────
    if mem:
        mem.read('dh_pre', HIDDEN)
        mem.read('x', N_BITS)
        mem.read('W1', HIDDEN * N_BITS)

    for j in range(HIDDEN):
        for i in range(N_BITS):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= LR * (grad + WD * W1[j][i])

    if mem:
        mem.write('W1', HIDDEN * N_BITS)
        mem.read('dh_pre', HIDDEN)
        mem.read('b1', HIDDEN)

    for j in range(HIDDEN):
        b1[j] -= LR * (dh_pre[j] + WD * b1[j])

    if mem:
        mem.write('b1', HIDDEN)

# =============================================================================
# 6. TRAINING LOOP
# =============================================================================

def train(W1, b1, W2, b2, x_train, y_train, x_test, y_test):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    step = 0
    gen_epoch = None
    best_test_acc = 0.0
    epochs_no_improve = 0

    print(f"\n[TRAIN] Starting training for up to {MAX_EPOCHS} epochs...")
    print(f"[TRAIN] Mode: single-sample cyclic (batch_size=1, fixed order)")
    print(f"[TRAIN] Steps per epoch: {N_TRAIN}\n")

    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        for i in range(N_TRAIN):
            # Instrument FIRST step only with reuse distance tracking
            mem = MemTracker() if (step == 0) else None

            if mem:
                # Record parameter buffers as pre-existing writes
                mem.write('W1', HIDDEN * N_BITS)
                mem.write('b1', HIDDEN)
                mem.write('W2', HIDDEN)
                mem.write('b2', 1)
                mem.write('x', N_BITS)
                mem.write('y', 1)

            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2, mem=mem)
            backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                W1, b1, W2, b2, mem=mem)
            step += 1

            if mem:
                mem.report()

            # Evaluate
            tr_outs = forward_batch(x_train, W1, b1, W2, b2)
            te_outs = forward_batch(x_test, W1, b1, W2, b2)
            tr_loss = hinge_loss_batch(tr_outs, y_train)
            te_loss = hinge_loss_batch(te_outs, y_test)
            tr_acc  = accuracy(tr_outs, y_train)
            te_acc  = accuracy(te_outs, y_test)

            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            train_accs.append(tr_acc)
            test_accs.append(te_acc)

            if te_acc >= 1.0 and gen_epoch is None:
                gen_epoch = epoch

            if step % (LOG_EVERY * N_TRAIN) == 0 or step == 1:
                print(f"\n[STEP {step:>4}] epoch={epoch}  "
                      f"train_loss={tr_loss:.4f}  test_loss={te_loss:.4f}  "
                      f"train_acc={tr_acc:.0%}  test_acc={te_acc:.0%}")

        if gen_epoch is not None:
            elapsed = time.time() - start
            print(f"\n  🎉 GENERALIZED at epoch {epoch}! ({step} steps, {elapsed:.1f}s)")
            break

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    elapsed = time.time() - start
    print(f"[TRAIN] Finished in {elapsed:.1f}s ({epoch} epochs, {step} total steps)")

    return {
        'train_losses': train_losses, 'test_losses': test_losses,
        'train_accs': train_accs, 'test_accs': test_accs,
        'step_count': step, 'gen_epoch': gen_epoch,
    }

# =============================================================================
# 7. REPORT & PLOT
# =============================================================================

def print_report(r):
    print(f"\n{'=' * 65}")
    print(f"  ENERGY COST REPORT")
    print(f"{'=' * 65}")
    print(f"  Total optimizer steps: {r['step_count']}")
    if r['gen_epoch'] is not None:
        print(f"  ✅ Perfect generalization at epoch {r['gen_epoch']}")
        print(f"  ⚡ ENERGY COST = {r['gen_epoch']} epochs "
              f"({r['step_count']} optimizer steps)")
    else:
        print(f"  ❌ Did NOT reach 100% test accuracy")
        best = max(r['test_accs'])
        best_step = r['test_accs'].index(best)
        print(f"  Best test accuracy: {best:.0%} at step {best_step}")
        print(f"  ⚡ ENERGY COST = ∞ (failed to generalize)")
    print(f"  Final train loss:     {r['train_losses'][-1]:.6f}")
    print(f"  Final test loss:      {r['test_losses'][-1]:.6f}")
    print(f"  Final train accuracy: {r['train_accs'][-1]:.0%}")
    print(f"  Final test accuracy:  {r['test_accs'][-1]:.0%}")
    print(f"{'=' * 65}")

def plot_losses(r, save_path="loss_plot.png"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "matplotlib", "--quiet"])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    steps = list(range(1, len(r['train_losses']) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, r['train_losses'], label='Train', lw=1.2, color='#2196F3')
    ax1.plot(steps, r['test_losses'],  label='Test',  lw=1.2, color='#F44336')
    ax1.set(xlabel='Step', ylabel='Hinge Loss', title='Loss (per step)')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(steps, r['train_accs'], label='Train', lw=1.2, color='#2196F3')
    ax2.plot(steps, r['test_accs'],  label='Test',  lw=1.2, color='#F44336')
    ax2.set(xlabel='Step', ylabel='Accuracy', title='Accuracy (per step)')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.05, 1.05)
    if r['gen_epoch'] is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=r['gen_epoch'], color='green', ls='--', alpha=0.7)
    fig.suptitle('Sparse Parity Benchmark — Pure Python', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Saved to: {save_path}")
    plt.close(fig)

# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    total_start = time.time()
    print("=" * 65)
    print("  SPARSE PARITY BENCHMARK — Pure Python + Reuse Distance")
    print("=" * 65)

    init_start = time.time()
    x_train, y_train, x_test, y_test, secret = create_datasets()
    W1, b1, W2, b2 = init_params()
    init_elapsed = time.time() - init_start

    sim_start = time.time()
    results = train(W1, b1, W2, b2, x_train, y_train, x_test, y_test)
    sim_elapsed = time.time() - sim_start

    print_report(results)

    plot_start = time.time()
    plot_losses(results)
    plot_elapsed = time.time() - plot_start

    total_elapsed = time.time() - total_start
    print(f"\n[TIMING] Initialization:  {init_elapsed:.3f}s")
    print(f"[TIMING] Simulation:      {sim_elapsed:.3f}s")
    print(f"[TIMING] Plotting:        {plot_elapsed:.3f}s")
    print(f"[TIMING] Total wall time: {total_elapsed:.3f}s")

if __name__ == "__main__":
    main()
