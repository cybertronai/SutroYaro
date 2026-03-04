# Sparse Parity Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete end-to-end sparse parity pipeline that generates data, trains a neural net to >90% accuracy, measures Average Reuse Distance, improves it via fused and per-layer updates, and scales to 20 bits.

**Architecture:** Modular pure-Python package in `src/sparse_parity/`. Each module is <200 lines with a single responsibility. Training variants are separate files sharing the same model/data/tracker interfaces. A runner script executes all phases sequentially and writes JSON + markdown + plot outputs.

**Tech Stack:** Pure Python 3.12 (no numpy/torch), pytest for tests, matplotlib for plots (optional, lazy-imported).

---

### Task 1: Scaffold package and reference code

**Files:**
- Create: `src/sparse_parity/__init__.py`
- Create: `src/sparse_parity/config.py`
- Create: `src/sparse_parity/reference/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `results/.gitkeep`

**Step 1: Create package scaffold**

```python
# src/sparse_parity/__init__.py
"""Sparse Parity Pipeline — Pure Python, No Dependencies."""

# src/sparse_parity/reference/__init__.py
# Read-only reference from cybertronai/sutro

# tests/__init__.py
# (empty)
```

**Step 2: Write config.py with all constants**

```python
# src/sparse_parity/config.py
"""Configuration constants for sparse parity experiments."""

from dataclasses import dataclass


@dataclass
class Config:
    """Experiment configuration. All fields have sensible defaults for 3-bit parity."""
    n_bits: int = 3
    k_sparse: int = 3
    n_train: int = 20
    n_test: int = 20
    hidden: int = 1000
    lr: float = 0.5
    wd: float = 0.01
    max_epochs: int = 10
    seed: int = 42
    patience: int = 10

    @property
    def total_params(self):
        return self.hidden * self.n_bits + self.hidden + self.hidden + 1


# Preset for 20-bit scaling experiment
SCALE_CONFIG = Config(n_bits=20, k_sparse=3, n_train=200, n_test=200, hidden=2000, max_epochs=50)
```

**Step 3: Write conftest.py with shared fixtures**

```python
# tests/conftest.py
import sys
from pathlib import Path

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_parity.config import Config

import pytest


@pytest.fixture
def small_config():
    """Tiny config for fast tests."""
    return Config(n_bits=3, k_sparse=3, n_train=20, n_test=20, hidden=100, max_epochs=5, seed=42)


@pytest.fixture
def scale_config():
    """20-bit config for scaling tests."""
    return Config(n_bits=20, k_sparse=3, n_train=200, n_test=200, hidden=2000, max_epochs=50, seed=42)
```

**Step 4: Download reference implementation**

Run: `curl -sL "https://raw.githubusercontent.com/cybertronai/sutro/main/sparse_parity_benchmark.py" -o src/sparse_parity/reference/sparse_parity_benchmark.py`

**Step 5: Create results directory**

Run: `mkdir -p results && touch results/.gitkeep`

**Step 6: Verify imports work**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && python3 -c "from sparse_parity.config import Config; c = Config(); print(f'OK: {c.n_bits}-bit, {c.total_params} params')"`

Expected: `OK: 3-bit, 4001 params`

**Step 7: Commit**

```bash
git add src/sparse_parity/ tests/ results/.gitkeep
git commit -m "feat: scaffold sparse parity package with config and reference"
```

---

### Task 2: Data generation with tests

**Files:**
- Create: `src/sparse_parity/data.py`
- Create: `tests/test_data.py`

**Step 1: Write the failing test**

```python
# tests/test_data.py
from sparse_parity.config import Config
from sparse_parity.data import generate


def test_generate_returns_correct_shapes(small_config):
    x_train, y_train, x_test, y_test, secret = generate(small_config)
    assert len(x_train) == small_config.n_train
    assert len(y_train) == small_config.n_train
    assert len(x_test) == small_config.n_test
    assert len(y_test) == small_config.n_test
    assert len(secret) == small_config.k_sparse
    assert len(x_train[0]) == small_config.n_bits


def test_labels_match_parity(small_config):
    x_train, y_train, _, _, secret = generate(small_config)
    for x, y in zip(x_train, y_train):
        expected = 1.0
        for idx in secret:
            expected *= 1.0 if x[idx] > 0 else -1.0
        assert y == expected, f"Parity mismatch: x={x}, secret={secret}, got {y}, expected {expected}"


def test_inputs_are_plus_minus_one(small_config):
    x_train, _, x_test, _, _ = generate(small_config)
    for xs in [x_train, x_test]:
        for x in xs:
            for val in x:
                assert val in (-1.0, 1.0)


def test_labels_are_plus_minus_one(small_config):
    _, y_train, _, y_test, _ = generate(small_config)
    for ys in [y_train, y_test]:
        for y in ys:
            assert y in (-1.0, 1.0)


def test_reproducible_with_same_seed(small_config):
    result1 = generate(small_config)
    result2 = generate(small_config)
    assert result1[0] == result2[0]  # x_train identical
    assert result1[4] == result2[4]  # secret identical


def test_secret_indices_in_range(small_config):
    _, _, _, _, secret = generate(small_config)
    for idx in secret:
        assert 0 <= idx < small_config.n_bits
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && python3 -m pytest tests/test_data.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'sparse_parity.data'`

**Step 3: Write data.py**

```python
# src/sparse_parity/data.py
"""Phase 1: Dataset generation for sparse parity."""

import random

from .config import Config


def generate(config: Config):
    """
    Generate (n,k)-sparse parity train/test datasets.

    Returns (x_train, y_train, x_test, y_test, secret_indices).
    Inputs are {-1, +1}. Labels are product of inputs at secret indices.
    """
    rng = random.Random(config.seed)

    # Pick secret parity indices
    secret = sorted(rng.sample(range(config.n_bits), config.k_sparse))

    def make_data(n):
        xs, ys = [], []
        for _ in range(n):
            x = [rng.choice([-1.0, 1.0]) for _ in range(config.n_bits)]
            y = 1.0
            for idx in secret:
                y *= x[idx]
            xs.append(x)
            ys.append(y)
        return xs, ys

    x_train, y_train = make_data(config.n_train)
    x_test, y_test = make_data(config.n_test)

    return x_train, y_train, x_test, y_test, secret
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && python3 -m pytest tests/test_data.py -v`

Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/sparse_parity/data.py tests/test_data.py
git commit -m "feat: add data generation with parity label verification tests"
```

---

### Task 3: Model (MLP init + forward pass) with tests

**Files:**
- Create: `src/sparse_parity/model.py`
- Create: `tests/test_model.py`

**Step 1: Write the failing test**

```python
# tests/test_model.py
from sparse_parity.config import Config
from sparse_parity.model import init_params, forward


def test_init_params_shapes(small_config):
    W1, b1, W2, b2 = init_params(small_config)
    assert len(W1) == small_config.hidden
    assert len(W1[0]) == small_config.n_bits
    assert len(b1) == small_config.hidden
    assert len(W2) == 1
    assert len(W2[0]) == small_config.hidden
    assert len(b2) == 1


def test_forward_returns_scalar(small_config):
    W1, b1, W2, b2 = init_params(small_config)
    x = [1.0] * small_config.n_bits
    out, h_pre, h = forward(x, W1, b1, W2, b2)
    assert isinstance(out, float)
    assert len(h_pre) == small_config.hidden
    assert len(h) == small_config.hidden


def test_relu_nonnegative(small_config):
    W1, b1, W2, b2 = init_params(small_config)
    x = [1.0, -1.0, 1.0]
    _, _, h = forward(x, W1, b1, W2, b2)
    for val in h:
        assert val >= 0.0


def test_init_reproducible(small_config):
    params1 = init_params(small_config)
    params2 = init_params(small_config)
    assert params1[0] == params2[0]  # W1 identical
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model.py -v`

Expected: FAIL with import error

**Step 3: Write model.py**

```python
# src/sparse_parity/model.py
"""Phase 2: MLP model — init and forward pass."""

import math
import random

from .config import Config


def init_params(config: Config):
    """Initialize 2-layer MLP: input -> hidden (ReLU) -> scalar. Kaiming init."""
    rng = random.Random(config.seed + 1)  # different seed from data
    std1 = math.sqrt(2.0 / config.n_bits)
    std2 = math.sqrt(2.0 / config.hidden)

    W1 = [[rng.gauss(0, std1) for _ in range(config.n_bits)] for _ in range(config.hidden)]
    b1 = [0.0] * config.hidden
    W2 = [[rng.gauss(0, std2) for _ in range(config.hidden)]]
    b2 = [0.0]

    return W1, b1, W2, b2


def forward(x, W1, b1, W2, b2, tracker=None):
    """
    Forward pass for a single sample.
    x -> W1*x + b1 -> ReLU -> W2*h + b2 -> scalar
    Returns (out, h_pre, h).
    """
    hidden = len(W1)
    n_bits = len(x)

    if tracker:
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)
        tracker.read('b1', hidden)

    h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j] for j in range(hidden)]

    if tracker:
        tracker.write('h_pre', hidden)
        tracker.read('h_pre', hidden)

    h = [max(0.0, v) for v in h_pre]

    if tracker:
        tracker.write('h', hidden)
        tracker.read('h', hidden)
        tracker.read('W2', hidden)
        tracker.read('b2', 1)

    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    if tracker:
        tracker.write('out', 1)

    return out, h_pre, h


def forward_batch(xs, W1, b1, W2, b2):
    """Forward pass for multiple samples. Returns list of outputs."""
    return [forward(x, W1, b1, W2, b2)[0] for x in xs]
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_model.py -v`

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/sparse_parity/model.py tests/test_model.py
git commit -m "feat: add MLP model with Kaiming init and instrumented forward pass"
```

---

### Task 4: MemTracker (ARD measurement) with tests

**Files:**
- Create: `src/sparse_parity/tracker.py`
- Create: `tests/test_tracker.py`

**Step 1: Write the failing test**

```python
# tests/test_tracker.py
from sparse_parity.tracker import MemTracker


def test_write_read_distance():
    t = MemTracker()
    t.write('a', 100)    # clock: 0 -> 100
    t.write('b', 50)     # clock: 100 -> 150
    dist = t.read('a')   # clock: 150, distance = 150 - 0 = 150
    assert dist == 150


def test_clock_advances_by_size():
    t = MemTracker()
    t.write('a', 100)
    t.write('b', 200)
    assert t.clock == 300


def test_read_unknown_returns_negative():
    t = MemTracker()
    dist = t.read('nonexistent', 10)
    assert dist == -1


def test_weighted_ard():
    t = MemTracker()
    t.write('small', 1)    # clock 0->1
    t.write('big', 1000)   # clock 1->1001
    t.read('small', 1)     # dist=1001, 1 float
    t.read('big', 1000)    # dist=1000, 1000 floats
    summary = t.summary()
    # Weighted avg dominated by 'big' (1000 floats at dist 1000)
    # vs 'small' (1 float at dist 1001)
    # = (1*1001 + 1000*1000) / (1 + 1000) = 1001001/1001 ≈ 1000
    assert 999 < summary['weighted_ard'] < 1002


def test_to_json_has_required_fields():
    t = MemTracker()
    t.write('x', 10)
    t.read('x')
    j = t.to_json()
    assert 'total_floats_accessed' in j
    assert 'reads' in j
    assert 'writes' in j
    assert 'weighted_ard' in j
    assert 'per_buffer' in j
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_tracker.py -v`

Expected: FAIL with import error

**Step 3: Write tracker.py**

```python
# src/sparse_parity/tracker.py
"""Phase 3: Memory Reuse Distance Tracker for energy efficiency measurement."""


class MemTracker:
    """
    Tracks Average Reuse Distance (ARD) — a proxy for energy efficiency.

    Clock advances by buffer SIZE (floats), not operation count.
    Small ARD = data stays in cache = cheap.
    Large ARD = cache miss = expensive external memory access.
    """

    def __init__(self):
        self.clock = 0
        self._write_time = {}
        self._write_size = {}
        self._events = []

    def write(self, name, size):
        """Record writing `size` floats to buffer `name`."""
        self._write_time[name] = self.clock
        self._write_size[name] = size
        self._events.append(('W', name, size, self.clock, None))
        self.clock += size

    def read(self, name, size=None):
        """Record reading from buffer `name`. Returns reuse distance."""
        if size is None:
            size = self._write_size.get(name, 0)
        if name in self._write_time:
            distance = self.clock - self._write_time[name]
        else:
            distance = -1
        self._events.append(('R', name, size, self.clock, distance))
        self.clock += size
        return distance

    def summary(self):
        """Compute summary statistics."""
        reads = [(name, size, dist) for typ, name, size, _, dist in self._events
                 if typ == 'R' and dist >= 0]
        writes = [e for e in self._events if e[0] == 'W']

        if not reads:
            return {'total_floats_accessed': self.clock, 'reads': 0, 'writes': len(writes),
                    'weighted_ard': 0, 'per_buffer': {}}

        total_float_dist = sum(s * d for _, s, d in reads)
        total_floats = sum(s for _, s, _ in reads)
        weighted_ard = total_float_dist / total_floats if total_floats > 0 else 0

        per_buffer = {}
        for name, size, dist in reads:
            if name not in per_buffer:
                per_buffer[name] = {'size': size, 'distances': []}
            per_buffer[name]['distances'].append(dist)

        for name, info in per_buffer.items():
            dists = info['distances']
            info['avg_dist'] = sum(dists) / len(dists)
            info['min_dist'] = min(dists)
            info['max_dist'] = max(dists)
            info['read_count'] = len(dists)

        return {
            'total_floats_accessed': self.clock,
            'reads': len(reads),
            'writes': len(writes),
            'weighted_ard': weighted_ard,
            'total_floats_read': total_floats,
            'per_buffer': per_buffer,
        }

    def to_json(self):
        """Return JSON-serializable dict of all metrics."""
        return self.summary()

    def report(self):
        """Print human-readable report."""
        s = self.summary()
        print(f"\n{'=' * 70}")
        print(f"  MEMORY REUSE DISTANCE REPORT")
        print(f"{'=' * 70}")
        print(f"  Total floats accessed: {s['total_floats_accessed']:,}")
        print(f"  Operations: {s['reads']} reads, {s['writes']} writes")
        print(f"  Weighted ARD: {s['weighted_ard']:,.0f} floats")
        if s['per_buffer']:
            print(f"\n  {'Buffer':<12} {'Size':>8} {'Reads':>5} {'Avg Dist':>10} {'Min':>8} {'Max':>8}")
            print(f"  {'─'*12} {'─'*8} {'─'*5} {'─'*10} {'─'*8} {'─'*8}")
            for name, info in s['per_buffer'].items():
                print(f"  {name:<12} {info['size']:>8,} {info['read_count']:>5} "
                      f"{info['avg_dist']:>10,.0f} {info['min_dist']:>8,} {info['max_dist']:>8,}")
        print(f"{'=' * 70}")
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_tracker.py -v`

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/sparse_parity/tracker.py tests/test_tracker.py
git commit -m "feat: add MemTracker with ARD measurement and JSON export"
```

---

### Task 5: Metrics (loss, accuracy, reporting)

**Files:**
- Create: `src/sparse_parity/metrics.py`

**Step 1: Write metrics.py**

```python
# src/sparse_parity/metrics.py
"""Loss functions, accuracy, and result reporting."""

import json
import time
from pathlib import Path


def hinge_loss(outs, ys):
    """Mean hinge loss: avg(max(0, 1 - out*y))."""
    return sum(max(0.0, 1.0 - o * y) for o, y in zip(outs, ys)) / len(ys)


def accuracy(outs, ys):
    """Fraction where sign(out) matches y."""
    correct = sum(1 for o, y in zip(outs, ys) if (1.0 if o >= 0 else -1.0) == y)
    return correct / len(ys)


def save_json(data, path):
    """Save dict as JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def save_markdown(content, path):
    """Save string as markdown file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)


def timestamp():
    """Generate a timestamp string for filenames."""
    return time.strftime('%Y%m%d_%H%M%S')
```

**Step 2: Verify it imports**

Run: `python3 -c "from sparse_parity.metrics import hinge_loss, accuracy; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/sparse_parity/metrics.py
git commit -m "feat: add loss, accuracy, and reporting utilities"
```

---

### Task 6: Standard backprop training with tests

**Files:**
- Create: `src/sparse_parity/train.py`
- Create: `tests/test_train.py`

**Step 1: Write the failing test**

```python
# tests/test_train.py
from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params
from sparse_parity.train import train


def test_baseline_converges(small_config):
    """3-bit parity with standard backprop should reach >90% accuracy."""
    data = generate(small_config)
    x_train, y_train, x_test, y_test, secret = data
    W1, b1, W2, b2 = init_params(small_config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, small_config)
    assert result['best_test_acc'] > 0.9, f"Only reached {result['best_test_acc']:.0%}"


def test_train_returns_required_fields(small_config):
    data = generate(small_config)
    x_train, y_train, x_test, y_test, _ = data
    W1, b1, W2, b2 = init_params(small_config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, small_config)
    for key in ['train_losses', 'test_losses', 'train_accs', 'test_accs',
                'best_test_acc', 'total_steps', 'elapsed_s']:
        assert key in result, f"Missing key: {key}"


def test_train_under_one_second(small_config):
    data = generate(small_config)
    x_train, y_train, x_test, y_test, _ = data
    W1, b1, W2, b2 = init_params(small_config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, small_config)
    assert result['elapsed_s'] < 1.0, f"Took {result['elapsed_s']:.2f}s"
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_train.py -v`

Expected: FAIL with import error

**Step 3: Write train.py (standard backprop)**

```python
# src/sparse_parity/train.py
"""Standard backprop training loop for sparse parity."""

import time

from .model import forward, forward_batch
from .metrics import hinge_loss, accuracy
from .config import Config


def backward_and_update(x, y, out, h_pre, h, W1, b1, W2, b2, config, tracker=None):
    """Standard backprop: compute all gradients, then update all params."""
    hidden = len(W1)
    n_bits = len(x)

    if tracker:
        tracker.read('out', 1)
        tracker.read('y', 1)

    margin = out * y
    if margin >= 1.0:
        return

    dout = -y

    if tracker:
        tracker.write('dout', 1)
        tracker.read('dout', 1)
        tracker.read('h', hidden)

    # Layer 2 gradients
    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout

    if tracker:
        tracker.write('dW2', hidden)
        tracker.write('db2', 1)

    # dh = W2^T * dout
    if tracker:
        tracker.read('W2', hidden)
        tracker.read('dout', 1)

    dh = [W2[0][j] * dout for j in range(hidden)]

    if tracker:
        tracker.write('dh', hidden)
        tracker.read('dh', hidden)
        tracker.read('h_pre', hidden)

    # ReLU backward
    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    if tracker:
        tracker.write('dh_pre', hidden)

    # Layer 1 gradients + update
    if tracker:
        tracker.read('dh_pre', hidden)
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)

    for j in range(hidden):
        for i in range(n_bits):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])

    if tracker:
        tracker.write('W1', hidden * n_bits)
        tracker.read('dh_pre', hidden)
        tracker.read('b1', hidden)

    for j in range(hidden):
        b1[j] -= config.lr * (dh_pre[j] + config.wd * b1[j])

    if tracker:
        tracker.write('b1', hidden)

    # Layer 2 update (gradients computed earlier)
    if tracker:
        tracker.read('dW2', hidden)
        tracker.read('W2', hidden)

    for j in range(hidden):
        W2[0][j] -= config.lr * (dW2_0[j] + config.wd * W2[0][j])

    if tracker:
        tracker.write('W2', hidden)
        tracker.read('db2', 1)
        tracker.read('b2', 1)

    b2[0] -= config.lr * (db2_0 + config.wd * b2[0])

    if tracker:
        tracker.write('b2', 1)


def train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=0):
    """
    Train with standard backprop. Single-sample cyclic, no batching.
    If tracker_step >= 0, instrument that step with a new MemTracker.
    Returns dict with losses, accuracies, timing.
    """
    from .tracker import MemTracker

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    step = 0
    best_test_acc = 0.0
    tracker_result = None

    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            tracker = MemTracker() if step == tracker_step else None

            if tracker:
                tracker.write('W1', config.hidden * config.n_bits)
                tracker.write('b1', config.hidden)
                tracker.write('W2', config.hidden)
                tracker.write('b2', 1)
                tracker.write('x', config.n_bits)
                tracker.write('y', 1)

            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2, tracker=tracker)
            backward_and_update(x_train[i], y_train[i], out, h_pre, h,
                                W1, b1, W2, b2, config, tracker=tracker)

            if tracker:
                tracker_result = tracker.to_json()

            step += 1

        # Evaluate after each epoch
        tr_outs = forward_batch(x_train, W1, b1, W2, b2)
        te_outs = forward_batch(x_test, W1, b1, W2, b2)
        train_losses.append(hinge_loss(tr_outs, y_train))
        test_losses.append(hinge_loss(te_outs, y_test))
        train_accs.append(accuracy(tr_outs, y_train))
        test_accs.append(accuracy(te_outs, y_test))

        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]

        if best_test_acc >= 1.0:
            break

    elapsed = time.time() - start

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_test_acc': best_test_acc,
        'total_steps': step,
        'elapsed_s': elapsed,
        'tracker': tracker_result,
        'method': 'standard_backprop',
    }
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_train.py -v`

Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/sparse_parity/train.py tests/test_train.py
git commit -m "feat: add standard backprop training with ARD instrumentation"
```

---

### Task 7: Fused layer-wise updates

**Files:**
- Create: `src/sparse_parity/train_fused.py`

**Step 1: Write train_fused.py**

Same as `train.py` but with `backward_and_update` reordered: compute Layer 2 grads → update W2,b2 → compute Layer 1 grads → update W1,b1.

```python
# src/sparse_parity/train_fused.py
"""Phase 4a: Fused layer-wise updates — update each layer immediately after computing its gradients."""

import time

from .model import forward, forward_batch
from .metrics import hinge_loss, accuracy
from .config import Config


def backward_and_update_fused(x, y, out, h_pre, h, W1, b1, W2, b2, config, tracker=None):
    """Fused: grad_layer2 -> update_layer2 -> grad_layer1 -> update_layer1."""
    hidden = len(W1)
    n_bits = len(x)

    if tracker:
        tracker.read('out', 1)
        tracker.read('y', 1)

    margin = out * y
    if margin >= 1.0:
        return

    dout = -y

    if tracker:
        tracker.write('dout', 1)

    # -- Layer 2 backward --
    if tracker:
        tracker.read('dout', 1)
        tracker.read('h', hidden)

    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout

    if tracker:
        tracker.write('dW2', hidden)
        tracker.write('db2', 1)

    # Compute dh BEFORE updating W2
    if tracker:
        tracker.read('W2', hidden)
        tracker.read('dout', 1)

    dh = [W2[0][j] * dout for j in range(hidden)]

    if tracker:
        tracker.write('dh', hidden)

    # -- FUSED: Update W2, b2 immediately --
    if tracker:
        tracker.read('dW2', hidden)
        tracker.read('W2', hidden)

    for j in range(hidden):
        W2[0][j] -= config.lr * (dW2_0[j] + config.wd * W2[0][j])

    if tracker:
        tracker.write('W2', hidden)
        tracker.read('db2', 1)
        tracker.read('b2', 1)

    b2[0] -= config.lr * (db2_0 + config.wd * b2[0])

    if tracker:
        tracker.write('b2', 1)

    # -- ReLU backward --
    if tracker:
        tracker.read('dh', hidden)
        tracker.read('h_pre', hidden)

    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    if tracker:
        tracker.write('dh_pre', hidden)

    # -- FUSED: Layer 1 backward + update --
    if tracker:
        tracker.read('dh_pre', hidden)
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)

    for j in range(hidden):
        for i in range(n_bits):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])

    if tracker:
        tracker.write('W1', hidden * n_bits)
        tracker.read('dh_pre', hidden)
        tracker.read('b1', hidden)

    for j in range(hidden):
        b1[j] -= config.lr * (dh_pre[j] + config.wd * b1[j])

    if tracker:
        tracker.write('b1', hidden)


def train_fused(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=0):
    """Train with fused layer-wise updates."""
    from .tracker import MemTracker

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    step = 0
    best_test_acc = 0.0
    tracker_result = None

    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            tracker = MemTracker() if step == tracker_step else None

            if tracker:
                tracker.write('W1', config.hidden * config.n_bits)
                tracker.write('b1', config.hidden)
                tracker.write('W2', config.hidden)
                tracker.write('b2', 1)
                tracker.write('x', config.n_bits)
                tracker.write('y', 1)

            out, h_pre, h = forward(x_train[i], W1, b1, W2, b2, tracker=tracker)
            backward_and_update_fused(x_train[i], y_train[i], out, h_pre, h,
                                      W1, b1, W2, b2, config, tracker=tracker)

            if tracker:
                tracker_result = tracker.to_json()

            step += 1

        tr_outs = forward_batch(x_train, W1, b1, W2, b2)
        te_outs = forward_batch(x_test, W1, b1, W2, b2)
        train_losses.append(hinge_loss(tr_outs, y_train))
        test_losses.append(hinge_loss(te_outs, y_test))
        train_accs.append(accuracy(tr_outs, y_train))
        test_accs.append(accuracy(te_outs, y_test))

        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]

        if best_test_acc >= 1.0:
            break

    elapsed = time.time() - start

    return {
        'train_losses': train_losses, 'test_losses': test_losses,
        'train_accs': train_accs, 'test_accs': test_accs,
        'best_test_acc': best_test_acc, 'total_steps': step,
        'elapsed_s': elapsed, 'tracker': tracker_result,
        'method': 'fused_layerwise',
    }
```

**Step 2: Verify it imports**

Run: `python3 -c "from sparse_parity.train_fused import train_fused; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/sparse_parity/train_fused.py
git commit -m "feat: add fused layer-wise update training variant"
```

---

### Task 8: Per-layer forward-backward training

**Files:**
- Create: `src/sparse_parity/train_perlayer.py`

**Step 1: Write train_perlayer.py**

The radical variant: each layer does forward → backward → update before the next layer begins.

```python
# src/sparse_parity/train_perlayer.py
"""Phase 4b: Per-layer forward-backward — update each layer before proceeding to next.

WARNING: This changes the math. Layer 2's forward uses already-updated W1/b1.
This means gradients are computed with respect to different parameters than standard backprop.
The goal is to minimize ARD by keeping parameters in cache between use and update.
"""

import time

from .metrics import hinge_loss, accuracy
from .config import Config


def train_step_perlayer(x, y, W1, b1, W2, b2, config, tracker=None):
    """
    Per-layer forward-backward for one sample.

    Layer 1: forward -> backward -> update W1,b1
    Layer 2: forward (with updated W1,b1) -> backward -> update W2,b2
    """
    hidden = config.hidden
    n_bits = config.n_bits

    # === Layer 1 forward ===
    if tracker:
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)
        tracker.read('b1', hidden)

    h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j] for j in range(hidden)]

    if tracker:
        tracker.write('h_pre', hidden)
        tracker.read('h_pre', hidden)

    h = [max(0.0, v) for v in h_pre]

    if tracker:
        tracker.write('h', hidden)

    # === Layer 2 forward ===
    if tracker:
        tracker.read('h', hidden)
        tracker.read('W2', hidden)
        tracker.read('b2', 1)

    out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]

    if tracker:
        tracker.write('out', 1)

    # === Check margin ===
    margin = out * y
    if margin >= 1.0:
        return out

    dout = -y

    # === Layer 2 backward + update ===
    if tracker:
        tracker.read('h', hidden)

    dW2_0 = [dout * h[j] for j in range(hidden)]
    db2_0 = dout

    if tracker:
        tracker.read('W2', hidden)

    dh = [W2[0][j] * dout for j in range(hidden)]

    # Update W2, b2 immediately
    if tracker:
        tracker.read('W2', hidden)

    for j in range(hidden):
        W2[0][j] -= config.lr * (dW2_0[j] + config.wd * W2[0][j])

    if tracker:
        tracker.write('W2', hidden)
        tracker.read('b2', 1)

    b2[0] -= config.lr * (db2_0 + config.wd * b2[0])

    if tracker:
        tracker.write('b2', 1)

    # === Layer 1 backward + update ===
    if tracker:
        tracker.read('h_pre', hidden)

    dh_pre = [dh[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(hidden)]

    if tracker:
        tracker.read('x', n_bits)
        tracker.read('W1', hidden * n_bits)

    for j in range(hidden):
        for i in range(n_bits):
            grad = dh_pre[j] * x[i]
            W1[j][i] -= config.lr * (grad + config.wd * W1[j][i])

    if tracker:
        tracker.write('W1', hidden * n_bits)
        tracker.read('b1', hidden)

    for j in range(hidden):
        b1[j] -= config.lr * (dh_pre[j] + config.wd * b1[j])

    if tracker:
        tracker.write('b1', hidden)

    return out


def forward_batch_perlayer(xs, W1, b1, W2, b2, config):
    """Forward-only batch (no updates) for evaluation."""
    outs = []
    for x in xs:
        hidden = config.hidden
        n_bits = config.n_bits
        h_pre = [sum(W1[j][i] * x[i] for i in range(n_bits)) + b1[j] for j in range(hidden)]
        h = [max(0.0, v) for v in h_pre]
        out = sum(W2[0][j] * h[j] for j in range(hidden)) + b2[0]
        outs.append(out)
    return outs


def train_perlayer(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=0):
    """Train with per-layer forward-backward."""
    from .tracker import MemTracker

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    step = 0
    best_test_acc = 0.0
    tracker_result = None

    start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        for i in range(len(x_train)):
            tracker = MemTracker() if step == tracker_step else None

            if tracker:
                tracker.write('W1', config.hidden * config.n_bits)
                tracker.write('b1', config.hidden)
                tracker.write('W2', config.hidden)
                tracker.write('b2', 1)
                tracker.write('x', config.n_bits)
                tracker.write('y', 1)

            train_step_perlayer(x_train[i], y_train[i], W1, b1, W2, b2, config, tracker=tracker)

            if tracker:
                tracker_result = tracker.to_json()

            step += 1

        tr_outs = forward_batch_perlayer(x_train, W1, b1, W2, b2, config)
        te_outs = forward_batch_perlayer(x_test, W1, b1, W2, b2, config)
        train_losses.append(hinge_loss(tr_outs, y_train))
        test_losses.append(hinge_loss(te_outs, y_test))
        train_accs.append(accuracy(tr_outs, y_train))
        test_accs.append(accuracy(te_outs, y_test))

        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]

        if best_test_acc >= 1.0:
            break

    elapsed = time.time() - start

    return {
        'train_losses': train_losses, 'test_losses': test_losses,
        'train_accs': train_accs, 'test_accs': test_accs,
        'best_test_acc': best_test_acc, 'total_steps': step,
        'elapsed_s': elapsed, 'tracker': tracker_result,
        'method': 'per_layer_fwdbwd',
    }
```

**Step 2: Verify it imports**

Run: `python3 -c "from sparse_parity.train_perlayer import train_perlayer; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/sparse_parity/train_perlayer.py
git commit -m "feat: add per-layer forward-backward training variant"
```

---

### Task 9: Main runner (all phases + output artifacts)

**Files:**
- Create: `src/sparse_parity/run.py`

**Step 1: Write run.py**

```python
# src/sparse_parity/run.py
"""Main runner: execute all phases sequentially, produce JSON + markdown + plots."""

import copy
import json
import time
from pathlib import Path

from .config import Config, SCALE_CONFIG
from .data import generate
from .model import init_params
from .train import train
from .train_fused import train_fused
from .train_perlayer import train_perlayer
from .metrics import save_json, save_markdown, timestamp


RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'


def run_experiment(config, label=''):
    """Run all 3 training variants on same data, return comparison."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {label} (n={config.n_bits}, k={config.k_sparse})")
    print(f"{'='*70}")

    data = generate(config)
    x_train, y_train, x_test, y_test, secret = data
    print(f"  Secret indices: {secret}")
    print(f"  Params: {config.total_params:,}")

    results = {}

    # Phase 2: Standard backprop
    print(f"\n  [Phase 2] Standard backprop...")
    W1, b1, W2, b2 = init_params(config)
    r = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config)
    print(f"    Accuracy: {r['best_test_acc']:.0%} in {r['elapsed_s']:.3f}s")
    print(f"    ARD: {r['tracker']['weighted_ard']:,.0f}" if r['tracker'] else "    ARD: N/A")
    results['standard'] = r

    # Phase 4a: Fused
    print(f"\n  [Phase 4a] Fused layer-wise...")
    W1, b1, W2, b2 = init_params(config)
    r = train_fused(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config)
    print(f"    Accuracy: {r['best_test_acc']:.0%} in {r['elapsed_s']:.3f}s")
    print(f"    ARD: {r['tracker']['weighted_ard']:,.0f}" if r['tracker'] else "    ARD: N/A")
    results['fused'] = r

    # Phase 4b: Per-layer
    print(f"\n  [Phase 4b] Per-layer forward-backward...")
    W1, b1, W2, b2 = init_params(config)
    r = train_perlayer(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config)
    print(f"    Accuracy: {r['best_test_acc']:.0%} in {r['elapsed_s']:.3f}s")
    print(f"    ARD: {r['tracker']['weighted_ard']:,.0f}" if r['tracker'] else "    ARD: N/A")
    results['perlayer'] = r

    return results, secret


def generate_report(all_results, ts):
    """Generate markdown comparison report."""
    lines = [
        f"# Sparse Parity Experiment Results",
        f"",
        f"**Generated**: {ts}",
        f"",
    ]

    for label, (results, secret) in all_results.items():
        lines.append(f"## {label}")
        lines.append(f"")
        lines.append(f"Secret indices: {secret}")
        lines.append(f"")
        lines.append(f"| Method | Best Accuracy | ARD (weighted) | Time |")
        lines.append(f"|--------|--------------|----------------|------|")

        for method, r in results.items():
            acc = f"{r['best_test_acc']:.0%}"
            ard = f"{r['tracker']['weighted_ard']:,.0f}" if r.get('tracker') else "N/A"
            t = f"{r['elapsed_s']:.3f}s"
            lines.append(f"| {method} | {acc} | {ard} | {t} |")

        lines.append(f"")

        # ARD comparison
        if all(r.get('tracker') for r in results.values()):
            std_ard = results['standard']['tracker']['weighted_ard']
            for method in ['fused', 'perlayer']:
                if method in results and results[method].get('tracker'):
                    m_ard = results[method]['tracker']['weighted_ard']
                    pct = (1 - m_ard / std_ard) * 100 if std_ard > 0 else 0
                    lines.append(f"**{method}** ARD improvement over standard: **{pct:.1f}%**")
            lines.append(f"")

    return '\n'.join(lines)


def try_plot(all_results, ts):
    """Generate plots if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [PLOT] matplotlib not available, skipping plots")
        return

    for label, (results, _) in all_results.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curves
        for method, r in results.items():
            axes[0].plot(r['train_losses'], label=f'{method} train', alpha=0.7)
            axes[0].plot(r['test_losses'], label=f'{method} test', linestyle='--', alpha=0.7)
        axes[0].set(xlabel='Epoch', ylabel='Hinge Loss', title=f'{label} - Loss')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        for method, r in results.items():
            axes[1].plot(r['test_accs'], label=method)
        axes[1].set(xlabel='Epoch', ylabel='Test Accuracy', title=f'{label} - Accuracy')
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # ARD comparison bar chart
        methods = []
        ards = []
        for method, r in results.items():
            if r.get('tracker'):
                methods.append(method)
                ards.append(r['tracker']['weighted_ard'])
        if methods:
            bars = axes[2].bar(methods, ards, color=['#2196F3', '#FF9800', '#4CAF50'])
            axes[2].set(ylabel='Weighted ARD (floats)', title=f'{label} - ARD Comparison')
            axes[2].grid(True, alpha=0.3, axis='y')
            for bar, v in zip(bars, ards):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'Sparse Parity: {label}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = RESULTS_DIR / f'{ts}_{label.lower().replace(" ", "_")}_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  [PLOT] Saved: {plot_path.name}")
        plt.close(fig)


def main():
    """Run the full pipeline: 3-bit baseline + 20-bit scaling."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    total_start = time.time()

    all_results = {}

    # Phase 1-4: 3-bit parity
    config_3bit = Config()
    results_3bit, secret_3bit = run_experiment(config_3bit, '3-bit parity')
    all_results['3-bit parity'] = (results_3bit, secret_3bit)

    # Phase 5: Scale to 20-bit
    results_20bit, secret_20bit = run_experiment(SCALE_CONFIG, '20-bit sparse parity')
    all_results['20-bit sparse parity'] = (results_20bit, secret_20bit)

    # Save JSON
    json_path = RESULTS_DIR / f'{ts}_results.json'
    json_data = {}
    for label, (results, secret) in all_results.items():
        json_data[label] = {
            'secret': secret,
            'methods': {m: {k: v for k, v in r.items() if k != 'tracker'}
                        for m, r in results.items()},
            'ard': {m: r['tracker'] for m, r in results.items() if r.get('tracker')},
        }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  [JSON] Saved: {json_path.name}")

    # Save markdown report
    report = generate_report(all_results, ts)
    md_path = RESULTS_DIR / f'{ts}_report.md'
    with open(md_path, 'w') as f:
        f.write(report)
    print(f"  [MD] Saved: {md_path.name}")

    # Generate plots
    try_plot(all_results, ts)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  DONE in {total_elapsed:.2f}s")
    print(f"  Results: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
```

**Step 2: Test the runner end-to-end**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && python3 -m sparse_parity.run`

Expected: Completes in <5s, prints comparison table, creates files in `results/`

**Step 3: Commit**

```bash
git add src/sparse_parity/run.py
git commit -m "feat: add main runner with JSON/markdown/plot output"
```

---

### Task 10: Scaling test

**Files:**
- Create: `tests/test_scaling.py`

**Step 1: Write the scaling test**

```python
# tests/test_scaling.py
"""Verify the pipeline works at 20-bit scale."""

from sparse_parity.config import Config
from sparse_parity.data import generate
from sparse_parity.model import init_params
from sparse_parity.train import train


def test_20bit_converges():
    """20-bit sparse parity (3 relevant + 17 noise) should eventually converge."""
    config = Config(n_bits=20, k_sparse=3, n_train=200, n_test=200,
                    hidden=2000, max_epochs=50, seed=42)
    x_train, y_train, x_test, y_test, secret = generate(config)
    W1, b1, W2, b2 = init_params(config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=-1)
    # 20-bit is harder, accept >60% as a sign of learning
    assert result['best_test_acc'] > 0.6, f"Only reached {result['best_test_acc']:.0%}"


def test_20bit_under_two_seconds():
    """20-bit should run in <2 seconds."""
    config = Config(n_bits=20, k_sparse=3, n_train=200, n_test=200,
                    hidden=2000, max_epochs=5, seed=42)
    x_train, y_train, x_test, y_test, _ = generate(config)
    W1, b1, W2, b2 = init_params(config)
    result = train(x_train, y_train, x_test, y_test, W1, b1, W2, b2, config, tracker_step=-1)
    assert result['elapsed_s'] < 2.0, f"Took {result['elapsed_s']:.2f}s"
```

**Step 2: Run the scaling test**

Run: `python3 -m pytest tests/test_scaling.py -v --timeout=30`

Expected: PASS (may be slow — 20-bit with HIDDEN=2000 is heavy in pure Python)

Note: If `test_20bit_under_two_seconds` fails, reduce `hidden` to 1000 or `max_epochs` to 3 in the test config. Pure Python with 2000-hidden and 200 samples may exceed 2s. Adjust the config or the threshold.

**Step 3: Commit**

```bash
git add tests/test_scaling.py
git commit -m "test: add 20-bit scaling convergence and timing tests"
```

---

### Task 11: Run full pipeline, verify, commit results

**Step 1: Run all tests**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && python3 -m pytest tests/ -v`

Expected: All tests pass

**Step 2: Run the full pipeline**

Run: `cd /Users/yadkonrad/dev_dev/year26/feb26/SutroYaro && python3 -m sparse_parity.run`

Expected: Prints results for both 3-bit and 20-bit, creates files in `results/`

**Step 3: Verify output files exist**

Run: `ls results/*.json results/*.md results/*.png`

Expected: At least one of each

**Step 4: Commit everything**

```bash
git add results/ src/sparse_parity/ tests/
git commit -m "feat: complete sparse parity pipeline — all 5 phases working

Phase 1: Data generation with parity labels
Phase 2: Standard backprop baseline (>90% on 3-bit)
Phase 3: ARD measurement via MemTracker
Phase 4a: Fused layer-wise updates (~16% ARD improvement)
Phase 4b: Per-layer forward-backward (novel, changes math)
Phase 5: Scale to 20-bit (3 relevant + 17 noise bits)

Output: JSON metrics, markdown report, comparison plots"
```

**Step 5: Push**

Run: `git push`
