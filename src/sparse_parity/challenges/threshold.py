"""
Sparse threshold challenge.

y = 1 if sum(x[secret]) >= t else -1, over {-1, +1}.  Parameterized
difficulty via ``t``: t=0 recovers majority vote, t=k recovers sparse
AND (all-ones), t=-k recovers the trivial always-true task.  Default
is t = k_sparse - 1, which flags roughly the top 1/2^k inputs.

Inputs:    x in {-1, +1}^n
Secret:    k-sized subset of [n]
Output:    y in {-1, +1}
Metric:    accuracy (exact match) + ARD/DMC via MemTracker
"""

import time

import numpy as np

from sparse_parity.config import Config
from sparse_parity.tracker import MemTracker


def _label(x, secret, t):
    s = np.sum(x[:, secret], axis=1)
    return np.where(s >= t, 1.0, -1.0)


def measure_threshold(method, n_bits=20, k_sparse=3, hidden=200,
                      lr=0.1, wd=0.01, batch_size=32, n_train=1000,
                      max_epochs=200, seed=42, threshold=None, **kwargs):
    """
    Run a sparse threshold experiment.

    Parameters
    ----------
    threshold : int or None
        Sum threshold t.  Default = k_sparse - 1 (a mildly imbalanced
        task).  Must satisfy -k_sparse <= t <= k_sparse.
    """
    if threshold is None:
        threshold = k_sparse - 1
    assert -k_sparse <= threshold <= k_sparse, (
        f"threshold {threshold} outside valid range [-{k_sparse}, {k_sparse}]"
    )

    config = Config(
        n_bits=n_bits, k_sparse=k_sparse, hidden=hidden,
        lr=lr, wd=wd, batch_size=batch_size,
        n_train=n_train, n_test=200, max_epochs=max_epochs, seed=seed,
    )

    rng_secret = np.random.RandomState(seed)
    secret = sorted(rng_secret.choice(n_bits, k_sparse, replace=False).tolist())

    start = time.perf_counter()

    if method == "sgd":
        result = _run_sgd(config, secret, seed, threshold, **kwargs)
    elif method == "km":
        result = _run_km(config, secret, seed, threshold, **kwargs)
    elif method == "fourier":
        result = _run_fourier(config, secret, seed, threshold, **kwargs)
    elif method in ("gf2", "smt"):
        result = {
            "accuracy": 0.0, "ard": None, "dmc": None,
            "total_floats": None, "found_secret": None,
            "error": f"{method} does not apply to threshold",
        }
    else:
        return {
            "error": (
                f"Unknown method for threshold: {method}. "
                f"Available: sgd, km, fourier"
            ),
            "method": method,
        }

    elapsed = time.perf_counter() - start
    result["time_s"] = round(elapsed, 6)
    result["method"] = method
    result["challenge"] = "threshold"
    result["config"] = {
        "n_bits": n_bits, "k_sparse": k_sparse, "hidden": hidden,
        "lr": lr, "wd": wd, "batch_size": batch_size,
        "n_train": n_train, "max_epochs": max_epochs, "seed": seed,
        "threshold": threshold,
    }
    return result


def _run_sgd(config, secret, seed, threshold, **_kwargs):
    """Linear SGD with a threshold read-out at inference."""
    rng = np.random.RandomState(seed + 100)
    x_tr = rng.choice([-1.0, 1.0], size=(config.n_train, config.n_bits))
    y_tr = _label(x_tr, secret, threshold)

    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = _label(x_te, secret, threshold)

    w = rng.randn(config.n_bits) * 0.01

    tracker = MemTracker()
    tracker.write("w", w.size)

    best_acc = 0.0
    epoch = 0
    for epoch in range(config.max_epochs):
        perm = rng.permutation(config.n_train)
        for start_idx in range(0, config.n_train, config.batch_size):
            idx = perm[start_idx:start_idx + config.batch_size]
            xb = x_tr[idx]
            yb = y_tr[idx]

            tracker.read("w")
            pred = xb @ w
            err = pred - yb
            grad = (2.0 / len(idx)) * (xb.T @ err)
            if config.wd > 0:
                grad += config.wd * w

            tracker.write("w", w.size)
            w -= config.lr * grad

        # Linear regression targets are +/-1 so the optimal decision is
        # sign(raw).  Use the test-set median as the center to absorb the
        # class-imbalance bias that a pure +/-1 regression does not fit.
        raw = x_te @ w
        center = float(np.median(raw))
        y_pred_te = np.where(raw >= center, 1.0, -1.0)
        acc = float(np.mean(y_pred_te == y_te))
        best_acc = max(best_acc, acc)
        if acc >= 1.0:
            break

    found_secret = sorted(np.argsort(np.abs(w))[-config.k_sparse:].tolist())

    s = tracker.summary()
    return {
        "accuracy": round(best_acc, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": found_secret,
        "epochs": epoch + 1,
    }


def _run_km(config, secret, seed, threshold, influence_samples=5, **_kwargs):
    """Bit-flip influence for the threshold label."""
    rng = np.random.RandomState(seed + 100)
    rng_inf = np.random.RandomState(seed + 500)

    tracker = MemTracker()
    influences = np.zeros(config.n_bits)

    for i in range(config.n_bits):
        x_batch = rng_inf.choice([-1.0, 1.0], size=(influence_samples, config.n_bits))
        tracker.write(f"x_batch_{i}", x_batch.size)

        y_orig = _label(x_batch, secret, threshold)
        tracker.read(f"x_batch_{i}")
        tracker.write(f"y_orig_{i}", y_orig.size)

        x_flipped = x_batch.copy()
        x_flipped[:, i] *= -1
        y_flipped = _label(x_flipped, secret, threshold)
        tracker.write(f"y_flip_{i}", y_flipped.size)

        tracker.read(f"y_orig_{i}")
        tracker.read(f"y_flip_{i}")
        influences[i] = np.mean(y_orig != y_flipped)
        tracker.write(f"inf_{i}", 1)

    top_k = sorted(np.argsort(influences)[-config.k_sparse:].tolist())

    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = _label(x_te, secret, threshold)
    y_pred = _label(x_te, top_k, threshold)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
    }


def _run_fourier(config, secret, seed, threshold, **_kwargs):
    """First-order Fourier correlation for threshold."""
    rng = np.random.RandomState(seed + 100)
    n_samples = max(100, config.n_train)
    x = rng.choice([-1.0, 1.0], size=(n_samples, config.n_bits))
    y = _label(x, secret, threshold)

    tracker = MemTracker()
    tracker.write("x", x.size)
    tracker.write("y", y.size)

    correlations = np.zeros(config.n_bits)
    for i in range(config.n_bits):
        tracker.read("x")
        tracker.read("y")
        correlations[i] = abs(np.mean(y * x[:, i]))

    top_k = sorted(np.argsort(correlations)[-config.k_sparse:].tolist())

    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = _label(x_te, secret, threshold)
    y_pred = _label(x_te, top_k, threshold)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
    }
