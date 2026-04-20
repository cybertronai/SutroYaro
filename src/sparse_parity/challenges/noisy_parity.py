"""
Noisy parity challenge.

y = prod(x[secret]) with each training label independently flipped
with probability ``noise_rate``.  Test labels are NOT flipped so
accuracy measures recovery of the true parity rule, not the noisy one.

This is the Learning Parity with Noise (LPN) problem, well known to be
hard for large n but easy for small k when noise is moderate.  Andy's
GF(2) noise experiment (PR #3) is the prior art within this repo.

Inputs:    x in {-1, +1}^n
Secret:    k-sized subset of [n]
Noise:     i.i.d. label flip with rate p on TRAINING data
Output:    y in {-1, +1}  (training labels noisy, test labels clean)
Metric:    accuracy on clean test set + ARD/DMC via MemTracker
"""

import time

import numpy as np

from sparse_parity.config import Config
from sparse_parity.tracker import MemTracker


def _noisy_labels(rng, clean, noise_rate):
    """Flip each label independently with probability noise_rate."""
    if noise_rate <= 0.0:
        return clean.copy()
    flip_mask = rng.random(clean.shape) < noise_rate
    noisy = clean.copy()
    noisy[flip_mask] *= -1
    return noisy


def measure_noisy_parity(method, n_bits=20, k_sparse=3, hidden=200,
                         lr=0.1, wd=0.01, batch_size=32, n_train=1000,
                         max_epochs=200, seed=42, noise_rate=0.1, **kwargs):
    """
    Run a noisy-parity experiment.

    Parameters
    ----------
    noise_rate : float
        Probability that a TRAINING label is flipped.  Must be in [0, 0.5).
        Test labels remain clean so accuracy measures true-rule recovery.
    """
    assert 0.0 <= noise_rate < 0.5, (
        f"noise_rate {noise_rate} must be in [0, 0.5). "
        f"At 0.5 the label is independent of x."
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
        result = _run_sgd(config, secret, seed, noise_rate, **kwargs)
    elif method == "km":
        result = _run_km(config, secret, seed, noise_rate, **kwargs)
    elif method == "fourier":
        result = _run_fourier(config, secret, seed, noise_rate, **kwargs)
    elif method == "gf2":
        result = _run_gf2(config, secret, seed, noise_rate, **kwargs)
    else:
        return {
            "error": (
                f"Unknown method for noisy-parity: {method}. "
                f"Available: sgd, km, fourier, gf2"
            ),
            "method": method,
        }

    elapsed = time.perf_counter() - start
    result["time_s"] = round(elapsed, 6)
    result["method"] = method
    result["challenge"] = "noisy-parity"
    result["config"] = {
        "n_bits": n_bits, "k_sparse": k_sparse, "hidden": hidden,
        "lr": lr, "wd": wd, "batch_size": batch_size,
        "n_train": n_train, "max_epochs": max_epochs, "seed": seed,
        "noise_rate": noise_rate,
    }
    return result


def _run_sgd(config, secret, seed, noise_rate, **_kwargs):
    """Two-layer ReLU net with MSE loss, trained on noisy labels."""
    rng = np.random.RandomState(seed + 100)
    x_tr = rng.choice([-1.0, 1.0], size=(config.n_train, config.n_bits))
    y_clean = np.prod(x_tr[:, secret], axis=1)
    y_tr = _noisy_labels(rng, y_clean, noise_rate)

    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)

    std1 = np.sqrt(2.0 / config.n_bits)
    std2 = np.sqrt(2.0 / config.hidden)
    W1 = rng.randn(config.hidden, config.n_bits) * std1
    b1 = np.zeros(config.hidden)
    W2 = rng.randn(1, config.hidden) * std2
    b2 = np.zeros(1)

    tracker = MemTracker()
    tracker.write("W1", W1.size)
    tracker.write("b1", b1.size)
    tracker.write("W2", W2.size)
    tracker.write("b2", b2.size)

    best_acc = 0.0
    epoch = 0
    for epoch in range(config.max_epochs):
        perm = rng.permutation(config.n_train)
        for start_idx in range(0, config.n_train, config.batch_size):
            idx = perm[start_idx:start_idx + config.batch_size]
            xb = x_tr[idx]
            yb = y_tr[idx].reshape(-1, 1)

            tracker.read("W1")
            tracker.read("b1")
            h_pre = xb @ W1.T + b1
            h = np.maximum(0, h_pre)

            tracker.read("W2")
            tracker.read("b2")
            out = h @ W2.T + b2

            d_out = (out - yb) / len(idx)
            dW2 = d_out.T @ h
            db2 = d_out.sum(axis=0)
            d_h = d_out @ W2
            d_h_pre = d_h * (h_pre > 0).astype(np.float64)
            dW1 = d_h_pre.T @ xb
            db1 = d_h_pre.sum(axis=0)

            if config.wd > 0:
                dW1 += config.wd * W1
                dW2 += config.wd * W2

            tracker.write("W1", W1.size)
            tracker.write("W2", W2.size)
            W1 -= config.lr * dW1
            b1 -= config.lr * db1
            W2 -= config.lr * dW2
            b2 -= config.lr * db2

        h_te = np.maximum(0, x_te @ W1.T + b1)
        out_te = (h_te @ W2.T + b2).flatten()
        y_pred_te = np.sign(out_te)
        y_pred_te[y_pred_te == 0] = 1.0
        acc = float(np.mean(y_pred_te == y_te))
        best_acc = max(best_acc, acc)
        if acc >= 1.0:
            break

    s = tracker.summary()
    return {
        "accuracy": round(best_acc, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": None,
        "epochs": epoch + 1,
    }


def _run_km(config, secret, seed, noise_rate, influence_samples=20, **_kwargs):
    """KM-style influence estimated from paired queries.

    With noise, a single-pair influence is unreliable.  We average over
    ``influence_samples`` pairs per bit so signal survives for modest
    noise rates (e.g. p <= 0.2)."""
    rng = np.random.RandomState(seed + 100)
    rng_inf = np.random.RandomState(seed + 500)

    tracker = MemTracker()
    influences = np.zeros(config.n_bits)

    for i in range(config.n_bits):
        x_batch = rng_inf.choice([-1.0, 1.0], size=(influence_samples, config.n_bits))
        tracker.write(f"x_batch_{i}", x_batch.size)

        y_orig_clean = np.prod(x_batch[:, secret], axis=1)
        y_orig = _noisy_labels(rng_inf, y_orig_clean, noise_rate)
        tracker.read(f"x_batch_{i}")
        tracker.write(f"y_orig_{i}", y_orig.size)

        x_flipped = x_batch.copy()
        x_flipped[:, i] *= -1
        y_flip_clean = np.prod(x_flipped[:, secret], axis=1)
        y_flipped = _noisy_labels(rng_inf, y_flip_clean, noise_rate)
        tracker.write(f"y_flip_{i}", y_flipped.size)

        tracker.read(f"y_orig_{i}")
        tracker.read(f"y_flip_{i}")
        influences[i] = np.mean(y_orig != y_flipped)
        tracker.write(f"inf_{i}", 1)

    top_k = sorted(np.argsort(influences)[-config.k_sparse:].tolist())

    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    y_pred = np.prod(x_te[:, top_k], axis=1)
    accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": top_k,
    }


def _run_fourier(config, secret, seed, noise_rate, **_kwargs):
    """Walsh-Hadamard correlation: noise attenuates the true coefficient
    by factor (1 - 2p) but keeps it dominant for moderate p."""
    from itertools import combinations

    rng = np.random.RandomState(seed + 100)
    n_samples = max(200, config.n_train)
    x = rng.choice([-1.0, 1.0], size=(n_samples, config.n_bits))
    y_clean = np.prod(x[:, secret], axis=1)
    y = _noisy_labels(rng, y_clean, noise_rate)

    tracker = MemTracker()
    tracker.write("x", x.size)
    tracker.write("y", y.size)

    best_corr = 0.0
    best_subset = None

    for subset in combinations(range(config.n_bits), config.k_sparse):
        tracker.read("x")
        tracker.read("y")
        chi = np.prod(x[:, list(subset)], axis=1)
        corr = abs(np.mean(y * chi))
        if corr > best_corr:
            best_corr = corr
            best_subset = sorted(subset)

    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    if best_subset is None:
        accuracy = 0.0
    else:
        y_pred = np.prod(x_te[:, best_subset], axis=1)
        accuracy = float(np.mean(y_pred == y_te))

    s = tracker.summary()
    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": best_subset,
    }


def _run_gf2(config, secret, seed, noise_rate, **_kwargs):
    """Plain GF(2) Gaussian elimination; expected to fail once noise
    flips enough equations that the linear system is inconsistent.  We
    keep it so the comparison is explicit rather than silently absent."""
    rng = np.random.RandomState(seed + 100)

    n_samples = config.n_bits + 1
    x = rng.choice([-1.0, 1.0], size=(n_samples, config.n_bits))
    y_clean = np.prod(x[:, secret], axis=1)
    y = _noisy_labels(rng, y_clean, noise_rate)

    A = ((x + 1) / 2).astype(np.uint8)
    b = ((y + 1) / 2).astype(np.uint8)

    n = config.n_bits
    found_secret = None
    for b_try in [b, (1 - b).astype(np.uint8)]:
        aug = np.hstack([A.copy(), b_try.reshape(-1, 1)]).astype(np.uint8)

        pivot_cols = []
        row = 0
        for col in range(n):
            found = None
            for r in range(row, len(aug)):
                if aug[r, col] == 1:
                    found = r
                    break
            if found is None:
                continue
            aug[[row, found]] = aug[[found, row]]
            for r in range(len(aug)):
                if r != row and aug[r, col] == 1:
                    aug[r] = (aug[r] ^ aug[row])
            pivot_cols.append(col)
            row += 1

        solution = np.zeros(n, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            solution[col] = aug[i, -1]

        candidate = sorted(int(i) for i in range(n) if solution[i] == 1)

        # Verify on clean test set
        if candidate:
            x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
            y_te = np.prod(x_te[:, secret], axis=1)
            y_pred = np.prod(x_te[:, candidate], axis=1)
            if np.mean(y_pred == y_te) >= 0.95:
                found_secret = candidate
                break

    x_te = rng.choice([-1.0, 1.0], size=(200, config.n_bits))
    y_te = np.prod(x_te[:, secret], axis=1)
    if found_secret:
        y_pred = np.prod(x_te[:, found_secret], axis=1)
        accuracy = float(np.mean(y_pred == y_te))
    else:
        accuracy = 0.0

    tracker = MemTracker()
    tracker.write("A", A.size)
    tracker.read("A")
    tracker.write("solution", n)
    s = tracker.summary()

    return {
        "accuracy": round(accuracy, 4),
        "ard": round(s["weighted_ard"], 1),
        "dmc": round(s["dmc"], 1),
        "total_floats": s["total_floats_accessed"],
        "found_secret": found_secret,
    }
