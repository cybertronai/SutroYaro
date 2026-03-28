# TrackedArray: Auto-instrumented DMD Tracking

## Problem

Manual `tracker.read()` / `tracker.write()` calls are error-prone. The GF(2) experiment reported DMC of 8,607 but the actual Gaussian elimination (row swaps, XOR operations) had zero tracking calls.

Manual instrumentation also creates a barrier for competition submissions. If someone submits a new algorithm, we can't trust them to instrument correctly, and we shouldn't require them to.

## The metric: DMD (Ding et al., arXiv:2312.14441)

Data Movement Distance (DMD) measures the cost of each memory access using an LRU stack model (Definition 2.1 in the paper).

Every float lives in an LRU stack ordered by recency of writes. When a float is read, its **stack distance** is its 1-indexed position in the stack. Its DMD is `sqrt(stack_distance)`. The total DMD of an algorithm is the sum of all read DMDs.

Key rules:
- **Writes** move the element to the top of the stack. Writes are free (no DMD cost). Inputs arrive pre-loaded on the stack.
- **Reads** observe the element's position but do not move it.
- **DMD** = sum of sqrt(stack_distance) across all reads.

From the paper: "in `abbbca`, the reuse distance of the second `a` is 3. Its DMD is sqrt(3)."

### Worked example: `(a+b)+a`

Arrays of size 1: `a=[1.0]`, `b=[5.0]`. Each array is a single float.

```
Step 1: write a       stack = [a]             free
Step 2: write b       stack = [a, b]          free

Step 3: compute a + b
  read a:  a is at position 2    dist = 2    stack unchanged [a, b]
  read b:  b is at position 1    dist = 1    stack unchanged [a, b]
  write c: c goes to top         stack = [a, b, c]

Step 4: compute c + a
  read c:  c is at position 1    dist = 1    stack unchanged [a, b, c]
  read a:  a is at position 3    dist = 3    stack unchanged [a, b, c]
  write d: d goes to top         stack = [a, b, c, d]
```

| Read | Stack distance | DMD contribution |
|------|---------------|-----------------|
| a (in a+b) | 2 | sqrt(2) = 1.414 |
| b (in a+b) | 1 | sqrt(1) = 1.000 |
| c (in c+a) | 1 | sqrt(1) = 1.000 |
| a (in c+a) | 3 | sqrt(3) = 1.732 |

**DMD = 1.414 + 1.000 + 1.000 + 1.732 = 5.146**

The second read of `a` has distance 3 because there are only 3 elements in the stack.

## Solution

Two components:

### LRUStackTracker

`src/sparse_parity/lru_tracker.py` -- per-element LRU stack tracker. Each float is identified by `(buffer_name, index)`. Writes push elements to top (free). Reads observe stack positions without modification and accumulate DMD.

### TrackedArray (ndarray subclass)

`src/sparse_parity/tracked_numpy.py` -- wraps `np.ndarray` to automatically call `tracker.write()` and `tracker.read()` on every operation.

Operations are intercepted at three levels:

1. **`__array_ufunc__`** -- catches all ufuncs: `+`, `-`, `*`, `^`, `==`, `<`, etc. For each TrackedArray input, records a `tracker.read()`. For the output, creates a new TrackedArray and records a `tracker.write()`.

2. **`__array_function__`** -- catches numpy functions: `np.where`, `np.prod`, `np.sum`, `np.all`, `np.sort`, `np.concatenate`, etc. A default handler covers any unregistered function.

3. **`__getitem__` / `__setitem__`** -- catches indexing, slicing, and fancy indexing. `__getitem__` records a read sized to the actual slice (not the whole array).

### tracking_context (context manager)

`np.zeros(shape)` has no array arguments, so `__array_function__` never fires. Inside a `tracking_context`, constructors (`np.zeros`, `np.ones`, `np.empty`) are monkey-patched to return TrackedArrays. Patches revert on exit.

### Propagation

Any array derived from a TrackedArray is itself a TrackedArray on the same tracker. Wrapping initial inputs is sufficient.

## Usage

```python
from sparse_parity.tracked_numpy import TrackedArray, tracking_context
from sparse_parity.lru_tracker import LRUStackTracker

tracker = LRUStackTracker()
with tracking_context(tracker):
    A = TrackedArray(A_raw, "A", tracker)
    b = TrackedArray(b_raw, "b", tracker)
    solution, rank = gf2_gauss_elim(A, b)

print(tracker.summary()["dmd"])
```

Zero changes inside `gf2_gauss_elim`.

## GF(2) results

| Method | DMD | Notes |
|--------|-----|-------|
| Manual harness (I/O only) | 8,607 | Only tracks data conversion, misses elimination |
| Yad's honest estimate | 189,056 | Manual count of row operations |
| LRUStackTracker auto | ~203,000 | Per-element LRU stack, all ops tracked |

## Limitations

- **Overhead**: LRUStackTracker is O(n) per element access where n = stack size. GF(2) with n=20 takes ~24s. For measurement, not production.
- **Pure-python loops**: Scalar access like `aug[row, col] == 1` generates per-element tracking events.
- **Non-numpy code**: Values extracted as plain python (`int(arr[0])`) exit the tracking world.
- **Constructor coverage**: Only `np.zeros`, `np.ones`, `np.empty` are patched inside `tracking_context`.

## Files

- `src/sparse_parity/lru_tracker.py` -- LRUStackTracker
- `src/sparse_parity/tracked_numpy.py` -- TrackedArray + tracking_context
- `tests/test_tracked_numpy.py` -- 30 tests organized by concern
