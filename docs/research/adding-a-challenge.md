# How to Add a New Challenge

Step-by-step recipe for adding a new learning task to the harness. Follow this in order. An agent or contributor should be able to add a challenge in one session.

## 1. Define the task

Write down:

- **Name**: short slug, e.g. `sparse-sum`, `majority-vote`, `nanogpt`
- **Inputs**: what x looks like (e.g. {-1, +1}^n)
- **Outputs**: what y looks like (e.g. integer in [-k, k])
- **Secret**: what the agent is trying to find (e.g. which k bits)
- **Success metric**: how you measure whether the agent found it (e.g. exact match, MSE, accuracy)

## 2. Add data generation to the harness

Edit `src/harness.py`. Add a `measure_{challenge_slug}()` function. Use `measure_sparse_parity()` as the template.

The data generation pattern is always:

```python
def measure_sparse_sum(method, n_bits=20, k_sparse=3, **kwargs):
    # 1. Pick secret indices (deterministic from seed)
    rng = np.random.RandomState(seed)
    secret = sorted(rng.choice(n_bits, k_sparse, replace=False).tolist())

    # 2. Generate training data
    x = rng.choice([-1.0, 1.0], size=(n_samples, n_bits))
    y = your_task_function(x, secret)  # <-- this is the only line that changes

    # 3. Run the method
    # 4. Measure accuracy + ARD/DMC
    # 5. Return standardized result dict
```

For sparse sum: `y = np.sum(x[:, secret], axis=1)` instead of `y = np.prod(x[:, secret], axis=1)`.

Add the challenge to the CLI dispatcher at the bottom of `harness.py`:

```python
if args.challenge == "sparse-sum":
    result = measure_sparse_sum(method=args.method, ...)
else:
    result = measure_sparse_parity(method=args.method, ...)
```

## 3. Add methods

Not every method works on every challenge. Start with 2: one baseline (SGD) and one alternative. Add more as you test them.

Each method is a `_run_{method}` function that takes a config and returns a dict with `accuracy`, `ard`, `dmc`, `total_floats`. Copy an existing one and change the data generation line.

If a method is expected to fail on the new challenge (e.g. GF(2) on sparse sum), implement it anyway and let the experiment prove it fails. That's a valid finding.

## 4. Add to search_space.yaml

Add a new challenge section to `research/search_space.yaml`:

```yaml
# --- Sparse Sum ---

# challenge: sparse-sum
# version: 1
#
# methods:
#   - sgd
#   - km
#   - fourier
#
# parameters:
#   n_bits: [3, 10, 20, 30, 50]
#   k_sparse: [3, 5, 7]
#   ... (same grid as parity, or customized)
#
# metrics:
#   primary: ard
#   secondary: [dmc, time_s, accuracy, total_floats]
#   locked_in: src/harness.py
```

Uncomment when the challenge is ready for agent use.

## 5. Add to questions.yaml

Add initial research questions for the new challenge:

```yaml
- id: S1
  challenge: sparse-sum
  question: "Does SGD solve sparse sum faster than sparse parity?"
  status: open
  depends_on: []

- id: S2
  challenge: sparse-sum
  question: "Do local learning rules succeed on sparse sum?"
  status: open
  depends_on: [S1]
```

## 6. Run baselines

```bash
PYTHONPATH=src python3 src/harness.py --challenge sparse-sum --method sgd --json
PYTHONPATH=src python3 src/harness.py --challenge sparse-sum --method km --json
```

Record the baseline numbers in DISCOVERIES.md and add them to `checks/baseline_check.py`.

## 7. Update DISCOVERIES.md

Add a section for the new challenge:

```markdown
## Challenge 2: Sparse Sum

y = sum of x[secret_indices]. Regression task (output in [-k, k]).
Unlike parity, each bit contributes independently (first-order signal).

### Baselines

| Method | Accuracy | ARD | Time |
|--------|----------|-----|------|
| SGD    | ...      | ... | ...  |

### Open Questions

1. Do local learning rules work on sum? (they fail on parity)
2. ...
```

## 8. Update TODO.md

Add hypotheses as unchecked items:

```markdown
## Sparse Sum

- [ ] SGD baseline on sparse sum (n=20, k=3)
- [ ] Test Hebbian on sparse sum (expect success, unlike parity)
- [ ] Compare ARD of SGD on sum vs parity (same config)
```

## 9. Test it

```bash
# Should work
PYTHONPATH=src python3 src/harness.py --challenge sparse-sum --method sgd --n_bits 20 --k_sparse 3

# Should fail gracefully (expected)
PYTHONPATH=src python3 src/harness.py --challenge sparse-sum --method gf2 --n_bits 20 --k_sparse 3

# Backward compat: parity still works without --challenge flag
PYTHONPATH=src python3 src/harness.py --method sgd --n_bits 20 --k_sparse 3
```

## Checklist

- [ ] Data generation function in harness.py
- [ ] At least 2 methods implemented
- [ ] `--challenge` flag works in CLI
- [ ] Backward compat (no flag = sparse parity)
- [ ] search_space.yaml section
- [ ] questions.yaml entries
- [ ] DISCOVERIES.md section with baselines
- [ ] TODO.md hypotheses
- [ ] Baselines recorded
