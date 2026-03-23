# Eval Environment Guide

Machine-readable guide for coding agents (Claude Code, Codex, Gemini CLI).

## Quick test

```bash
PYTHONPATH=src python3 -c "
import gymnasium as gym
import sparse_parity.eval
env = gym.make('SutroYaro/SparseParity-v0', metric='dmc', budget=10)
obs, info = env.reset()
print('Methods:', info['methods'])
obs, r, _, _, info = env.step(5)  # try gf2
print(f'{info[\"method\"]}: DMC={info[\"dmc\"]}, reward={r:.2f}')
env.render()
"
```

Prerequisites: `pip install gymnasium numpy`

## What this environment tests

An AI agent picks methods to solve a learning problem and observes energy metrics. The goal is to find the lowest-cost method within a fixed experiment budget.

- 3 challenges: `sparse-parity`, `sparse-sum`, `sparse-and`
- 16 methods (5 implemented in harness, 11 return failure)
- Metrics: ARD (average reuse distance) and DMC (data movement complexity)
- 36 experiments as ground truth, 49-point discovery grading
- Episode = research trajectory (5-30 steps), not a game

## Action space

`Discrete(16)` -- each integer maps to a method:

| Index | Method | Category | Implemented |
|-------|--------|----------|-------------|
| 0 | `sgd` | neural_net | yes |
| 1 | `perlayer` | neural_net | no |
| 2 | `sign_sgd` | neural_net | no |
| 3 | `curriculum` | neural_net | no |
| 4 | `forward_forward` | neural_net | no |
| 5 | `gf2` | algebraic | yes |
| 6 | `km` | algebraic | yes |
| 7 | `smt` | algebraic | yes |
| 8 | `fourier` | algebraic | yes |
| 9 | `lasso` | information_theoretic | no |
| 10 | `mdl` | information_theoretic | no |
| 11 | `mutual_info` | information_theoretic | no |
| 12 | `random_proj` | information_theoretic | no |
| 13 | `rl` | alternative | no |
| 14 | `genetic_prog` | alternative | no |
| 15 | `evolutionary` | alternative | no |

## Constructor parameters

```python
env = gym.make("SutroYaro/SparseParity-v0",
    challenge="sparse-parity",  # "sparse-parity" | "sparse-sum" | "sparse-and"
    n_bits=20,                  # int, 3..100
    k_sparse=3,                 # int, 3..10
    metric="dmc",               # "ard" | "dmc"
    budget=20,                  # int, max steps per episode
    seed=42,                    # int
    harness_timeout=10.0,       # float, max seconds per method call
)
```

Multi-challenge variant:

```python
env = gym.make("SutroYaro/MultiChallenge-v0",
    budget_per=10,              # budget per challenge
    n_bits=20, k_sparse=3, metric="dmc",
)
```

## Running the full evaluation

```bash
PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py
```

Runs 3 baseline agents (Random, Greedy, Oracle) x 5 episodes in ~4 seconds. Outputs to `results/eval/baselines.json` and `results/eval/multi_challenge.json`.

## How to add a new method

1. Register it before creating the environment:

```python
from sparse_parity.eval.registry import register_method

register_method(
    "my_method",
    category="algebraic",           # "neural_net" | "algebraic" | "information_theoretic" | "alternative"
    applicable_challenges=["sparse-parity"],  # None = all challenges
    description="What this method does",
)
```

2. Method registration order determines action index. Default methods are 0-15. New methods get 16, 17, etc.

3. Add answer key entry to `src/sparse_parity/eval/answer_key.json`:

```json
{
    "exp_id": "my-exp1",
    "method": "my_method",
    "challenge": "sparse-parity",
    "accuracy": 1.0,
    "ard": 500.0,
    "dmc": 1200.0,
    "category": "algebraic",
    "result": "SOLVED"
}
```

4. Re-run baselines: `PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py`

## How to add a new challenge

See `docs/research/adding-an-eval-challenge.md` for the full guide. Summary:

1. Write `measure_my_challenge(method, n_bits, k_sparse, seed, **kwargs) -> dict`
2. Register: `register_challenge("my-challenge", harness_fn=my_fn, description="...")`
3. Register methods for it
4. Add answer key entries
5. Run baselines

## Compute backends

| Backend | How to use | Notes |
|---------|-----------|-------|
| Local (default) | `gym.make(...)` | Direct Python import, runs harness locally |
| Modal | Prototype only | Requires `pip install modal`, MODAL_TOKEN_ID/SECRET env vars |
| Remote | Prototype only | HTTP POST to hosted harness endpoint |

Backend selection is handled by `sparse_parity.eval.backends.get_backend(name)`.

## Discovery grading categories

The `DiscoveryGrader` scores research quality beyond raw metric improvement:

| Category | Points | What it measures |
|----------|--------|-----------------|
| Discovered algebraic solver | 10 | Found GF2, KM, or SMT and solved with it |
| Identified local learning failure | 5 | Tried forward_forward, observed it fails |
| Found metric disagreement | 5 | Solved with both KM (ARD winner) and GF2 (DMC winner) |
| Optimized beyond baseline | 3 | Beat SGD baseline DMC of 1,278,460 |
| Correct failure classification | 16 | Observed failures and moved on (2 pts each) |
| Efficiency | 5 | Found best method in fewer steps (5 pts if steps 1-3) |
| Exploration breadth | 5 | Number of unique successful methods (1 pt each, max 5) |

**Total: 49 points.**

Usage:

```python
from sparse_parity.eval.grader import DiscoveryGrader

grader = DiscoveryGrader()
report = grader.grade(env.experiment_log, challenge="sparse-parity")
print(report)
print(f"Score: {report.total_score}/{report.max_possible} ({report.percentage:.0f}%)")
```

## Reward function

```
accuracy < 0.95         -> -0.1  (method failed)
first successful solve  -> 10 / (1 + log10(max(score, 1)))
improved best score     -> 10 * (previous_best - score) / previous_best
no improvement          -> -0.01
```

## Ground truth (sparse-parity, n=20, k=3)

| Method | DMC | ARD | Time |
|--------|-----|-----|------|
| KM-min (1 sample) | 3,578 | 20 | ~0.001s |
| GF2 | 8,607 | ~420 | 509 us |
| KM (5 samples) | 20,633 | 92 | 0.001-0.006s |
| SMT | 348,336 | 3,360 | 0.002s |
| SGD | 1,278,460 | 8,504 | 0.12s |
| Fourier | 78,140,662,852 | -- | -- |

## Files

| File | Purpose |
|------|---------|
| `src/sparse_parity/eval/__init__.py` | Gymnasium registration, imports defaults |
| `src/sparse_parity/eval/env.py` | `SutroYaroEnv` and `MultiChallengeEnv` implementation |
| `src/sparse_parity/eval/registry.py` | `register_challenge()`, `register_method()` |
| `src/sparse_parity/eval/default_registry.py` | Ships 3 challenges, 16 methods |
| `src/sparse_parity/eval/backends.py` | Local, Modal, Remote compute backends |
| `src/sparse_parity/eval/baselines.py` | Random, Greedy, Oracle agents |
| `src/sparse_parity/eval/grader.py` | Discovery scoring (49-point rubric) |
| `src/sparse_parity/eval/answer_key.json` | Ground truth: 36 experiments, 12 negative results |
| `src/sparse_parity/eval/run_eval.py` | Evaluation script (3 agents x 5 episodes) |
| `src/sparse_parity/eval/README.md` | Full interface specification |
| `docs/research/eval-environment.md` | Human-readable docs page |
| `docs/research/adding-an-eval-challenge.md` | Guide for adding challenges |
