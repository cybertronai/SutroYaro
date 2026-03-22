# Research Eval Environment

Can an AI agent do energy-efficient ML research? This Gymnasium-compatible environment tests that.

## Quick start

```bash
git clone https://github.com/cybertronai/SutroYaro
cd SutroYaro
pip install gymnasium numpy
PYTHONPATH=src python3 -c "
import gymnasium as gym
import sparse_parity.eval.env

env = gym.make('SutroYaro/SparseParity-v0', metric='dmc', budget=16)
obs, info = env.reset()

# Try GF(2) Gaussian elimination
obs, reward, done, trunc, info = env.step(5)
print(f'Method: {info[\"method\"]}, DMC: {info[\"dmc\"]}, Reward: {reward:.2f}')
env.render()
"
```

## What the agent does

Each step, the agent picks a method (discrete action, 16 options). The environment runs the method on sparse parity using the real evaluation harness and returns the energy metric (ARD or DMC). The agent's goal: find the lowest-cost method within a fixed budget of experiments.

This is a research trajectory, not a game. 5-30 steps per episode, not hundreds. The optimal policy is sequential experiment design.

## Two environments

| Environment | Registration | What it tests |
|-------------|-------------|---------------|
| `SutroYaro/SparseParity-v0` | Single challenge | Find best method for one problem |
| `SutroYaro/MultiChallenge-v0` | All three challenges | Generalize across parity, sum, AND |

## Methods (action space)

| Index | Method | Category |
|-------|--------|----------|
| 0 | SGD | Neural net |
| 1 | Per-layer | Neural net |
| 2 | Sign SGD | Neural net |
| 3 | Curriculum | Neural net |
| 4 | Forward-Forward | Neural net |
| 5 | GF(2) | Algebraic |
| 6 | KM Influence | Algebraic |
| 7 | SMT | Algebraic |
| 8 | Fourier | Algebraic |
| 9 | LASSO | Info-theoretic |
| 10 | MDL | Info-theoretic |
| 11 | Mutual Info | Info-theoretic |
| 12 | Random Proj | Info-theoretic |
| 13 | RL | Alternative |
| 14 | Genetic Prog | Alternative |
| 15 | Evolutionary | Alternative |

5 methods are implemented in the harness (SGD, GF2, KM, SMT, Fourier). The other 11 return failure, which is itself a signal the agent must learn from.

## Baseline results

| Agent | Mean Reward | Discovery Score | Best Method |
|-------|------------|-----------------|-------------|
| Oracle | 0.89 | 49/49 (100%) | GF2 |
| Greedy | 10.21 | 48/49 (98%) | GF2 |
| Random | 9.16 | 41/49 (84%) | GF2 |

Oracle gets the lowest reward but the highest discovery score. The reward function favors improvement trajectories (finding SGD first, then improving to GF2). The discovery grader measures what the agent figured out regardless of order.

## Discovery grading

Beyond metric improvement, the grader scores research quality:

| Category | Points | What it measures |
|----------|--------|-----------------|
| Discovered algebraic solver | 10 | Found GF2, KM, or SMT |
| Identified local learning failure | 5 | Tried Forward-Forward, observed it fails |
| Found metric disagreement | 5 | Noticed KM wins ARD but GF2 wins DMC |
| Optimized beyond baseline | 3 | Beat SGD's DMC of 1,278,460 |
| Correct failure classification | 16 | Observed failures and moved on |
| Efficiency | 5 | Found best method quickly |
| Exploration breadth | 5 | Number of successful methods found |

Total: 49 points.

## Running the evaluation

```bash
PYTHONPATH=src python3 src/sparse_parity/eval/run_eval.py
```

Runs 3 baseline agents x 5 episodes in ~4 seconds. Outputs results to `results/eval/baselines.json` and `results/eval/multi_challenge.json`.

## Answer key

The answer key at `src/sparse_parity/eval/answer_key.json` contains 36 experiments, 12 negative results, and the grading rubric. This is what makes the environment different from typical benchmarks: we know the optimal policy, so we can measure how close the agent gets.

## Portability

The Gymnasium interface is the standard. This environment can be adopted by:

- **PrimeIntellect**: Accepts Gymnasium envs for their research grants program. Our answer key and grading rubric are the differentiator.
- **Modal Labs**: Swap the harness to run experiments on GPU. Needed for scaling to larger n/k values.
- **Anthropic / OpenAI evals**: Wrap as tool-use evaluation. Agent gets `run_experiment` and `read_discoveries` tools instead of discrete indices.
- **HuggingFace Spaces**: Host a leaderboard where agents submit code and get scored.

The key advantage: we have ground truth. Most research envs don't know the optimal policy. We have 36 experiments showing what works, what fails, and why.

## Files

| File | Purpose |
|------|---------|
| `src/sparse_parity/eval/env.py` | Gymnasium environment |
| `src/sparse_parity/eval/baselines.py` | Random, Greedy, Oracle agents |
| `src/sparse_parity/eval/grader.py` | Discovery scoring |
| `src/sparse_parity/eval/answer_key.json` | Ground truth (36 experiments) |
| `src/sparse_parity/eval/README.md` | Full interface spec |
| `src/sparse_parity/eval/run_eval.py` | Evaluation script |
