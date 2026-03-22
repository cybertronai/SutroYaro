# Task 008: Pre-Experiment Plan

**Priority**: MEDIUM
**Status**: IN PROGRESS
**Source**: Weekly catchup 2026-03-22, Telegram discussion

## Overview

Strategic tasks that set up the next phase of research. Not urgent for tomorrow's meeting but important for the week ahead.

## Checklist

### DMC vs ARD Comparison (Issue #6)

- [ ] Run all 33 experiments with DMC tracking enabled
- [ ] Generate side-by-side ranking: DMC vs ARD for each method
- [ ] Identify cases where rankings differ (these are the interesting ones)
- [ ] Write up comparison as a finding
- [ ] Comment on Issue #6 with results

### Sparse Parity as RL Environment

- [ ] Design env interface: observation space, action space, reward signal
- [ ] Observation: problem spec (n, k, metric type) + current best score
- [ ] Action: algorithm choice + hyperparameter selection
- [ ] Reward: improvement in DMC/ARD over baseline
- [ ] Prototype `src/sparse_parity/rl_env.py` with Gymnasium interface
- [ ] Test: can a simple agent (random search) find GF(2) through the env?
- [ ] Document as a finding if the 33 experiments serve as ground truth

### Public Domain License

- [x] Add LICENSE file (CC0-1.0 or Unlicense) to repo root
- [x] Confirm with Yaroslav on preferred license text
- [ ] Update README if needed

### Lukas Kaiser / Mar 30 Meeting Prep

- [ ] Ensure docs site is up to date before Mar 30
- [ ] Review survey page for completeness
- [ ] Prepare a 5-minute "state of research" summary for newcomers

## Context

Yaroslav's idea (chat-yaroslav, Mar 21): wrap challenges into RL environments for Anthropic/PrimeIntellect. Our 33 experiments serve as an answer key -- richer signal than most RL envs.

PrimeIntellect research grants: compute + stipends for novel environments.
