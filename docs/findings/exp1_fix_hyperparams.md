# Experiment 1: Fix Hyperparameters (Barak et al. 2022)

**Date**: 2026-03-03
**Status**: SUCCESS -- 99.0% test accuracy on 20-bit sparse parity (k=3)

## Hypothesis

Matching Barak et al. 2022's hyperparameters (LR=0.1, batch_size=32, more epochs) will trigger the phase transition on 20-bit sparse parity (k=3), breaking past the ~54% ceiling.

## Result

**Hypothesis confirmed.** The model achieved 99.0% test accuracy at epoch 52 (832 SGD steps), solving 20-bit sparse parity cleanly.

## Configuration

| Parameter   | Old (baseline) | New (this experiment) |
|-------------|----------------|----------------------|
| n_bits      | 20             | 20                   |
| k_sparse    | 3              | 3                    |
| hidden      | 2000           | 1000                 |
| n_train     | 200            | 500                  |
| n_test      | 200            | 200                  |
| lr          | 0.5            | 0.1                  |
| wd          | 0.01           | 0.01                 |
| batch_size  | 1 (online)     | 32 (mini-batch)      |
| max_epochs  | 50             | 200 (solved at 52)   |

## Observations

### Phase transition / grokking pattern

The training curve follows the grokking pattern:

1. **Epochs 1-20**: Train accuracy rises to ~80%, test accuracy stays at chance (~50%). The model memorizes.
2. **Epochs 20-40**: Train accuracy continues climbing, test accuracy begins to move (59% at epoch 30, 75% at epoch 40).
3. **Epochs 40-52**: Phase transition. Test accuracy jumps from 75% to 99% in about 10 epochs.

### Hidden progress tracking

The L1 weight movement ||w_t - w_0||_1 grew steadily throughout training:
- Epoch 1: 241
- Epoch 30: 3,109 (test acc still ~59%)
- Epoch 43: 3,830 (test acc crosses 90%)
- Epoch 52: 4,149 (solved at 99%)

This confirms the "hidden progress" phenomenon from Barak et al.: SGD moves weights well before test accuracy responds. The weight movement metric is a useful leading indicator.

### What fixed it

Two changes mattered most:

1. **Mini-batch SGD (batch_size=32)**: Single-sample online SGD produces noisy gradient estimates. Averaging over 32 samples provides a cleaner signal, especially for sparse parity where 3 of 20 bits are relevant.

2. **Lower learning rate (0.1 vs 0.5)**: LR=0.5 with noisy single-sample gradients causes overshooting. LR=0.1 with mini-batches gives stable convergence.

More training data also helped (n_train=500 vs 200): more samples reduce overfitting to noise bits. With 500 samples and batch_size=32, each epoch has ~16 update steps.

### Performance

- Total training time: ~111 seconds (pure Python, no NumPy)
- Steps to solve: 832 (52 epochs x 16 batches/epoch)
- This is well within the n^O(k) ~ 8000 theoretical bound from Barak et al.

## Secret indices

The randomly selected parity bits were [0, 3, 8] out of 20 bits. The model successfully learned to ignore the 17 noise bits.

## Next Steps

- Experiment 2: Sweep weight decay to see if it can accelerate the phase transition
- Experiment 3: Try Sign SGD to see if it matches the SQ lower bound
- Experiment 4: Try GrokFast to see if amplifying slow gradients speeds up convergence

## Files

- Experiment script: `src/sparse_parity/experiments/exp1_fix_hyperparams.py`
- Results JSON: `results/exp1_20260303_221628/results.json`
