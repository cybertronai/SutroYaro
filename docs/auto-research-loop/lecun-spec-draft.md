# Spec: minimum implementation requirements for LeCun-problem stubs (v1): DRAFT

> Working draft, not yet posted. This becomes issue #1 in a `lecun-problems`
> repo. Companion to
> [hinton-problems #1](https://github.com/cybertronai/hinton-problems/issues/1)
> and [schmidhuber-problems #1](https://github.com/cybertronai/schmidhuber-problems/issues/1).
> Same scaffold contract, regrouped for LeCun's representational / applied lineage.
> The wave plan below is a **seed**: the paper-to-stub index step refines the exact
> stub list before wave 0.

## Why this work

Hinton's catalog gave us 53 representational-toy baselines. Schmidhuber's gave us
the algorithmic counterpart. LeCun's lineage sits closest to Hinton's but leans
**applied**: convolution and weight sharing, the training tricks that made
backprop work in practice, capacity control by second-order pruning, invariance to
transformations, energy-based and contrastive losses, and the self-supervised
methods that came later.

**v1 (this issue)** = pure-numpy, laptop-runnable baselines for every implementable
stub. Same role as Hinton/Schmidhuber v1: produce a filtered list of "actually
reproducible" problems. v2 instruments those baselines with
[ByteDMD](https://github.com/cybertronai/ByteDMD) and searches for solvers that
beat backprop on data movement.

## What every implemented stub must contain

For a stub at folder `<slug>/`:

```
<slug>/
├── <slug>.py             # model + train + eval; runnable from CLI (--seed)
├── README.md             # 8 sections (see below)
├── make_<slug>_gif.py    # generates <slug>.gif
├── visualize_<slug>.py   # static curves + weight/feature viz → viz/
├── <slug>.gif            # the animation
└── viz/                  # PNGs from visualize_<slug>.py
```

## Dependencies (pure numpy by default)

**Allowed by default:** `numpy` (all model logic, training, data generation),
`matplotlib` (viz, GIFs), Python stdlib.

**Allowed for the documented use case only:** `torchvision.datasets` for
**MNIST / CIFAR / USPS loading only** (the dataset object, never model code);
`imageio` or `pillow` only for GIF assembly.

**Disallowed in v1:** `torch` / `tensorflow` / `jax` for model code; `scipy`
(hand-roll in numpy; if a specific distribution is unavoidable, justify in
§Deviations); any install heavier than a single `pip install`.

## README sections (8, all required)

1. **Header**: `# <slug>` + paper citation + the GIF
2. **Problem**: concise restatement, including dataset/input shape
3. **Files**: table mapping each file to its purpose
4. **Running**: exact command(s) to reproduce the headline result + laptop time
5. **Results**: final metric table with seed, hyperparameters, multi-seed rate
6. **Visualizations**: caption each PNG/GIF (learned filters, embeddings,
   decision boundaries, energy surfaces, training curves)
7. **Deviations from the original**: one line per change from the paper, with the
   reason
8. **Open questions / next experiments**: what is not yet replicated; what would
   close the gap; what is interesting for v2

## Reproducibility rules

- **Seed is a CLI flag.** `python3 <slug>.py --seed N` is deterministic on the
  same machine.
- **Hyperparameters live in §Results**, not buried in code.
- **The §Running command reproduces the §Results headline.** No undocumented flags.
- **Final accuracy is reported with its seed.** If solve rate < 100%, report a
  multi-seed success rate.

## Acceptance checklist (10 boxes)

- [ ] `python3 <slug>.py --seed <N>` reproduces under 5 min on a laptop CPU
- [ ] Final metric reported with that seed in §Results
- [ ] `<slug>.gif` exists and shows training dynamics (≤2 MB target)
- [ ] Static weight / feature / embedding visualization under `viz/`
- [ ] Training curves PNG under `viz/`
- [ ] §Deviations enumerates every change from the paper
- [ ] §Open questions flags what is not yet matched
- [ ] No `NotImplementedError` left in the file
- [ ] PR description states **"Paper reports X; we got Y. Reproduces: yes/no."**
- [ ] PR description states **wallclock to run** + **agent token / wallclock budget to implement**

## Per-stub catalog row

Every stub PR contributes a row to the top-level `README.md` table:

| Problem | Source paper (year) | Reproduces? | Difficulty (tokens or agent-h) | Run wallclock |
|---|---|---|---|---|
| lenet1-digits | LeCun et al. 1989 | yes / no | e.g. 22k tokens | e.g. 40s |

## LeCun-specific faithfulness guidance

LeCun's catalog is mostly representational and applied. Unlike the Schmidhuber
lineage, the **exact optimizer is usually not the experiment**: a small numpy
backprop loop is an acceptable stand-in. What must be preserved is the
**architectural or loss ingredient that is the paper's claim**:

- ConvNet stubs: keep **convolution + weight sharing + local receptive fields**.
  The claim is that a constrained, shift-invariant net generalizes better than an
  unconstrained one of similar size. Show both.
- Invariance stubs: keep the **tangent / transformation penalty** (Tangent Prop)
  or the **transformation-invariant distance** (Tangent Distance). The claim is
  better generalization from few samples under known transformations.
- Energy-based / metric stubs: keep the **contrastive or margin loss** that pushes
  energy down on data and up elsewhere. The claim is that the loss choice prevents
  collapse; show a collapsing loss as the negative control.
- Self-supervised stubs: keep the **decorrelation / variance term** (Barlow
  cross-correlation, VICReg variance-invariance-covariance). The claim is collapse
  avoidance without negative pairs.

If the paper's exact architecture provably cannot converge under numpy-only
constraints: run a ≥30-seed sweep documenting the failure, propose a justified
alternative with the mathematical reason, and document both in §Deviations.

## Wave plan (one PR per wave, grouped by method family)

One PR per wave from wave 0. Each wave PR review passes before the next wave
starts. Waves are grouped by method family so each teammate in a wave shares one
methodology context and the audit reviews one family at a time.

| Wave | Family | Seed stubs |
|---|---|---|
| 0 | Sanity (single-stub validation) | `conv-weight-sharing-sanity` (tiny shift-invariant conv on a 2-class toy) (1) |
| 1 | ConvNets / weight sharing | `lenet1-digits` (LeCun et al. 1989, NIPS), `lenet5-mnist` (LeCun-Bottou-Bengio-Haffner 1998, Proc. IEEE), `constrained-vs-unconstrained` (weight-sharing generalization claim) (3) |
| 2 | Training dynamics + capacity control | `efficient-backprop-tricks` (LeCun-Bottou-Orr-Müller 1998: input normalization, tanh, per-weight LR), `optimal-brain-damage` (LeCun-Denker-Solla 1989/1990: second-derivative saliency pruning vs magnitude pruning) (2) |
| 3 | Invariance to transformations | `tangent-prop` (Simard-Victorri-LeCun-Denker ~1991-92), `tangent-distance` (Simard-LeCun-Denker 1993), `siamese-signature` (Bromley-Bentz-Bottou-LeCun 1993, toy) (3) |
| 4 | Energy-based + metric learning | `drlim-contrastive` (Hadsell-Chopra-LeCun 2006, CVPR: invariant low-dim mapping), `ebm-tutorial-losses` (LeCun-Chopra-Hadsell-Ranzato-Huang 2006: good vs collapsing loss) (2) |
| 5 | Sparse coding / unsupervised features | `predictive-sparse-decomposition` (Kavukcuoglu-Ranzato-LeCun ~2008-10), `multi-stage-architecture` (Jarrett-Kavukcuoglu-Ranzato-LeCun 2009, ICCV: random-filter + rectification + normalization finding) (2) |
| 6 | Modern self-supervised (toy numpy) | `barlow-twins` (Zbontar-Jing-Misra-LeCun-Deny 2021, ICML), `vicreg` (Bardes-Ponce-LeCun 2022, ICLR) (2) |

**v1 seed total: ~15 stubs across 7 waves.** Expect the index step to add stubs
within families (more of the 1989-1998 ConvNet/digit line, more EBM loss variants).

**v1.5 deferred (need real datasets or heavier compute, one follow-up issue each):**
LeNet-5 at full-MNIST paper scale, Graph Transformer Networks (the 1998 paper's
structured-prediction half), multi-column object recognition on CIFAR/NORB,
real-image self-supervised pretraining at ImageNet-style scale.

Why method-family grouping: a ConvNet wave shares one conv + backprop utility; an
invariance wave shares the transformation-jacobian code; a self-supervised wave
shares the augmentation + decorrelation-loss code. Teammates in a wave can build a
small shared helper before fanning out. The audit reviews one family with one
paper-set open.

## Out of scope for v1 (deferred)

- **Energy / data-movement metrics**: v2 with ByteDMD instrumentation
- **GPU runs**: laptop CPU only
- **Paper-scale datasets and training**: v1.5 once the data is wired

## What v1 success looks like

- Every v1 stub implemented, each runnable in <5 min on an M-series laptop
- Catalog table in `README.md`: paper claim vs achieved, reproduces y/n,
  difficulty, wallclock
- Follow-up issues opened for v1.5 deferred stubs
- Honest gap analysis (reproduced vs partial vs non-replication) in a closing
  comment on this issue

---

_agent-0bserver07 (Claude Code) on behalf of Yad_
