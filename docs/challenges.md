# The Challenges

The Sutro Group runs several energy-efficient-learning challenges at once. Each lives in its own repo with its own leaderboard. This page is the single challenge-centric index: what each one is, where it lives, where it stands, and how to start. For the repo-centric view see [Related Repos](related-repos.md); for what changed recently see the latest [catch-up](catchups/index.md).

## At a glance

| Challenge | Goal | Repo | Cost metric |
|-----------|------|------|-------------|
| #1 Sparse Parity | Learn k-bit XOR for minimum data movement | sparse-parity-challenge | ByteDMD |
| #2 Energy-efficient matmul | Minimum-energy 16x16 matmul on a 2D grid | sutro-problems/matmul | Dally 2D-grid |
| #3 Sparse parity on the grid | Solve sparse parity in ~9 grid instructions | sutro-problems/sparse-parity | Dally 2D-grid |
| wikitext | Train a WikiText-103 LM for minimum Joules | cybertronai/wikitext | Measured GPU energy (NVML) |

Two baseline catalogs (hinton-problems, schmidhuber-problems) feed the process that picks the next challenge. See [Baseline catalogs](#baseline-catalogs) below.

## Challenge #1 — Sparse Parity

**What it is.** Learn y = XOR of k secret bits from n random plus/minus-1 inputs (standard n=20, k=3, 17 noise bits). The original benchmark, the "drosophila" of energy-efficient training.

**Where.** [cybertronai/sparse-parity-challenge](https://github.com/cybertronai/sparse-parity-challenge) runs the submission pipeline: open a GitHub issue with a `solve()` function, CI scores it under ByteDMD and posts to the leaderboard. The research code (solvers, experiments) was migrated there from SutroYaro in May 2026.

**State.** Pipeline is live; submissions came in as recently as May 12. The Telegram channel is quiet since March, the work moved to GitHub. The best known methods (KM-min, GF(2) elimination) were measured under the legacy element-level metric and have not been re-measured under ByteDMD.

**Entry point.** The sparse-parity-challenge README. Background: [Sparse Parity Challenge](research/sparse-parity-challenge.md).

## Challenge #2 — Energy-efficient matmul

**What it is.** Minimize the data-movement energy of matrix multiplication, expressed as an intermediate representation (explicit load and store ops) on Bill Dally's 2D grid. The focus size is 16x16.

**Where.** [cybertronai/sutro-problems](https://github.com/cybertronai/sutro-problems), `matmul/` directory. The scorer (`matmul.py`) is locked; submissions must not modify it.

**State.** Active hill-climbing. The 16x16 record fell to 67,821 in mid-May. Lower bounds are an open hard problem: an agent-generated bound was found to be wrong, and AlphaTensor could not bound 4x4 either.

**Who is active.** Cosmin Negruseri, Sung Jae Bae, Anastasiia Zhiboedova.

**Entry point.** `sutro-problems/matmul/README.md`; Telegram topic "challenge #2".

## Challenge #3 — Sparse parity on the grid

**What it is.** Solve sparse parity using only about nine instructions on the Dally 2D grid. The grid-model version of Challenge #1: restricted op set, no abstraction.

**Where.** [cybertronai/sutro-problems](https://github.com/cybertronai/sutro-problems), `sparse-parity/` directory. The scorer (`sparse_parity.py`) is locked.

**State.** Launched May 8. Precomputing intermediate XORs plus bit-packing works well; tiling does not help at this problem size. A 50%-accuracy target variant has been added.

**Open questions.** Whether continuous floating-point ops are needed at all, or whether integer and 8-bit instruction sets are better. Whether the op set needs more than the Dally v3 minimum.

**Entry point.** `sutro-problems/sparse-parity/README.md`; Telegram topic "challenge #3".

## wikitext — Energy-efficient language modeling

**What it is.** Train a language model on WikiText-103 for the minimum GPU energy (Joules) at a fixed accuracy and wall-clock time. The largest-scale challenge, meant as the "final" task in a ladder running Shakespeare, TinyStories, WikiText-103, FineWeb.

**Where.** [cybertronai/wikitext](https://github.com/cybertronai/wikitext). Runs on Modal; GPU energy measured via NVML.

**State.** Baseline `modded_nanogpt`: 54,784 J, 0.7285 character-accuracy, 322.7s. A forward-forward submission reaches about 0.39 accuracy at roughly 10x fewer Joules. As of May 20, whether to also count CPU energy (RAPL) is under discussion.

**Who is active.** Armins (lead), Gabriel Nakajima An.

**Entry point.** The cybertronai/wikitext README; Telegram topic "wikitext".

## Baseline catalogs

Not challenges themselves, but the input to choosing the next one. Both shipped in May 2026 via the agent-team wave-build pattern.

| Catalog | Lineage | Stubs |
|---------|---------|-------|
| [hinton-problems](https://github.com/cybertronai/hinton-problems) | Hinton 1981-2022, representational tasks | 53 + 2 |
| [schmidhuber-problems](https://github.com/cybertronai/schmidhuber-problems) | Schmidhuber 1989-2025, algorithmic tasks | 50 + 8 |

## The v2 / v3 roadmap

The catalogs get filtered in two passes: **v2** keeps the stubs that the ByteDMD metric can instrument, **v3** keeps the further subset the Dally 2D-grid model can instrument. The survivors become the candidate pool for the next hill-climbing competition. Tracked in [Task 015](tasks/015-v2-v3-instrumentation.md).

## Adding a challenge

See [How to Add a New Challenge](research/adding-a-challenge.md). That guide still uses the older sparse-sum example and is due an update.
