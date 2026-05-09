"""
Out-of-harness challenge modules.

Challenges live here instead of in `src/harness.py` when they are added
AFTER the harness was locked (LAB.md rule #9). Each module exports a
`measure_<slug>()` function with the same signature as
`measure_sparse_parity()` in harness.py, and each is registered via
`sparse_parity.eval.default_registry`.

Currently provided:
    - majority_vote    (issue #7)
    - threshold        (issue #7)
    - noisy_parity     (issue #7)

See docs/research/adding-a-challenge.md for the full recipe.
"""

from sparse_parity.challenges.majority_vote import measure_majority_vote
from sparse_parity.challenges.threshold import measure_threshold
from sparse_parity.challenges.noisy_parity import measure_noisy_parity

__all__ = [
    "measure_majority_vote",
    "measure_threshold",
    "measure_noisy_parity",
]
