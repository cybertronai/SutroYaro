"""
SutroYaro evaluation environment for Gymnasium.

Importing this package registers:
    SutroYaro/SparseParity-v0
    SutroYaro/MultiChallenge-v0
"""

# Registration happens on import of env module
from sparse_parity.eval.env import SutroYaroEnv, MultiChallengeEnv, METHOD_MAP  # noqa: F401
