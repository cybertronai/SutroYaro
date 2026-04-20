"""
SutroYaro evaluation environment for Gymnasium.

Importing this package registers:
    SutroYaro/SparseParity-v0
    SutroYaro/MultiChallenge-v0

The registry (sparse_parity.eval.registry) is populated with default
challenges and methods before the environment classes are loaded.
To add custom challenges/methods, call registry.register_challenge()
or registry.register_method() after importing this package.
"""

# 1. Populate the registry with built-in challenges and methods.
#    This MUST happen before env.py is imported, because env.py reads
#    the registry to set up observation/action spaces.
from sparse_parity.eval.default_registry import register_defaults  # noqa: F401
register_defaults()

# 2. Import env module (triggers Gymnasium registration + sets up classes).
from sparse_parity.eval.env import SutroYaroEnv, MultiChallengeEnv, METHOD_MAP  # noqa: F401
from sparse_parity.eval import registry  # noqa: F401

# 3. Set the module-level NUM_METHODS snapshot now that defaults are loaded.
import sparse_parity.eval.env as _env_module
_env_module.NUM_METHODS = len(registry.list_methods())


def register_all():
    """Gymnasium entry point (see pyproject.toml ``gymnasium.envs``).

    Importing this module already calls ``gymnasium.register(...)`` at the
    bottom of ``env.py``, so this function is a no-op most of the time.
    It exists so Gymnasium's plugin loader has a stable callable to
    invoke; re-calling it is safe because ``gymnasium.register`` will
    overwrite (or refuse) duplicate ids without raising.
    """
    # Touching these names keeps linters happy and guarantees the
    # side-effectful ``env`` module is on ``sys.modules``.
    _ = (SutroYaroEnv, MultiChallengeEnv)
