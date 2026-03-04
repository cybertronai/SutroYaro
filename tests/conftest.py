import sys
from pathlib import Path

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_parity.config import Config

import pytest


@pytest.fixture
def small_config():
    """Tiny config for fast tests."""
    return Config(n_bits=3, k_sparse=3, n_train=20, n_test=20, hidden=100, max_epochs=5, seed=42)


@pytest.fixture
def scale_config():
    """20-bit config for scaling tests."""
    return Config(n_bits=20, k_sparse=3, n_train=200, n_test=200, hidden=2000, max_epochs=50, seed=42)
