"""Deterministic seeding shared by every entry point and every probe."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

_SEEDED: int | None = None


def set_seed(seed: int) -> int:
    """Seed every generator the release draws from and return the seed used."""
    global _SEEDED
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    _SEEDED = seed
    return seed


def active_seed() -> int | None:
    """Return the seed most recently installed, or None when never seeded."""
    return _SEEDED


def numpy_generator(seed: int) -> np.random.Generator:
    """A generator that does not disturb the process-wide NumPy state."""
    return np.random.default_rng(seed)
