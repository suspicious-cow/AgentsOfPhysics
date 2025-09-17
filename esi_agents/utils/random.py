"""Random seed helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SeedConfig:
    """Configuration for deterministic seeding."""

    python_seed: int = 42
    numpy_seed: int = 42


def set_global_seed(seed: Optional[SeedConfig] = None) -> None:
    """Set deterministic seeds across supported libraries."""

    config = seed or SeedConfig()
    random.seed(config.python_seed)
    np.random.seed(config.numpy_seed)


__all__ = ["SeedConfig", "set_global_seed"]
