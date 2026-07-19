"""Summing matrix utilities for two-level TEC hierarchies."""

from __future__ import annotations

import numpy as np


def build_summing_matrix(n_units: int) -> np.ndarray:
    """Build S for y = S b, where y = [N_chp, N_1, ..., N_m]."""
    if n_units <= 0:
        raise ValueError("n_units must be positive")
    top = np.ones((1, n_units), dtype=float)
    bottom = np.eye(n_units, dtype=float)
    return np.vstack([top, bottom])
