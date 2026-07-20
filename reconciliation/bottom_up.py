"""Bottom-Up forecast reconciliation."""

from __future__ import annotations

import numpy as np


def reconcile_bottom_up(unit_forecasts: np.ndarray) -> np.ndarray:
    """Return [N_chp, N_1, ..., N_m] with N_chp equal to the unit sum."""
    units = np.asarray(unit_forecasts, dtype=float).reshape(-1)
    return np.concatenate([[float(np.sum(units))], units])
