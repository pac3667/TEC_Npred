"""Ordinary Least Squares reconciliation."""

from __future__ import annotations

import numpy as np


def reconcile_ols(y_hat: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Reconcile forecasts with y_rec = S inv(S.T S) S.T y_hat."""
    y = np.asarray(y_hat, dtype=float).reshape(-1)
    matrix = np.asarray(S, dtype=float)
    middle = np.linalg.pinv(matrix.T @ matrix)
    return matrix @ middle @ matrix.T @ y
