"""Weighted Least Squares reconciliation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_error_variances(residuals: pd.DataFrame, eps: float = 1e-8) -> np.ndarray:
    """Estimate diagonal error variances from calibration residuals."""
    if residuals.empty:
        raise ValueError("residuals must not be empty")
    variances = residuals.var(axis=0, ddof=1).fillna(eps).clip(lower=eps)
    return variances.to_numpy(dtype=float)


def reconcile_wls(y_hat: np.ndarray, S: np.ndarray, variances: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Reconcile forecasts using diagonal WLS error covariance."""
    y = np.asarray(y_hat, dtype=float).reshape(-1)
    matrix = np.asarray(S, dtype=float)
    var = np.asarray(variances, dtype=float).reshape(-1)
    if var.shape[0] != matrix.shape[0]:
        raise ValueError("variances length must match number of hierarchy series")
    var = np.clip(var, eps, None)
    w_inv = np.diag(1.0 / var)
    middle = np.linalg.pinv(matrix.T @ w_inv @ matrix)
    return matrix @ middle @ matrix.T @ w_inv @ y
