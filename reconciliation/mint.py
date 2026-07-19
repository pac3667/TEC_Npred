"""MinT forecast reconciliation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_error_covariance(
    residuals: pd.DataFrame,
    ridge_eps: float = 1e-6,
) -> np.ndarray:
    """Estimate full residual covariance for MinT on calibration residuals."""
    if residuals.empty:
        raise ValueError("residuals must not be empty")
    cov = np.cov(residuals.to_numpy(dtype=float), rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + np.eye(cov.shape[0]) * ridge_eps
    return cov


def reconcile_mint(
    y_hat: np.ndarray,
    S: np.ndarray,
    covariance: np.ndarray,
    ridge_eps: float = 1e-6,
    use_pinv: bool = True,
) -> np.ndarray:
    """Reconcile forecasts using MinT with a full error covariance matrix."""
    y = np.asarray(y_hat, dtype=float).reshape(-1)
    matrix = np.asarray(S, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (matrix.shape[0], matrix.shape[0]):
        raise ValueError("covariance shape must match hierarchy size")
    cov = cov + np.eye(cov.shape[0]) * ridge_eps
    inv = np.linalg.pinv if use_pinv else np.linalg.inv
    w_inv = inv(cov)
    middle = inv(matrix.T @ w_inv @ matrix)
    return matrix @ middle @ matrix.T @ w_inv @ y
