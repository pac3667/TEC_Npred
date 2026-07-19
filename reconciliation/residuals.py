"""Residual helpers for WLS and MinT calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_residuals(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Return residuals y_true - y_pred with aligned columns and rows."""
    common = [col for col in y_true.columns if col in y_pred.columns]
    if not common:
        raise ValueError("No common columns between y_true and y_pred")
    residuals = y_true[common].reset_index(drop=True) - y_pred[common].reset_index(drop=True)
    return residuals.replace([np.inf, -np.inf], np.nan).dropna()
