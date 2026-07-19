"""Top-Down forecast reconciliation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_historical_shares(
    station_true: pd.Series,
    units_true: pd.DataFrame,
    eps: float = 1e-8,
) -> np.ndarray:
    """Estimate p_i = mean(N_i_true / N_chp_true) on calibration data."""
    station = pd.to_numeric(station_true, errors="coerce")
    units = units_true.apply(pd.to_numeric, errors="coerce")
    valid = station.gt(eps) & units.notna().all(axis=1)
    if not valid.any():
        n_units = units.shape[1]
        return np.ones(n_units, dtype=float) / n_units
    shares = units.loc[valid].div(station.loc[valid], axis=0).mean(axis=0)
    shares = shares.clip(lower=0).to_numpy(dtype=float)
    total = float(np.sum(shares))
    if total <= eps:
        return np.ones(units.shape[1], dtype=float) / units.shape[1]
    return shares / total


def reconcile_top_down(
    station_forecast: float,
    shares: np.ndarray,
    availability: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return [N_chp, N_1, ..., N_m] by distributing station forecast by shares."""
    p = np.asarray(shares, dtype=float).reshape(-1)
    if availability is not None:
        available = np.asarray(availability, dtype=float).reshape(-1)
        if available.shape != p.shape:
            raise ValueError("availability must have the same length as shares")
        adjusted = p * available
        if float(np.sum(adjusted)) > eps:
            p = adjusted
    total = float(np.sum(p))
    if total <= eps:
        raise ValueError("top_down shares are all zero after availability adjustment")
    p = p / total
    units = p * float(station_forecast)
    return np.concatenate([[float(station_forecast)], units])
