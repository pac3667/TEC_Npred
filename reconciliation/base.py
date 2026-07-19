"""Shared orchestration for classical reconciliation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bottom_up import reconcile_bottom_up
from .mint import reconcile_mint
from .ols import reconcile_ols
from .summing_matrix import build_summing_matrix
from .top_down import reconcile_top_down
from .wls import reconcile_wls

METHOD_INDEPENDENT = "independent"
METHOD_BOTTOM_UP = "bottom_up"
METHOD_TOP_DOWN = "top_down"
METHOD_OLS = "ols"
METHOD_WLS = "wls"
METHOD_MINT = "mint"

STAGE_RAW = "raw"
STAGE_HIERARCHICAL_RECONCILED = "hierarchical_reconciled"


def hierarchy_columns(n_units: int) -> list[str]:
    """Return unified hierarchy columns: N_chp_pred, N_1_pred, ..., N_m_pred."""
    return ["N_chp_pred"] + [f"N_{idx}_pred" for idx in range(1, n_units + 1)]


def row_to_y_hat(row: pd.Series, n_units: int) -> np.ndarray:
    """Convert a unified prediction row to y_hat = [N_chp, N_1, ..., N_m]."""
    return row[hierarchy_columns(n_units)].to_numpy(dtype=float)


def y_to_row_values(y: np.ndarray, n_units: int) -> dict[str, float]:
    """Convert reconciled y vector to unified prediction columns."""
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.shape[0] != n_units + 1:
        raise ValueError("Reconciled vector length does not match hierarchy")
    return {col: float(values[idx]) for idx, col in enumerate(hierarchy_columns(n_units))}


def apply_reconciliation_method(
    df: pd.DataFrame,
    method: str,
    unit_ids: list[str],
    shares: np.ndarray | None = None,
    variances: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    eps: float = 1e-8,
    ridge_eps: float = 1e-6,
    use_pinv: bool = True,
) -> pd.DataFrame:
    """Apply a reconciliation method to a unified prediction table."""
    n_units = len(unit_ids)
    if n_units <= 0:
        raise ValueError("unit_ids must not be empty")
    method = method.lower()
    output = df.copy()
    output["method"] = method
    output["forecast_stage"] = (
        STAGE_RAW if method == METHOD_INDEPENDENT else STAGE_HIERARCHICAL_RECONCILED
    )

    if method == METHOD_INDEPENDENT:
        return output

    S = build_summing_matrix(n_units)
    pred_cols = hierarchy_columns(n_units)
    reconciled_rows: list[dict[str, float]] = []

    if method == METHOD_TOP_DOWN and shares is None:
        raise ValueError("shares are required for top_down")
    if method == METHOD_WLS and variances is None:
        raise ValueError("variances are required for wls")
    if method == METHOD_MINT and covariance is None:
        raise ValueError("covariance is required for mint")

    for _, row in output.iterrows():
        y_hat = row_to_y_hat(row, n_units)
        if method == METHOD_BOTTOM_UP:
            y_rec = reconcile_bottom_up(y_hat[1:])
        elif method == METHOD_TOP_DOWN:
            availability = _row_availability(row, n_units)
            y_rec = reconcile_top_down(y_hat[0], shares=shares, availability=availability, eps=eps)
        elif method == METHOD_OLS:
            y_rec = reconcile_ols(y_hat, S)
        elif method == METHOD_WLS:
            y_rec = reconcile_wls(y_hat, S, variances=variances, eps=eps)
        elif method == METHOD_MINT:
            y_rec = reconcile_mint(
                y_hat,
                S,
                covariance=covariance,
                ridge_eps=ridge_eps,
                use_pinv=use_pinv,
            )
        else:
            raise ValueError(f"Unknown reconciliation method: {method}")
        reconciled_rows.append(y_to_row_values(y_rec, n_units))

    reconciled = pd.DataFrame(reconciled_rows, index=output.index)
    output.loc[:, pred_cols] = reconciled[pred_cols]
    return output


def _row_availability(row: pd.Series, n_units: int) -> np.ndarray | None:
    flags = []
    for idx in range(1, n_units + 1):
        min_value = row.get(f"N_min_{idx}", np.nan)
        max_value = row.get(f"N_max_{idx}", np.nan)
        if pd.isna(min_value) or pd.isna(max_value):
            return None
        flags.append(1.0 if float(max_value) > 0 or float(min_value) > 0 else 0.0)
    return np.asarray(flags, dtype=float)
