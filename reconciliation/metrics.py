"""Metrics for reconciliation baseline experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "plant_id",
    "model_name",
    "method",
    "forecast_stage",
    "MAE_station",
    "RMSE_station",
    "MAPE_station",
    "MAE_units_mean",
    "RMSE_units_mean",
    "MAPE_units_mean",
    "OCV_min_total",
    "OCV_max_total",
    "OCV_total",
    "OCV_rate",
    "OCV_min_severity_total",
    "OCV_max_severity_total",
    "HIC_count_total",
    "HIC_rate",
    "HIC_less_1_percent",
    "HIC_1_5_percent",
    "HIC_more_5_percent",
    "HIC_mean_percent",
    "HIC_median_percent",
    "HIC_max_percent",
    "RD_signed_percent",
    "RD_abs_percent",
    "RD_mean_abs_mw",
    "RD_max_abs_mw",
    "physical_success_rate",
    "physical_runtime_mean_sec",
    "physical_iterations_mean",
]

UNIT_COLUMNS = [
    "plant_id",
    "model_name",
    "method",
    "forecast_stage",
    "unit_id",
    "MAE",
    "RMSE",
    "MAPE",
    "OCV_min",
    "OCV_max",
    "OCV_total",
    "OCV_min_severity",
    "OCV_max_severity",
]


def calculate_ocv(unit_forecasts: pd.DataFrame, n_min: pd.DataFrame, n_max: pd.DataFrame) -> dict[str, float]:
    """Calculate operational constraint violations for unit forecasts."""
    pred = unit_forecasts.to_numpy(dtype=float)
    low = n_min.to_numpy(dtype=float)
    high = n_max.to_numpy(dtype=float)
    min_violation = pred < low
    max_violation = pred > high
    min_severity = np.where(min_violation, low - pred, 0.0)
    max_severity = np.where(max_violation, pred - high, 0.0)
    total = int(np.sum(min_violation) + np.sum(max_violation))
    denom = max(pred.size, 1)
    return {
        "OCV_min_total": int(np.sum(min_violation)),
        "OCV_max_total": int(np.sum(max_violation)),
        "OCV_total": total,
        "OCV_rate": total / denom,
        "OCV_min_severity_total": float(np.sum(min_severity)),
        "OCV_max_severity_total": float(np.sum(max_severity)),
    }


def calculate_hic(
    station_forecast: pd.Series,
    unit_forecasts: pd.DataFrame,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Calculate hierarchical inconsistency categories and severity."""
    station = pd.to_numeric(station_forecast, errors="coerce").to_numpy(dtype=float)
    unit_sum = unit_forecasts.sum(axis=1).to_numpy(dtype=float)
    diff = np.abs(station - unit_sum)
    denom = np.maximum(np.abs(station), eps)
    percent = diff / denom * 100.0
    is_inconsistent = percent > 0
    values = percent[is_inconsistent]
    n_obs = max(len(percent), 1)
    return {
        "HIC_count_total": int(np.sum(is_inconsistent)),
        "HIC_rate": float(np.sum(is_inconsistent) / n_obs),
        "HIC_less_1_percent": int(np.sum((values > 0) & (values < 1))),
        "HIC_1_5_percent": int(np.sum((values >= 1) & (values <= 5))),
        "HIC_more_5_percent": int(np.sum(values > 5)),
        "HIC_mean_percent": float(np.mean(values)) if len(values) else 0.0,
        "HIC_median_percent": float(np.median(values)) if len(values) else 0.0,
        "HIC_max_percent": float(np.max(values)) if len(values) else 0.0,
    }


def calculate_rd(
    raw_station_forecast: pd.Series,
    reconciled_station_schedule: pd.Series,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Calculate RD using the article formula and absolute diagnostics."""
    raw = pd.to_numeric(raw_station_forecast, errors="coerce").to_numpy(dtype=float)
    recon = pd.to_numeric(reconciled_station_schedule, errors="coerce").to_numpy(dtype=float)
    denom = np.where(np.abs(recon) <= eps, eps, recon)
    delta = recon - raw
    abs_denom = np.maximum(np.abs(recon), eps)
    return {
        "RD_signed_percent": float(np.mean(delta / denom) * 100.0),
        "RD_abs_percent": float(np.mean(np.abs(delta) / abs_denom) * 100.0),
        "RD_mean_abs_mw": float(np.mean(np.abs(delta))),
        "RD_max_abs_mw": float(np.max(np.abs(delta))) if len(delta) else 0.0,
    }


def calculate_forecast_accuracy(y_true: pd.DataFrame, y_pred: pd.DataFrame, eps: float = 1e-8) -> dict[str, float]:
    """Calculate MAE, RMSE and MAPE for aligned columns."""
    common = [col for col in y_true.columns if col in y_pred.columns]
    if not common:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    true = y_true[common].to_numpy(dtype=float)
    pred = y_pred[common].to_numpy(dtype=float)
    err = pred - true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(true), eps)) * 100.0),
    }


def summarize_prediction_table(
    df: pd.DataFrame,
    unit_ids: list[str],
    raw_station_forecast: pd.Series | None = None,
    physical_stats: dict[str, float] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Build one summary row and per-unit metric rows for a prediction table."""
    n_units = len(unit_ids)
    unit_pred_cols = [f"N_{idx}_pred" for idx in range(1, n_units + 1)]
    unit_true_cols = [f"N_{idx}_true" for idx in range(1, n_units + 1)]
    min_cols = [f"N_min_{idx}" for idx in range(1, n_units + 1)]
    max_cols = [f"N_max_{idx}" for idx in range(1, n_units + 1)]

    station_metrics = calculate_forecast_accuracy(
        df[["N_chp_true"]].rename(columns={"N_chp_true": "N_chp"}),
        df[["N_chp_pred"]].rename(columns={"N_chp_pred": "N_chp"}),
    )
    units_metrics = calculate_forecast_accuracy(
        df[unit_true_cols].rename(columns=dict(zip(unit_true_cols, unit_pred_cols))),
        df[unit_pred_cols],
    )
    ocv = calculate_ocv(df[unit_pred_cols], df[min_cols], df[max_cols])
    hic = calculate_hic(df["N_chp_pred"], df[unit_pred_cols])
    rd = calculate_rd(
        raw_station_forecast if raw_station_forecast is not None else df["N_chp_pred"],
        df["N_chp_pred"],
    )
    phys = physical_stats or {}

    first = df.iloc[0]
    summary = {
        "plant_id": first.get("plant_id"),
        "model_name": first.get("model_name"),
        "method": first.get("method"),
        "forecast_stage": first.get("forecast_stage"),
        "MAE_station": station_metrics["MAE"],
        "RMSE_station": station_metrics["RMSE"],
        "MAPE_station": station_metrics["MAPE"],
        "MAE_units_mean": units_metrics["MAE"],
        "RMSE_units_mean": units_metrics["RMSE"],
        "MAPE_units_mean": units_metrics["MAPE"],
        **ocv,
        **hic,
        **rd,
        "physical_success_rate": phys.get("physical_success_rate", np.nan),
        "physical_runtime_mean_sec": phys.get("physical_runtime_mean_sec", np.nan),
        "physical_iterations_mean": phys.get("physical_iterations_mean", np.nan),
    }
    summary = {col: summary.get(col, np.nan) for col in SUMMARY_COLUMNS}

    unit_rows = []
    for idx, unit_id in enumerate(unit_ids, start=1):
        pred_col = f"N_{idx}_pred"
        true_col = f"N_{idx}_true"
        min_col = f"N_min_{idx}"
        max_col = f"N_max_{idx}"
        acc = calculate_forecast_accuracy(
            df[[true_col]].rename(columns={true_col: pred_col}),
            df[[pred_col]],
        )
        unit_ocv = calculate_ocv(df[[pred_col]], df[[min_col]], df[[max_col]])
        unit_rows.append(
            {
                "plant_id": first.get("plant_id"),
                "model_name": first.get("model_name"),
                "method": first.get("method"),
                "forecast_stage": first.get("forecast_stage"),
                "unit_id": unit_id,
                "MAE": acc["MAE"],
                "RMSE": acc["RMSE"],
                "MAPE": acc["MAPE"],
                "OCV_min": unit_ocv["OCV_min_total"],
                "OCV_max": unit_ocv["OCV_max_total"],
                "OCV_total": unit_ocv["OCV_total"],
                "OCV_min_severity": unit_ocv["OCV_min_severity_total"],
                "OCV_max_severity": unit_ocv["OCV_max_severity_total"],
            }
        )
    return summary, pd.DataFrame(unit_rows, columns=UNIT_COLUMNS)
