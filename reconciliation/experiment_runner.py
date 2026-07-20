"""Experiment runner for classical reconciliation baselines."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import (
    METHOD_BOTTOM_UP,
    METHOD_INDEPENDENT,
    METHOD_MINT,
    METHOD_OLS,
    METHOD_TOP_DOWN,
    METHOD_WLS,
    STAGE_RAW,
    apply_reconciliation_method,
)
from .metrics import SUMMARY_COLUMNS, UNIT_COLUMNS, summarize_prediction_table
from .mint import estimate_error_covariance
from .top_down import estimate_historical_shares
from .wls import estimate_error_variances

DEFAULT_CONFIG: dict[str, Any] = {
    "data_file": "data/TEC22_Data.csv",
    "reports_dir": None,
    "output_dir": "outputs/reconciliation_baselines",
    "plant_id": None,
    "model_name": "all",
    "methods": [
        METHOD_INDEPENDENT,
        METHOD_BOTTOM_UP,
        METHOD_TOP_DOWN,
        METHOD_OLS,
        METHOD_WLS,
        METHOD_MINT,
    ],
    "eps": 1.0e-8,
    "ridge_eps": 1.0e-6,
    "use_pinv": True,
}


def run_reconciliation_baselines(config: dict[str, Any]) -> dict[str, Path]:
    """Run all requested reconciliation baseline methods and save outputs."""
    cfg = {**DEFAULT_CONFIG, **config}
    project_root = Path(cfg.get("project_root", ".")).resolve()
    data_file = _resolve_path(project_root, cfg["data_file"])
    plant_id = cfg["plant_id"] or _plant_id_from_data_file(data_file)
    reports_dir = _resolve_path(
        project_root,
        cfg["reports_dir"] or f"data/reports/{data_file.stem}",
    )
    output_dir = _resolve_path(project_root, cfg["output_dir"])
    pred_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    logs_dir = output_dir / "logs"
    for path in [pred_dir, metrics_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    for old_prediction in pred_dir.glob("*.csv"):
        old_prediction.unlink()

    log_lines = [
        f"Run started: {datetime.now().isoformat(timespec='seconds')}",
        f"data_file={data_file}",
        f"reports_dir={reports_dir}",
        f"plant_id={plant_id}",
        f"methods={','.join(cfg['methods'])}",
    ]

    model_names = _select_model_names_for_run(reports_dir, cfg["model_name"])
    log_lines.append(f"models={','.join(model_names)}")

    summary_rows = []
    unit_metric_frames = []

    for model_name in model_names:
        try:
            source = load_prediction_table_from_reports(
                reports_dir=reports_dir,
                data_file=data_file,
                plant_id=plant_id,
                model_name=model_name,
                eps=float(cfg["eps"]),
            )
            unit_ids = source["unit_ids"]
            raw_df = source["predictions"]
            calibration = source["calibration"]

            shares = estimate_historical_shares(
                calibration["N_chp_true"],
                calibration[[f"N_{idx}_true" for idx in range(1, len(unit_ids) + 1)]],
                eps=float(cfg["eps"]),
            )
            residuals = _load_calibration_residuals(reports_dir, model_name, unit_ids, len(unit_ids), log_lines)
            variances = estimate_error_variances(residuals, eps=float(cfg["eps"]))
            covariance = estimate_error_covariance(residuals, ridge_eps=float(cfg["ridge_eps"]))
        except Exception as exc:
            log_lines.append(f"ERROR model={model_name}: {exc}")
            continue

        for method in cfg["methods"]:
            try:
                hierarchical = apply_reconciliation_method(
                    raw_df,
                    method=method,
                    unit_ids=unit_ids,
                    shares=shares,
                    variances=variances,
                    covariance=covariance,
                    eps=float(cfg["eps"]),
                    ridge_eps=float(cfg["ridge_eps"]),
                    use_pinv=bool(cfg["use_pinv"]),
                )
                _save_predictions(pred_dir, plant_id, model_name, method, hierarchical)
                summary, unit_metrics = summarize_prediction_table(
                    hierarchical,
                    unit_ids=unit_ids,
                    raw_station_forecast=raw_df["N_chp_pred"],
                )
                summary_rows.append(summary)
                unit_metric_frames.append(unit_metrics)
            except Exception as exc:
                log_lines.append(f"ERROR model={model_name} method={method}: {exc}")

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    unit_df = (
        pd.concat(unit_metric_frames, ignore_index=True)
        if unit_metric_frames
        else pd.DataFrame(columns=UNIT_COLUMNS)
    )
    summary_path = metrics_dir / "summary_metrics.csv"
    unit_path = metrics_dir / "unit_metrics.csv"
    log_path = logs_dir / "run_log.txt"
    summary_df.to_csv(summary_path, index=False)
    unit_df.to_csv(unit_path, index=False)
    log_lines.append(f"Run finished: {datetime.now().isoformat(timespec='seconds')}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return {"summary": summary_path, "unit": unit_path, "log": log_path}


def load_prediction_table_from_reports(
    reports_dir: Path,
    data_file: Path,
    plant_id: str,
    model_name: str,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Build the unified prediction table from existing project report files."""
    station_path = reports_dir / "Power_Station_final_predictions_comparison.xlsx"
    if not station_path.exists():
        raise FileNotFoundError(f"Missing station prediction report: {station_path}")
    station_df = pd.read_excel(station_path)
    unit_ids = _detect_units(reports_dir)
    if not unit_ids:
        raise ValueError(f"No unit prediction reports found in {reports_dir}")
    unit_dfs = {
        unit: pd.read_excel(reports_dir / f"{unit}_final_predictions_comparison.xlsx")
        for unit in unit_ids
    }
    selected_model = _select_model_name(reports_dir, model_name, station_df)
    n_rows = min([len(station_df), *[len(df) for df in unit_dfs.values()]])
    bounds = build_bounds_table(data_file, n_rows=n_rows)
    n_rows = min(n_rows, len(bounds))

    table = pd.DataFrame(
        {
            "datetime": bounds["datetime"].tail(n_rows).to_numpy(),
            "plant_id": plant_id,
            "model_name": selected_model,
            "method": METHOD_INDEPENDENT,
            "forecast_stage": STAGE_RAW,
            "N_chp_pred": station_df[selected_model].tail(n_rows).to_numpy(dtype=float),
            "N_chp_true": station_df["Actual_N"].tail(n_rows).to_numpy(dtype=float),
        }
    )
    for idx, unit in enumerate(unit_ids, start=1):
        unit_df = unit_dfs[unit]
        table[f"N_{idx}_pred"] = unit_df[selected_model].tail(n_rows).to_numpy(dtype=float)
        table[f"N_{idx}_true"] = unit_df["Actual_N"].tail(n_rows).to_numpy(dtype=float)
        table[f"N_min_{idx}"] = bounds[f"{unit}_Available_Nmin"].tail(n_rows).to_numpy(dtype=float)
        table[f"N_max_{idx}"] = bounds[f"{unit}_Available_Nmax"].tail(n_rows).to_numpy(dtype=float)

    calibration = build_bounds_table(data_file, n_rows=None)
    calibration = calibration.iloc[: max(len(calibration) - n_rows, 1)].copy()
    calibration = _rename_calibration_columns(calibration, unit_ids)
    return {"predictions": table, "unit_ids": unit_ids, "calibration": calibration}


def build_bounds_table(data_file: Path, n_rows: int | None) -> pd.DataFrame:
    """Recompute true values and Nmin/Nmax constraints from the source CSV."""
    data = pd.read_csv(data_file, delimiter=";", parse_dates=["Date"], dayfirst=True)
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    first_work_day = data[data["TEC_N_Aver"] > 0].index[0]
    data = data.loc[first_work_day:].copy()
    if data_file.name == "TEC22_Data.csv":
        _add_tec22_bounds(data)
        units = ["B1", "B2", "B3", "B4"]
    elif data_file.name == "TEC14_Data.csv":
        _add_tec14_bounds(data)
        units = ["B1", "B2"]
    else:
        raise ValueError(f"Unsupported data file: {data_file.name}")
    cols = ["Date", "TEC_N_Aver"] + [f"{unit}_N_Aver" for unit in units]
    cols += [f"{unit}_Available_Nmin" for unit in units]
    cols += [f"{unit}_Available_Nmax" for unit in units]
    out = data[cols].dropna().copy()
    out = out.rename(columns={"Date": "datetime", "TEC_N_Aver": "N_chp_true"})
    if n_rows is not None:
        out = out.tail(n_rows)
    return out.reset_index(drop=True)


def load_config(path: Path | None) -> dict[str, Any]:
    """Load a simple YAML-like config without adding a PyYAML dependency."""
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return _parse_simple_yaml(text)


def _add_tec22_bounds(data: pd.DataFrame) -> None:
    data["B1_N_Aver"] = data["B1_N"] / 24
    data["B2_N_Aver"] = data["B2_N"] / 24
    data["B3_N_Aver"] = data["B3_N"] / 24
    data["B4_GT41_N_Aver"] = data["B4_GT41_N"] / 24
    data["B4_GT42_N_Aver"] = data["B4_GT42_N"] / 24
    data["B4_N_Aver"] = data["B4_N_Aver"]
    data["B1_inWork"] = np.where(data["B1_N_Aver"] >= 125, 1, 0)
    data["B2_inWork"] = np.where(data["B2_N_Aver"] >= 125, 1, 0)
    data["B3_inWork"] = np.where(data["B3_N_Aver"] >= 125, 1, 0)
    data["B4_GT41_inWork"] = np.where(data["B4_GT41_N_Aver"] >= 50, 1, 0)
    data["B4_GT42_inWork"] = np.where(data["B4_GT42_N_Aver"] >= 50, 1, 0)
    data["B1_Available_Nmax"] = data["B1_inWork"] * 250
    data["B2_Available_Nmax"] = data["B2_inWork"] * 250
    data["B3_Available_Nmax"] = data["B3_inWork"] * 250
    data["B4_GT41_Available_Nmax"] = np.where(
        data["T"] <= -2.3,
        data["B4_GT41_inWork"] * 172.7,
        data["B4_GT41_inWork"] * (-0.9484 * data["T"] + 170.78),
    )
    data["B4_GT42_Available_Nmax"] = np.where(
        data["T"] <= -2.3,
        data["B4_GT42_inWork"] * 172.7,
        data["B4_GT42_inWork"] * (-0.9484 * data["T"] + 170.78),
    )
    for gt in ["GT41", "GT42"]:
        col = f"B4_{gt}_Available_Nmax"
        data[col] = np.where((data["Year"] >= 2024) & (data[col] > 160.0), 160.0, data[col])
    data["B4_Available_Nmax"] = (
        data["B4_GT41_Available_Nmax"]
        + data["B4_GT42_Available_Nmax"]
        + 72 * (data["B4_GT41_inWork"] + data["B4_GT42_inWork"])
    )
    data["B1_Available_Nmin"] = data["B1_inWork"] * 125
    data["B2_Available_Nmin"] = data["B2_inWork"] * 125
    data["B3_Available_Nmin"] = data["B3_inWork"] * 125
    data["B4_Available_Nmin"] = np.where(
        data["T"] <= -2.3,
        0.5 * data["B4_GT41_inWork"] * 172.7
        + 0.5 * data["B4_GT42_inWork"] * 172.7
        + 26 * (data["B4_GT41_inWork"] + data["B4_GT42_inWork"]),
        0.5 * data["B4_GT41_inWork"] * (-0.9484 * data["T"] + 170.78)
        + 0.5 * data["B4_GT42_inWork"] * (-0.9484 * data["T"] + 170.78)
        + 26 * (data["B4_GT41_inWork"] + data["B4_GT42_inWork"]),
    )


def _add_tec14_bounds(data: pd.DataFrame) -> None:
    data["B1_GT11_N_Aver"] = data["B1_GT11_N"] / 24
    data["B1_GT12_N_Aver"] = data["B1_GT12_N"] / 24
    data["B2_GT21_N_Aver"] = data["B2_GT21_N"] / 24
    data["B2_GT22_N_Aver"] = data["B2_GT22_N"] / 24
    data["B1_GT11_inWork"] = np.where(data["B1_GT11_N_Aver"] >= 30, 1, 0)
    data["B1_GT12_inWork"] = np.where(data["B1_GT12_N_Aver"] >= 30, 1, 0)
    data["B2_GT21_inWork"] = np.where(data["B2_GT21_N_Aver"] >= 30, 1, 0)
    data["B2_GT22_inWork"] = np.where(data["B2_GT22_N_Aver"] >= 30, 1, 0)
    temp_curve = (
        2 / 1000000000 * data["T"] ** 6
        + 7 / 1000000000 * data["T"] ** 5
        - 4 / 1000000 * data["T"] ** 4
        - 0.0001 * data["T"] ** 3
        + 0.0006 * data["T"] ** 2
        - 0.2475 * data["T"]
        + 69.719
    )
    data["B1_Available_Nmax"] = (
        data["B1_GT11_inWork"] * temp_curve
        + data["B1_GT12_inWork"] * temp_curve
        + 26 * (data["B1_GT11_inWork"] + data["B1_GT12_inWork"])
    )
    data["B2_Available_Nmax"] = (
        data["B2_GT21_inWork"] * temp_curve
        + data["B2_GT22_inWork"] * temp_curve
        + 26 * (data["B2_GT21_inWork"] + data["B2_GT22_inWork"])
    )
    data["B1_Available_Nmin"] = (
        data["B1_GT11_inWork"] * 0.4 * temp_curve
        + data["B1_GT12_inWork"] * 0.4 * temp_curve
        + 15 * (data["B1_GT11_inWork"] + data["B1_GT12_inWork"])
    )
    data["B2_Available_Nmin"] = (
        data["B2_GT21_inWork"] * 0.4 * temp_curve
        + data["B2_GT22_inWork"] * 0.4 * temp_curve
        + 15 * (data["B2_GT21_inWork"] + data["B2_GT22_inWork"])
    )


def _load_calibration_residuals(
    reports_dir: Path,
    model_name: str,
    unit_ids: list[str],
    n_units: int,
    log_lines: list[str],
) -> pd.DataFrame:
    station_path = reports_dir / "Power_Station_calibration_predictions_comparison.csv"
    unit_paths = [reports_dir / f"{unit}_calibration_predictions_comparison.csv" for unit in unit_ids]
    if station_path.exists() and all(path.exists() for path in unit_paths):
        station_df = pd.read_csv(station_path)
        unit_dfs = [pd.read_csv(path) for path in unit_paths]
        if model_name in station_df.columns and all(model_name in df.columns for df in unit_dfs):
            n_rows = min([len(station_df), *[len(df) for df in unit_dfs]])
            residual_data = {
                "N_chp_error": station_df["Actual_N"].tail(n_rows).to_numpy(dtype=float)
                - station_df[model_name].tail(n_rows).to_numpy(dtype=float)
            }
            for idx, unit_df in enumerate(unit_dfs, start=1):
                residual_data[f"N_{idx}_error"] = (
                    unit_df["Actual_N"].tail(n_rows).to_numpy(dtype=float)
                    - unit_df[model_name].tail(n_rows).to_numpy(dtype=float)
                )
            return pd.DataFrame(residual_data).replace([np.inf, -np.inf], np.nan).dropna()
        log_lines.append(f"WARNING: calibration predictions found, but model={model_name} is missing in one or more files.")

    cols = ["N_chp_error"] + [f"N_{idx}_error" for idx in range(1, n_units + 1)]
    log_lines.append(
        f"WARNING: calibration predictions for model={model_name} were not found; "
        "using unit covariance fallback for WLS/MinT to avoid test leakage."
    )
    return pd.DataFrame(np.eye(len(cols)), columns=cols)


def _rename_calibration_columns(df: pd.DataFrame, unit_ids: list[str]) -> pd.DataFrame:
    renamed = df.copy()
    for idx, unit in enumerate(unit_ids, start=1):
        renamed[f"N_{idx}_true"] = renamed[f"{unit}_N_Aver"]
    return renamed


def _detect_units(reports_dir: Path) -> list[str]:
    units = []
    for path in sorted(reports_dir.glob("B*_final_predictions_comparison.xlsx")):
        units.append(path.name.split("_", 1)[0])
    return units


def _select_model_name(reports_dir: Path, configured: str, station_df: pd.DataFrame) -> str:
    if configured != "best":
        if configured not in station_df.columns:
            raise ValueError(f"Configured model_name not found in reports: {configured}")
        return configured
    report = pd.read_csv(reports_dir / "Power_Station_model_evaluation_report.csv")
    best = str(report.iloc[0]["Model Name"])
    if best not in station_df.columns:
        raise ValueError(f"Best model from report is missing in predictions: {best}")
    return best


def _select_model_names_for_run(reports_dir: Path, configured: str) -> list[str]:
    station_df = pd.read_excel(reports_dir / "Power_Station_final_predictions_comparison.xlsx")
    units = _detect_units(reports_dir)
    common = set(station_df.columns) - {"Actual_N"}
    for unit in units:
        unit_df = pd.read_excel(reports_dir / f"{unit}_final_predictions_comparison.xlsx", nrows=1)
        common &= set(unit_df.columns) - {"Actual_N"}
    ordered = [col for col in station_df.columns if col in common and col != "Actual_N"]
    if configured == "all":
        return ordered
    if configured == "best":
        return [_select_model_name(reports_dir, configured, station_df)]
    selected = [part.strip() for part in str(configured).split(",") if part.strip()]
    missing = [name for name in selected if name not in ordered]
    if missing:
        raise ValueError(f"Configured model_name values are missing in reports: {missing}")
    return selected


def _save_predictions(pred_dir: Path, plant_id: str, model_name: str, method: str, df: pd.DataFrame) -> None:
    clean_model = str(model_name).replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    path = pred_dir / f"{plant_id}_{clean_model}_{method}_predictions.csv"
    df.to_csv(path, index=False)


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _plant_id_from_data_file(data_file: Path) -> str:
    if data_file.name == "TEC22_Data.csv":
        return "TEC22"
    if data_file.name == "TEC14_Data.csv":
        return "TEC14"
    return data_file.stem


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            result.setdefault(current_key, []).append(_coerce_value(line[4:].strip()))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            if not value:
                result[key] = []
            else:
                result[key] = _coerce_value(value)
    return result


def _coerce_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return json.loads(value)
    except Exception:
        try:
            return float(value)
        except ValueError:
            return value.strip("\"'")
