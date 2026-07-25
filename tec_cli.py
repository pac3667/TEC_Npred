import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
EPS = 1e-8

STATION_FILES = {
    "TEC14": "TEC14_Data.csv",
    "TEC22": "TEC22_Data.csv",
}

DEFAULT_UNITS = {
    "TEC14": ["B1", "B2"],
    "TEC22": ["B1", "B2", "B3", "B4"],
}

REQUIRED_FIELDS = {
    "TEC14": [
        "Date",
        "B1_N",
        "B1_Q",
        "B2_N",
        "B2_Q",
        "OVK_Q",
        "B1_N_Aver",
        "B2_N_Aver",
        "T",
        "TEC_Q_Aver",
        "TEC_N_Aver",
        "B1_GT11_N",
        "B1_GT12_N",
        "B2_GT21_N",
        "B2_GT22_N",
        "B1_PT10_N",
        "B2_PT20_N",
    ],
    "TEC22": [
        "Date",
        "Qmsk",
        "Qmsk_positive",
        "Qsof",
        "Qsof_positive",
        "Qfrunz",
        "Qfrunz_positive",
        "TEC_Q",
        "B4_N",
        "TEC_N",
        "TEC_Q_Aver",
        "B4_N_Aver",
        "TEC_N_Aver",
        "T",
        "B1_N",
        "B2_N",
        "B3_N",
        "B4_GT41_N",
        "B4_GT42_N",
    ],
}


class CliError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CliError("input_not_found", f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CliError("invalid_json", f"Invalid JSON: {exc}") from exc


def _station(payload):
    value = str(payload.get("station", "")).upper()
    if value not in STATION_FILES:
        raise CliError("invalid_station", "Field 'station' must be one of: TEC14, TEC22")
    return value


def _target_units(payload, station):
    units = payload.get("target_power_unit") or DEFAULT_UNITS[station]
    if not isinstance(units, list) or not all(isinstance(item, str) for item in units):
        raise CliError("invalid_target_power_unit", "Field 'target_power_unit' must be a list of strings")
    allowed = set(DEFAULT_UNITS[station])
    unknown = [unit for unit in units if unit not in allowed]
    if unknown:
        raise CliError("invalid_target_power_unit", f"Unknown units for {station}: {', '.join(unknown)}")
    return units


def _records(payload, station):
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise CliError("missing_records", "Field 'records' must be a non-empty list")
    required = REQUIRED_FIELDS[station]
    for row_idx, row in enumerate(records):
        if not isinstance(row, dict):
            raise CliError("invalid_record", f"records[{row_idx}] must be an object")
        for field in required:
            if field not in row:
                raise CliError("missing_required_field", f"Missing required field: records[{row_idx}].{field}")
    return records


def _write_temp_csv(records, station, tmpdir):
    csv_path = Path(tmpdir) / STATION_FILES[station]
    df = pd.DataFrame(records)
    df = df[REQUIRED_FIELDS[station]]
    df.to_csv(csv_path, sep=";", index=False)
    return csv_path


def _as_float(value, path):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CliError("invalid_numeric_value", f"Field '{path}' must be numeric") from exc
    if not math.isfinite(result):
        raise CliError("invalid_numeric_value", f"Field '{path}' must be finite")
    return result


def _round_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


def _accuracy(actual_values, forecast_values):
    actual = np.asarray(actual_values, dtype=float)
    forecast = np.asarray(forecast_values, dtype=float)
    error = forecast - actual
    denom = float(np.sum(np.abs(actual)))
    r2 = None
    if len(actual) >= 2:
        ss_res = float(np.sum(error**2))
        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
        if ss_tot > EPS:
            r2 = 1.0 - ss_res / ss_tot
    return {
        "MAE": _round_or_none(float(np.mean(np.abs(error)))),
        "RMSE": _round_or_none(float(np.sqrt(np.mean(error**2)))),
        "WAPE": _round_or_none(float(np.sum(np.abs(error)) / denom * 100.0)) if denom > EPS else None,
        "R2": _round_or_none(r2),
    }


def _compute_metrics(payload, station, units):
    previous_day = payload.get("previous_day")
    if not isinstance(previous_day, dict):
        raise CliError("missing_previous_day", "Field 'previous_day' is required for predict-score")
    entities = previous_day.get("entities")
    if not isinstance(entities, dict):
        raise CliError("missing_previous_day_entities", "Field 'previous_day.entities' must be an object")

    all_entities = ["TEC", *units]
    by_entity = {}
    actual_all = []
    forecast_all = []
    for entity in all_entities:
        item = entities.get(entity)
        if not isinstance(item, dict):
            raise CliError("missing_previous_day_entity", f"Field 'previous_day.entities.{entity}' is required")
        forecast = _as_float(item.get("forecast_power"), f"previous_day.entities.{entity}.forecast_power")
        actual = _as_float(item.get("actual_power"), f"previous_day.entities.{entity}.actual_power")
        by_entity[entity] = _accuracy([actual], [forecast])
        actual_all.append(actual)
        forecast_all.append(forecast)

    return {
        "station": {"TEC": by_entity["TEC"]},
        "units": {unit: by_entity[unit] for unit in units},
        "summary": _accuracy(actual_all, forecast_all),
    }


def _actual_power_by_date(records, station, units):
    df = pd.DataFrame(records)
    df["_date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.date.astype(str)

    actuals = {}
    for _, row in df.iterrows():
        entities = {"TEC": _as_float(row["TEC_N_Aver"], "records[].TEC_N_Aver")}
        for unit in units:
            aver_col = f"{unit}_N_Aver"
            raw_col = f"{unit}_N"
            if aver_col in row and pd.notna(row[aver_col]):
                entities[unit] = _as_float(row[aver_col], f"records[].{aver_col}")
            elif raw_col in row and pd.notna(row[raw_col]):
                entities[unit] = _as_float(row[raw_col], f"records[].{raw_col}") / 24.0
            else:
                raise CliError("missing_actual_power", f"Cannot determine actual power for {unit}")
        actuals[row["_date"]] = entities
    return actuals


def _metrics_for_entities(actuals, forecasts, units):
    unit_actual = [actuals[unit] for unit in units]
    unit_forecast = [forecasts[unit] for unit in units]
    all_entities = ["TEC", *units]

    return {
        "station": {
            "TEC": _accuracy([actuals["TEC"]], [forecasts["TEC"]]),
        },
        "units": {
            "by_entity": {
                unit: _accuracy([actuals[unit]], [forecasts[unit]])
                for unit in units
            },
            "summary": _accuracy(unit_actual, unit_forecast),
        },
        "summary": _accuracy(
            [actuals[entity] for entity in all_entities],
            [forecasts[entity] for entity in all_entities],
        ),
    }


def _add_shifted_daily_metrics(result, records, station, units):
    actual_by_date = _actual_power_by_date(records, station, units)
    rows = result["results"]

    for row_index in range(1, len(rows)):
        previous = rows[row_index - 1]
        previous_date = previous["date"]
        if previous_date not in actual_by_date:
            raise CliError("missing_actual_power", f"Actual power is missing for previous day: {previous_date}")

        forecasts = {
            entity: _as_float(values.get("forecast_power"), f"results[{row_index - 1}].entities.{entity}.forecast_power")
            for entity, values in previous["entities"].items()
        }
        rows[row_index]["metrics_for_previous_day"] = {
            "date": previous_date,
            **_metrics_for_entities(actual_by_date[previous_date], forecasts, units),
        }

    result["metrics_mode"] = "previous_day_per_result_row"


def _date_to_json(value):
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)


def _number_to_json(value):
    if pd.isna(value):
        return None
    return float(value)


def _forecast_to_json(forecast_result, station, command, forecast_window):
    df = forecast_result.get("final_report", forecast_result["df_report"])
    units = forecast_result["target_units"]
    rows = []
    for date_value, row in df.iterrows():
        entities = {
            "TEC": {
                "forecast_power": _number_to_json(row.get("TEC_Reconciled", row["TEC_N_Aver_pred"])),
                "available_min": _number_to_json(row["TEC_Available_Nmin"]),
                "available_max": _number_to_json(row["TEC_Available_Nmax"]),
            }
        }
        for unit in units:
            entities[unit] = {
                "forecast_power": _number_to_json(row.get(f"{unit}_Reconciled", row[f"{unit}_N_Aver_pred"])),
                "available_min": _number_to_json(row[f"{unit}_Available_Nmin"]),
                "available_max": _number_to_json(row[f"{unit}_Available_Nmax"]),
            }
        rows.append({"date": _date_to_json(date_value), "entities": entities})

    return {
        "status": "ok",
        "command": command,
        "station": station,
        "forecast_window": forecast_window,
        "results": rows,
    }


def _run(command, payload):
    station = _station(payload)
    units = _target_units(payload, station)
    records = _records(payload, station)
    start = int(payload.get("start", 2000 if station == "TEC14" else 2700))
    forecast_window = int(payload.get("forecast_window", 14))
    hierarchical_features = int(payload.get("hierarchical_features", 1))
    cliping_and_customLoss = int(payload.get("cliping_and_customLoss", 1))

    with tempfile.TemporaryDirectory(prefix="tec_npred_cli_") as tmpdir:
        data_path = _write_temp_csv(records, station, tmpdir)
        from forecast_runner import run_forecast

        forecast_result = run_forecast(
            data_path=str(data_path),
            test_start_index=start,
            n_out=forecast_window,
            target_power_unit=units,
            hierarchical_features=hierarchical_features,
            cliping_and_customLoss=cliping_and_customLoss,
            draw_plots=False,
            save_final_report=False,
        )

    result = _forecast_to_json(forecast_result, station, command, forecast_window)
    if command == "predict-score":
        _add_shifted_daily_metrics(result, records, station, units)
    return result


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="TEC_Npred JSON CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "predict-score"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--input", required=True, help="Path to input JSON")
        sub.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    os.chdir(PROJECT_DIR)

    try:
        payload = _load_json(input_path)
        result = _run(args.command, payload)
        _write_json(output_path, result)
        return 0
    except CliError as exc:
        _write_json(
            output_path,
            {
                "status": "error",
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            },
        )
        return 1
    except Exception as exc:
        _write_json(
            output_path,
            {
                "status": "error",
                "error": {
                    "code": "runtime_error",
                    "message": str(exc),
                },
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
