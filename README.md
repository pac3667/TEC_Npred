# TEC_Npred: Dispatch-Ready CHP Load Forecasting

TEC_Npred is a forecasting pipeline for daily electrical load prediction at combined heat and power (CHP) plants and their individual power units. The project focuses on dispatch-ready hierarchical forecasting: the final forecast must be accurate, structurally coherent across the plant and unit levels, and feasible under time-varying operational limits.

The current implementation supports TEC-14 and TEC-22 datasets, model comparison across classical ML and neural forecasting approaches, physical reconciliation with SLSQP, and JSON command-line execution for use as an external utility.

## Research Context

The project follows the formulation described in the KDD article draft, "Dispatch-Ready Hierarchical Electrical Load Forecasting for Power Plants". The central idea is that a coherent forecast is not necessarily dispatch-ready. Classical hierarchical reconciliation can make the plant forecast equal to the sum of unit forecasts, but it does not guarantee that each unit forecast lies within the admissible operating range.

A dispatch-ready forecast must satisfy three requirements:

1. Forecast accuracy: predicted values should be close to observed CHP-level and unit-level loads.
2. Structural coherence: the reconciled CHP-level forecast must match the sum of reconciled unit-level forecasts.
3. Operational feasibility: every unit-level forecast must respect the corresponding time-varying lower and upper power limits.

Operational limits are derived from equipment state, availability, weather-dependent capacity, and engineering rules. They are used as model features, in constraint-aware neural learning, and as hard bounds in the final physical reconciliation layer.

## Forecasting Pipeline

The forecasting workflow is orchestrated by `forecast_runner.py` and `calc_body.py`.

1. Data preprocessing and feature engineering:
   - parses daily operational telemetry;
   - removes non-operating or incomplete intervals;
   - builds calendar, weather, lagged, equipment-state, and operational-limit features;
   - computes available minimum and maximum power for each unit and the CHP plant.

2. Base forecasting:
   - trains and evaluates station-level and unit-level forecasts;
   - compares Linear Regression, Polynomial Regression, Random Forest, CatBoost Gradient Boosting, MLP, LSTM, and window-based variants;
   - uses Optuna for hyperparameter search and stores reusable parameters and checkpoints under `checkpoint/`.

3. Hierarchical unit modeling:
   - uses the CHP-level forecast as a coordinating signal for unit-level forecasts when hierarchical features are enabled;
   - supports direct multi-step forecasting through CatBoost and LSTM direct-forecast modules.

4. Physical reconciliation:
   - applies an SLSQP-based optimization layer (`reconcile_with_l2`) to transform base forecasts into dispatch-ready forecasts;
   - minimizes the squared correction from the original unit-level forecasts;
   - enforces unit-level operating bounds;
   - enforces plant-unit balance;
   - clips the CHP target to the feasible intersection of CHP-level limits and the aggregate unit capacity.

## Reconciliation Approaches Studied

The research compares multiple strategies for hierarchical forecasting and reconciliation:

- Independent forecasts: CHP and unit forecasts are produced separately, without enforcing balance.
- Bottom-Up: unit forecasts are preserved and aggregated to the CHP level.
- Top-Down: the CHP forecast is preserved and distributed to units.
- OLS reconciliation: classical least-squares reconciliation for structural coherence.
- WLS reconciliation: weighted least-squares reconciliation using residual statistics.
- MinT reconciliation: minimum-trace reconciliation using forecast-error covariance structure.
- Proposed physical reconciliation: SLSQP-based correction with both structural coherence and operational bounds.

The article draft reports that classical reconciliation methods can eliminate hierarchical inconsistency, but may still produce physically infeasible unit schedules. The proposed SLSQP-based physical reconciliation is designed to produce forecasts that are coherent and operationally feasible while remaining close to the original model outputs.

## Ablation Study

The ablation study evaluates the contribution of pipeline components by adding them sequentially:

- A0: baseline independent raw forecasts.
- A1: A0 with constraint-aware loss and clipping.
- A2: A1 with hierarchical features.
- A3: A2 with SLSQP-based reconciliation.

The main findings reflected in the project design are:

- Constraint-aware learning and clipping mainly reduce operational constraint violations before final reconciliation.
- Hierarchical features provide CHP-level operating context to unit-level models, but do not enforce structural coherence by themselves.
- SLSQP-based reconciliation is the key stage that removes hierarchical imbalance and produces dispatch-ready outputs.

## Evaluation Metrics

The pipeline evaluates forecast quality using both standard accuracy metrics and dispatch-oriented checks.

Accuracy metrics:

- MAE;
- RMSE;
- WAPE;
- R2.

Dispatch-oriented evaluation used in the research:

- HIC: hierarchical incoherence between the CHP-level forecast and the sum of unit-level forecasts.
- OCV: operational constraint violations below minimum or above maximum unit power.
- RD: reconciliation deviation, measuring the correction introduced by reconciliation.

The JSON `predict-score` command currently reports MAE, RMSE, WAPE, and R2. Metrics are attached to each forecast row starting from the second row: row 2 contains metrics for day 1, row 3 contains metrics for day 2, and so on.

## Project Structure

```text
TEC_Npred/
|-- calc_body.py                 # Core forecasting engine
|-- forecast_runner.py           # Shared runner used by main.py and tec_cli.py
|-- main.py                      # CSV-based research pipeline entry point
|-- tec_cli.py                   # JSON command-line interface
|-- print_results.py             # Model reports, plots, and comparison output
|-- utils.py                     # Preprocessing, metrics, constraints, reconciliation
|-- requirements.txt             # Python dependencies
|-- checkpoint/                  # Saved model parameters and Keras checkpoints
|-- data/
|   |-- TEC14_Data.csv           # TEC-14 input dataset
|   |-- TEC22_Data.csv           # TEC-22 input dataset
|   `-- reports/                 # Generated reports
|-- forecast_models/
|   |-- CBRDirectForecast.py
|   |-- CBRmetaDirectForecast.py
|   |-- LSTMDirectForecast.py
|   |-- LSTMmetaDirectForecast.py
|   `-- models.py
`-- notebooks/                   # Research notebooks and prototyping artifacts
```

## Installation

Python 3.10 or 3.11 is recommended. TensorFlow support on native Windows depends on the installed TensorFlow version; CPU execution is supported.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CSV Research Run

The traditional CSV pipeline is available through `main.py`.

```bash
python main.py --file data/TEC14_Data.csv --start 2000 --forecast_window 14 --target_power_unit B1 B2
```

Common arguments:

- `--file`: path to `TEC14_Data.csv` or `TEC22_Data.csv`;
- `--start`: index where the test period begins;
- `--forecast_window`: direct multi-step forecasting horizon;
- `--target_power_unit`: unit identifiers, for example `B1 B2` for TEC-14 or `B1 B2 B3 B4` for TEC-22;
- `--hierarchical_features`: enables or disables CHP-level hierarchical features;
- `--cliping_and_customLoss`: controls clipping and custom-loss behavior retained from the research code.

## JSON Command-Line Interface

`tec_cli.py` is intended for running the project as an external utility without starting an HTTP server.

### Command 1: Forecast Only

```bash
python tec_cli.py predict --input input.json --output output.json
```

The output contains dispatch-ready forecasts for the CHP plant and each requested unit. Each entity includes:

- `forecast_power`;
- `available_min`;
- `available_max`.

### Command 2: Forecast and Score Previous Days

```bash
python tec_cli.py predict-score --input input.json --output output.json
```

The command produces the same reconciled forecasts and adds shifted daily metrics:

- the first forecast row has no metrics;
- the second row contains metrics for the first forecast day;
- the third row contains metrics for the second forecast day;
- this continues until the end of the result period.

For a single entity on a single day, `R2` is mathematically undefined and is returned as `null`. Aggregated summaries, such as all units or all entities, contain `R2` when at least two values are available and the actual values have non-zero variance.

### Input JSON

The input JSON must contain:

- `station`: `TEC14` or `TEC22`;
- `records`: daily telemetry records with the same fields as the corresponding test dataset;
- optional `start`;
- optional `forecast_window`;
- optional `target_power_unit`;
- optional `hierarchical_features`;
- optional `cliping_and_customLoss`.

Example:

```json
{
  "station": "TEC14",
  "start": 2000,
  "forecast_window": 1,
  "target_power_unit": ["B1", "B2"],
  "hierarchical_features": 1,
  "cliping_and_customLoss": 1,
  "records": [
    {
      "Date": "01.01.2019",
      "B1_N": 0,
      "B1_Q": 0,
      "B2_N": 0,
      "B2_Q": 0,
      "OVK_Q": 0,
      "B1_N_Aver": 0,
      "B2_N_Aver": 0,
      "T": -5.0,
      "TEC_Q_Aver": 0,
      "TEC_N_Aver": 0,
      "B1_GT11_N": 0,
      "B1_GT12_N": 0,
      "B2_GT21_N": 0,
      "B2_GT22_N": 0,
      "B1_PT10_N": 0,
      "B2_PT20_N": 0
    }
  ]
}
```

### Output JSON

Example output fragment:

```json
{
  "status": "ok",
  "command": "predict-score",
  "station": "TEC14",
  "forecast_window": 1,
  "metrics_mode": "previous_day_per_result_row",
  "results": [
    {
      "date": "2024-06-23",
      "entities": {
        "TEC": {
          "forecast_power": 268.42,
          "available_min": 162.94,
          "available_max": 361.36
        },
        "B1": {
          "forecast_power": 137.24,
          "available_min": 90.0,
          "available_max": 192.0
        },
        "B2": {
          "forecast_power": 131.18,
          "available_min": 90.0,
          "available_max": 192.0
        }
      }
    },
    {
      "date": "2024-06-24",
      "entities": {
        "TEC": {
          "forecast_power": 271.10,
          "available_min": 162.94,
          "available_max": 361.36
        },
        "B1": {
          "forecast_power": 138.00,
          "available_min": 90.0,
          "available_max": 192.0
        },
        "B2": {
          "forecast_power": 133.10,
          "available_min": 90.0,
          "available_max": 192.0
        }
      },
      "metrics_for_previous_day": {
        "date": "2024-06-23",
        "station": {
          "TEC": {
            "MAE": 5.78,
            "RMSE": 5.78,
            "WAPE": 2.11,
            "R2": null
          }
        },
        "units": {
          "by_entity": {
            "B1": {
              "MAE": 3.90,
              "RMSE": 3.90,
              "WAPE": 2.79,
              "R2": null
            },
            "B2": {
              "MAE": 5.94,
              "RMSE": 5.94,
              "WAPE": 4.42,
              "R2": null
            }
          },
          "summary": {
            "MAE": 4.92,
            "RMSE": 5.10,
            "WAPE": 3.61,
            "R2": 0.98
          }
        },
        "summary": {
          "MAE": 5.21,
          "RMSE": 5.36,
          "WAPE": 2.94,
          "R2": 0.99
        }
      }
    }
  ]
}
```

## Outputs and Artifacts

Depending on the entry point, the pipeline can generate:

- per-model evaluation reports in `data/reports/{station_name}/`;
- model comparison tables and plots;
- saved Optuna parameter files and Keras checkpoints under `checkpoint/`;
- reconciled forecasts with available power ranges;
- JSON outputs for external utility execution.

The final JSON forecasts are post-reconciliation values. Disabled units are forced to zero when their admissible range is `[0, 0]`, working units are kept inside their power limits, and the reconciled unit sum matches the reconciled CHP forecast up to numerical tolerance.
