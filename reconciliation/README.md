# Reconciliation Baselines

This module adds classical hierarchical forecast reconciliation baselines without replacing the existing TEC_Npred pipeline.

Implemented methods:

- `independent`
- `bottom_up`
- `top_down`
- `ols`
- `wls`
- `mint`

Run:

```bash
python experiments/run_reconciliation_baselines.py --config configs/reconciliation_config.yaml
```

To produce calibration residuals for WLS and MinT, first run the main forecasting pipeline with an explicit calibration boundary:

```bash
python main.py --file data/TEC22_Data.csv --calibration_start 2160 --start 2700
```

This writes `*_calibration_predictions_comparison.csv` files to `data/reports/{plant}/`. WLS and MinT use those calibration predictions to estimate residual variances and covariance matrices without using the final test period.

The runner reads existing `data/reports/{plant}/..._final_predictions_comparison.xlsx` files, builds a unified prediction table, applies the requested classical reconciliation methods, and saves outputs to:

```text
outputs/reconciliation_baselines/predictions/
outputs/reconciliation_baselines/metrics/summary_metrics.csv
outputs/reconciliation_baselines/metrics/unit_metrics.csv
outputs/reconciliation_baselines/logs/run_log.txt
```

Main metrics:

- `OCV`: operational constraint violations against `N_min_i` / `N_max_i`.
- `HIC`: hierarchical inconsistency between station forecast and unit sum.
- `RD`: relative deviation between the raw station forecast and reconciled station forecast.
- `MAE`, `RMSE`, `MAPE`: forecast accuracy at station and unit levels.

Set `model_name: all` in `configs/reconciliation_config.yaml` to run every model column that is present in the station and all unit prediction reports. Use one model name, or comma-separated names, to restrict the run.
