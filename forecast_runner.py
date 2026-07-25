import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from calc_body import calc_power_generation
from print_results import plot_final_graph
from utils import reconcile_with_l2

tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.random.set_seed(42)

PROJECT_DIR = Path(__file__).resolve().parent


DEFAULT_TARGET_UNITS = {
    "TEC14_Data.csv": ["B1", "B2"],
    "TEC22_Data.csv": ["B1", "B2", "B3", "B4"],
}


def _target_units_for(data_path, target_power_unit):
    if target_power_unit:
        return list(target_power_unit)
    filename = os.path.basename(str(data_path))
    try:
        return DEFAULT_TARGET_UNITS[filename]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset file name: {data_path}") from exc


def _seed_legacy_params(checkpoint_dir, model_name):
    legacy_dir = PROJECT_DIR / "checkpoint" / model_name
    if checkpoint_dir == legacy_dir or not legacy_dir.exists():
        return
    for source in legacy_dir.rglob("*.json"):
        target = checkpoint_dir / source.relative_to(legacy_dir)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def run_forecast(
    data_path,
    test_start_index,
    n_out,
    target_power_unit=None,
    hierarchical_features=1,
    cliping_and_customLoss=1,
    draw_plots=False,
    save_final_report=False,
):
    model_name = os.path.basename(str(data_path)).split('.')[0]
    target_units = _target_units_for(data_path, target_power_unit)
    suffix = f"_hf{hierarchical_features}_ccl{cliping_and_customLoss}"
    checkpoint_path = PROJECT_DIR / "checkpoint" / f"{model_name}{suffix}"
    reports_path = PROJECT_DIR / "data" / "reports" / f"{model_name}{suffix}"
    checkpoint_dir = str(checkpoint_path) + os.sep
    reports_dir = str(reports_path) + os.sep

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    (checkpoint_path / "multistep").mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    _seed_legacy_params(checkpoint_path, model_name)

    tec_results = {}
    results = {}
    test_idx = []
    best_window_model_name = ''
    best_stat_model_name = ''
    best_step_model = []

    warnings.filterwarnings("ignore")

    calc_goal = 'TEC'
    tec_results, tec_constraints, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(
        data_path,
        test_start_index,
        checkpoint_dir,
        n_out,
        calc_goal,
        tec_results,
        test_idx,
        best_window_model_name,
        best_stat_model_name,
        best_step_model,
        reports_dir,
        hierarchical_features,
        cliping_and_customLoss,
    )
    results[calc_goal + '_N_Aver_pred'] = tec_results[best_stat_model_name]
    results[calc_goal + '_Available_Nmin'] = tec_constraints[calc_goal + '_Available_Nmin']
    results[calc_goal + '_Available_Nmax'] = tec_constraints[calc_goal + '_Available_Nmax']

    for calc_goal in target_units:
        print(f"--- Prediction for: {calc_goal} ---")
        tg_results, tg_constraints, test_idx, best_window_model_name, best_stat_model_name, best_step_model = calc_power_generation(
            data_path,
            test_start_index,
            checkpoint_dir,
            n_out,
            calc_goal,
            tec_results,
            test_idx,
            best_window_model_name,
            best_stat_model_name,
            best_step_model,
            reports_dir,
            hierarchical_features,
            cliping_and_customLoss,
        )
        results[calc_goal + '_N_Aver_pred'] = tg_results[best_stat_model_name]
        results[calc_goal + '_Available_Nmin'] = tg_constraints[calc_goal + '_Available_Nmin']
        results[calc_goal + '_Available_Nmax'] = tg_constraints[calc_goal + '_Available_Nmax']
        print(calc_goal, 'best_stat_model_name:', best_stat_model_name)

    flat_results = {k: np.array(v).flatten() for k, v in results.items()}
    min_len = min(len(v) for v in flat_results.values())
    flat_results = {k: v[-min_len:] for k, v in flat_results.items()}
    dates = list(test_idx)[-min_len:]
    df_report = pd.DataFrame(flat_results, index=pd.Index(dates, name="Date"))

    final_report = reconcile_with_l2(
        df_report.reset_index(drop=True),
        tec_col='TEC_N_Aver_pred',
        boiler_prefixes=target_units,
    )
    final_report.index = df_report.index

    if draw_plots:
        plot_final_graph(final_report)

    if save_final_report:
        final_report.to_csv(reports_dir + 'final_tec_report_reconciled.csv', index=False, sep=';')
        print("Файл сохранен: final_tec_report_reconciled.csv")

    return {
        "df_report": df_report,
        "final_report": final_report,
        "target_units": target_units,
        "reports_dir": reports_dir,
    }
