import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf

from forecast_models.CBRmetaDirectForecast import train_cbr_meta_direct_multistep
from forecast_models.LSTMmetaDirectForecast import train_lstm_meta_direct_multistep
from utils import prepare_stat_data, prepare_window_data, prepare_meta_data, prepare_meta_window_data
from print_results import print_stat_results, print_step_results, plot_compare_models_violations, \
    plot_single_model_violations, save_calibration_predictions, save_step_predictions
from sklearn.preprocessing import PolynomialFeatures
from forecast_models.CBRDirectForecast import train_cbr_direct_multistep
from forecast_models.LSTMDirectForecast import train_lstm_direct_multistep
from forecast_models.models import (
    get_catboost, get_linear, get_lstm,
    get_mlp, get_rfr
)
from tensorflow.keras.callbacks import ModelCheckpoint

tf.random.set_seed(42)

def calc_power_generation(data_path, test_start_index, checkpoint_dir, n_out, calc_goal, results, test_idx, best_window_model_name, best_stat_model_name, best_step_model, reports_dir, hierarchical_features=1, cliping_and_customLoss=1, calibration_start_index=None, skip_direct_forecast=False):
    if calc_goal == 'TEC':
        x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, datas, test_idx,  y_train_s_combined, y_test_s_combined, y_test_combined = prepare_stat_data(data_path, test_start_index, cliping_and_customLoss)
        xw_train_s, xw_test_s, yw_train_s, yw_test_s, yw_test, scaler_yw, datasw, testw_idx,  yw_train_s_combined, yw_test_s_combined, yw_test_combined = prepare_window_data(data_path, test_start_index, cliping_and_customLoss)
    else:
        x_train_s, x_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, datas, test_idx,  y_train_s_combined, y_test_s_combined, y_test_combined = prepare_meta_data(results, data_path, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model, hierarchical_features, cliping_and_customLoss)
        xw_train_s, xw_test_s, yw_train_s, yw_test_s, yw_test, scaler_yw, datasw, testw_idx,  yw_train_s_combined, yw_test_s_combined, yw_test_combined = prepare_meta_window_data(results, data_path, test_idx, test_start_index, calc_goal, best_window_model_name, best_stat_model_name, best_step_model, hierarchical_features, cliping_and_customLoss)
    results = {}
    calibration_results = {}
    constraints = {calc_goal + '_Available_Nmin': scaler_y.inverse_transform(y_test_s_combined[:, 2][:, None]),
                   calc_goal + '_Available_Nmax': scaler_y.inverse_transform(y_test_s_combined[:, 1][:, None])}

    cal_start = _normalize_calibration_start(calibration_start_index, x_train_s.shape[0])
    has_calibration = cal_start is not None
    if has_calibration:
        x_fit_s, x_cal_s = x_train_s[:cal_start], x_train_s[cal_start:]
        y_fit_s, y_cal_s = y_train_s[:cal_start], y_train_s[cal_start:]
        y_fit, y_cal = y_train[:cal_start], y_train[cal_start:]
        y_fit_s_combined = y_train_s_combined[:cal_start]
        y_cal_s_combined = y_train_s_combined[cal_start:]
        y_cal_combined = scaler_y.inverse_transform(y_cal_s_combined[:, 0:1])

        xw_fit_s, xw_cal_s = xw_train_s[:cal_start], xw_train_s[cal_start:]
        yw_fit_s, yw_cal_s = yw_train_s[:cal_start], yw_train_s[cal_start:]
        yw_fit_s_combined = yw_train_s_combined[:cal_start]
        yw_cal_s_combined = yw_train_s_combined[cal_start:]
        yw_cal = scaler_yw.inverse_transform(yw_cal_s)
        yw_cal_combined = scaler_yw.inverse_transform(yw_cal_s_combined[:, 0:1])
    else:
        x_fit_s, y_fit_s, y_fit, y_fit_s_combined = x_train_s, y_train_s, y_train, y_train_s_combined
        x_cal_s = y_cal_s = y_cal = y_cal_s_combined = y_cal_combined = None
        xw_fit_s, yw_fit_s, yw_fit_s_combined = xw_train_s, yw_train_s, yw_train_s_combined
        xw_cal_s = yw_cal_s = yw_cal = yw_cal_s_combined = yw_cal_combined = None

    # ---  LSTM ---
    print('LSTM_Stat_Model')
    x_train_lstm = x_fit_s.reshape((x_fit_s.shape[0], 1, x_fit_s.shape[1]))
    x_test_lstm = x_test_s.reshape((x_test_s.shape[0], 1, x_test_s.shape[1]))
    x_val_lstm = x_cal_s.reshape((x_cal_s.shape[0], 1, x_cal_s.shape[1])) if has_calibration else x_test_lstm
    y_val_s = y_cal_s if has_calibration else y_test_s
    y_val_s_combined = y_cal_s_combined if has_calibration else y_test_s_combined

    loss_types = _get_loss_types(cliping_and_customLoss)

    for loss_type in loss_types:
        print(f"\n--- Processing LSTM with {loss_type.upper()} loss ---")

        model_lstm, early_stop_callback, current_epochs, checkpoint_filepath = get_lstm(
            (1, x_fit_s.shape[1]), x_train_lstm, x_val_lstm, y_fit_s, y_val_s, y_fit, y_cal if has_calibration else y_test,
            scaler_y, y_fit_s_combined, y_val_s_combined, checkpoint_dir, calc_goal, loss_type=loss_type
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=0
        )

        model_lstm.fit(
            x_train_lstm, y_fit_s_combined,
            epochs=current_epochs,
            batch_size=1024,
            validation_data=(x_val_lstm, y_val_s_combined),
            callbacks=[early_stop_callback, model_checkpoint_callback],
            verbose=0,
            shuffle=False
        )

        if has_calibration:
            yhat_cal_s = model_lstm.predict(x_val_lstm)
            yhat_cal = scaler_y.inverse_transform(yhat_cal_s)
            if loss_type == 'custom':
                yhat_cal = _apply_bounds(yhat_cal, y_train_s_combined[cal_start:, 2], y_train_s_combined[cal_start:, 1], scaler_y)
            calibration_results[f'LSTM_{loss_type}'] = yhat_cal

        yhat_s = model_lstm.predict(x_test_lstm)
        yhat = scaler_y.inverse_transform(yhat_s)

        n_max = y_test_combined[:, 1][:, None]
        n_min = y_test_combined[:, 2][:, None]

        if loss_type == 'custom':
            yhat[yhat < n_min] = 0
            yhat = np.where(yhat > n_max, n_max, yhat)
            yhat = np.maximum(yhat, 0)

        over_max_mask = yhat > n_max
        count_over_max = np.sum(over_max_mask)
        mean_over_max = np.mean(yhat[over_max_mask] - n_max[over_max_mask]) if count_over_max > 0 else 0

        under_min_mask = yhat < n_min
        count_under_min = np.sum(under_min_mask)
        mean_under_min = np.mean(n_min[under_min_mask] - yhat[under_min_mask]) if count_under_min > 0 else 0

        print(f"\n[{loss_type.upper()} Loss] Bounds violations summary:")
        print(f"  -> Upper bound violations: {count_over_max} times (Mean excess: {mean_over_max:.4f})")
        print(f"  -> Lower bound violations: {count_under_min} times (Mean deficit: {mean_under_min:.4f})")
        # ------------------------------------------


        results[f'LSTM_{loss_type}'] = yhat

    # ---  Linear Regression ---
    print('LR_Stat_Model')
    model_lr = get_linear()
    model_lr.fit(x_fit_s, y_fit)
    if has_calibration:
        yhat_cal = model_lr.predict(x_cal_s)
        yhat_cal = _clip_predictions(yhat_cal, cliping_and_customLoss, scaler_y.inverse_transform(y_train_s_combined[cal_start:, 2][:, None]), scaler_y.inverse_transform(y_train_s_combined[cal_start:, 1][:, None]))
        calibration_results['Linear'] = yhat_cal
    yhat = model_lr.predict(x_test_s)
    results['Linear'] = _clip_predictions(yhat, cliping_and_customLoss, y_test_combined[:, 2][:, None], y_test_combined[:, 1][:, None])

    # ---  poly Regression ---
    print('LRpoly_Stat_Model')
    poly = PolynomialFeatures(2, include_bias=False)
    x_train_poly = poly.fit_transform(x_fit_s)
    x_test_poly = poly.transform(x_test_s)
    x_cal_poly = poly.transform(x_cal_s) if has_calibration else None

    model_poly = get_linear()
    model_poly.fit(x_train_poly, y_fit)
    if has_calibration:
        yhat_cal = model_poly.predict(x_cal_poly)
        yhat_cal = _clip_predictions(yhat_cal, cliping_and_customLoss, scaler_y.inverse_transform(y_train_s_combined[cal_start:, 2][:, None]), scaler_y.inverse_transform(y_train_s_combined[cal_start:, 1][:, None]))
        calibration_results['Polynomial (d=2)'] = yhat_cal
    yhat = model_poly.predict(x_test_poly)
    results['Polynomial (d=2)'] = _clip_predictions(yhat, cliping_and_customLoss, y_test_combined[:, 2][:, None], y_test_combined[:, 1][:, None])

    # ---  Gradient Boosting ---
    print('CBR_Stat_Model')
    model_cb = get_catboost(x_fit_s, x_cal_s if has_calibration else x_test_s, y_fit_s, y_cal_s if has_calibration else y_test_s, y_fit, y_cal if has_calibration else y_test, scaler_y)
    model_cb.fit(x_fit_s, y_fit_s, eval_set=(x_cal_s if has_calibration else x_test_s, y_cal_s if has_calibration else y_test_s), early_stopping_rounds=250, use_best_model=True)
    if has_calibration:
        yhat_cal_s = model_cb.predict(x_cal_s).reshape(-1, 1)
        yhat_cal = scaler_y.inverse_transform(yhat_cal_s)
        yhat_cal = _clip_predictions(yhat_cal, cliping_and_customLoss, scaler_y.inverse_transform(y_train_s_combined[cal_start:, 2][:, None]), scaler_y.inverse_transform(y_train_s_combined[cal_start:, 1][:, None]))
        calibration_results['Boosting'] = yhat_cal
    yhat_s = model_cb.predict(x_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)
    results['Boosting'] = _clip_predictions(yhat, cliping_and_customLoss, y_test_combined[:, 2][:, None], y_test_combined[:, 1][:, None])

    # ---  MLP ---
    print('MLP_Stat_Model')

    checkpoint_filepath = checkpoint_dir + calc_goal + '_MLP.keras'
    params_filepath = checkpoint_dir + calc_goal + '_MLP_params.json'
    model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False,
                                                monitor='val_loss', mode='min', save_best_only=True, verbose=0)

    model_mlp, early_stop_callback, current_epochs = get_mlp((x_fit_s.shape[1],), x_fit_s, x_cal_s if has_calibration else x_test_s, y_fit_s, y_cal_s if has_calibration else y_test_s, y_fit, y_cal if has_calibration else y_test, scaler_y, y_fit_s_combined, y_cal_s_combined if has_calibration else y_test_s_combined, checkpoint_filepath, params_filepath)

    model_mlp.fit(x_fit_s ,
                y_fit_s_combined,
                epochs=current_epochs,
                batch_size=1024,
                validation_data=(x_cal_s if has_calibration else x_test_s, y_cal_s_combined if has_calibration else y_test_s_combined),
                validation_batch_size=1024,
                callbacks=[early_stop_callback,model_checkpoint_callback],
                verbose=0,
                shuffle=False)

    if has_calibration:
        yhat_cal_s = model_mlp.predict(x_cal_s)
        yhat_cal = scaler_y.inverse_transform(yhat_cal_s)
        yhat_cal = _clip_predictions(yhat_cal, cliping_and_customLoss, scaler_y.inverse_transform(y_train_s_combined[cal_start:, 2][:, None]), scaler_y.inverse_transform(y_train_s_combined[cal_start:, 1][:, None]))
        calibration_results['MLP'] = yhat_cal
    yhat_s = model_mlp.predict(x_test_s)
    yhat = scaler_y.inverse_transform(yhat_s)
    results['MLP'] = _clip_predictions(yhat, cliping_and_customLoss, y_test_combined[:, 2][:, None], y_test_combined[:, 1][:, None])

    # ---  Random Forest Regression---
    print('RFR_Stat_Model')
    model_rfr = get_rfr(x_fit_s, x_cal_s if has_calibration else x_test_s, y_fit_s, y_cal_s if has_calibration else y_test_s, y_fit, y_cal if has_calibration else y_test, scaler_y)
    model_rfr.fit(x_fit_s, y_fit_s)
    if has_calibration:
        yhat_cal_s = model_rfr.predict(x_cal_s).reshape(-1, 1)
        yhat_cal = scaler_y.inverse_transform(yhat_cal_s)
        yhat_cal = _clip_predictions(yhat_cal, cliping_and_customLoss, scaler_y.inverse_transform(y_train_s_combined[cal_start:, 2][:, None]), scaler_y.inverse_transform(y_train_s_combined[cal_start:, 1][:, None]))
        calibration_results['RandomForest'] = yhat_cal
    yhat_s = model_rfr.predict(x_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)
    results['RandomForest'] = _clip_predictions(yhat, cliping_and_customLoss, y_test_combined[:, 2][:, None], y_test_combined[:, 1][:, None])

    # ---  Gradient booster with window ---
    print('CBR_with_window_Stat_Model')
    model_cbw = get_catboost(xw_fit_s, xw_cal_s if has_calibration else xw_test_s, yw_fit_s, yw_cal_s if has_calibration else yw_test_s, y_fit, yw_cal if has_calibration else yw_test, scaler_yw)
    model_cbw.fit(xw_fit_s, yw_fit_s, eval_set=(xw_cal_s if has_calibration else xw_test_s, yw_cal_s if has_calibration else yw_test_s), early_stopping_rounds=250, use_best_model=True)
    if has_calibration:
        yhat_cal_s = model_cbw.predict(xw_cal_s).reshape(-1, 1)
        yhat_cal = scaler_yw.inverse_transform(yhat_cal_s)
        yhat_cal = _clip_predictions(yhat_cal, cliping_and_customLoss, scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 2][:, None]), scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 1][:, None]))
        calibration_results['Boosting_custom_with_window'] = yhat_cal
    yhat_s = model_cbw.predict(xw_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)

    results['Boosting_custom_with_window'] = _clip_predictions(yhat, cliping_and_customLoss, yw_test_combined[:, 2][:, None], yw_test_combined[:, 1][:, None])

    # ---  LSTM with window ---
    print('LSTM_with_window_Stat_Model Evaluation')

    xw_train_lstm = xw_fit_s.reshape((xw_fit_s.shape[0], 1, xw_fit_s.shape[1]))
    xw_test_lstm = xw_test_s.reshape((xw_test_s.shape[0], 1, xw_test_s.shape[1]))
    xw_val_lstm = xw_cal_s.reshape((xw_cal_s.shape[0], 1, xw_cal_s.shape[1])) if has_calibration else xw_test_lstm

    loss_types = _get_loss_types(cliping_and_customLoss)

    for loss_type in loss_types:
        print(f"\n--- Processing LSTM with window ({loss_type.upper()} loss) ---")

        model_lstmrw, early_stop_callback, current_epochs, checkpoint_filepath = get_lstm(
            (1, xw_fit_s.shape[1]), xw_train_lstm, xw_val_lstm, yw_fit_s, yw_cal_s if has_calibration else yw_test_s, y_fit, yw_cal if has_calibration else yw_test,
            scaler_yw, yw_fit_s_combined, yw_cal_s_combined if has_calibration else yw_test_s_combined, checkpoint_dir, calc_goal + '_with_RW',
            loss_type=loss_type
        )

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=0
        )

        model_lstmrw.fit(
            xw_train_lstm, yw_fit_s_combined,
            epochs=current_epochs,
            batch_size=1024,
            validation_data=(xw_val_lstm, yw_cal_s_combined if has_calibration else yw_test_s_combined),
            validation_batch_size=1024,
            callbacks=[early_stop_callback, model_checkpoint_callback],
            verbose=0,
            shuffle=False
        )

        if has_calibration:
            yhat_cal_s = model_lstmrw.predict(xw_val_lstm)
            yhat_cal = scaler_yw.inverse_transform(yhat_cal_s)
            if loss_type == 'custom':
                yhat_cal[yhat_cal < scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 2][:, None])] = 0
                yhat_cal = np.where(yhat_cal > scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 1][:, None]), scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 1][:, None]), yhat_cal)
                yhat_cal = np.maximum(yhat_cal, 0)
            calibration_results[f'LSTM_{loss_type}_with_window'] = yhat_cal

        yhat_s = model_lstmrw.predict(xw_test_lstm)
        yhat = scaler_yw.inverse_transform(yhat_s)

        n_max = yw_test_combined[:, 1][:, None]
        n_min = yw_test_combined[:, 2][:, None]

        if loss_type == 'custom':
            yhat[yhat < n_min] = 0
            yhat = np.where(yhat > n_max, n_max, yhat)
            yhat = np.maximum(yhat, 0)

        over_max_mask = yhat > n_max
        count_over_max = np.sum(over_max_mask)
        mean_over_max = np.mean(yhat[over_max_mask] - n_max[over_max_mask]) if count_over_max > 0 else 0

        under_min_mask = yhat < n_min
        count_under_min = np.sum(under_min_mask)
        mean_under_min = np.mean(n_min[under_min_mask] - yhat[under_min_mask]) if count_under_min > 0 else 0

        print(f"[{loss_type.upper()} Loss - Window] Bounds violations summary:")
        print(f"  -> Upper bound violations: {count_over_max} times (Mean excess: {mean_over_max:.4f})")
        print(f"  -> Lower bound violations: {count_under_min} times (Mean deficit: {mean_under_min:.4f})")

        results[f'LSTM_{loss_type}_with_window'] = yhat

    # ---  Random Forest Regression with window---
    print('RFR_with_window_Stat_Model')
    model_rfr = get_rfr(xw_fit_s, xw_cal_s if has_calibration else xw_test_s, yw_fit_s, yw_cal_s if has_calibration else yw_test_s, y_fit, yw_cal if has_calibration else yw_test, scaler_yw)
    model_rfr.fit(xw_fit_s, yw_fit_s)
    if has_calibration:
        yhat_cal_s = model_rfr.predict(xw_cal_s).reshape(-1, 1)
        yhat_cal = scaler_yw.inverse_transform(yhat_cal_s)
        calibration_results['RandomForest_mae_with_window'] = yhat_cal
        yhat_cal_custom = yhat_cal.copy()
        yhat_cal_custom[yhat_cal_custom < scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 2][:, None])] = 0
        yhat_cal_custom = np.where(yhat_cal_custom > scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 1][:, None]), scaler_yw.inverse_transform(yw_train_s_combined[cal_start:, 1][:, None]), yhat_cal_custom)
        calibration_results['RandomForest_custom_with_window'] = np.maximum(yhat_cal_custom, 0)
    yhat_s = model_rfr.predict(xw_test_s).reshape(-1, 1)
    yhat = scaler_y.inverse_transform(yhat_s)
    results['RandomForest_mae_with_window'] = yhat

    yhat[yhat < yw_test_combined[:, 2][:, None]] = 0
    n_max = yw_test_combined[:, 1][:, None]
    yhat = np.where(yhat > n_max, n_max, yhat)
    results['RandomForest_custom_with_window'] = np.maximum(yhat, 0)

    best_window_model_name, best_stat_model_name  = print_stat_results(reports_dir, results, y_test, calc_goal)
    if has_calibration:
        save_calibration_predictions(reports_dir, calibration_results, y_cal, calc_goal)

    plot_single_model_violations(
        y_true_real=y_test,
        y_pred=results[best_stat_model_name],
        n_max=y_test_combined[:, 1][:, None],
        n_min=y_test_combined[:, 2][:, None],
        model_name=best_stat_model_name,
    )

    plot_single_model_violations(
        y_true_real=yw_test,
        y_pred=results[best_window_model_name],
        n_max=yw_test_combined[:, 1][:, None],
        n_min=yw_test_combined[:, 2][:, None],
        model_name=best_window_model_name,
    )

    if skip_direct_forecast:
        if calc_goal == 'TEC':
            best_step_model = np.array([results[best_stat_model_name].flatten()[1:15]])
        print('Direct multistep forecast skipped.')
        return results, constraints, test_idx, best_window_model_name, best_stat_model_name, best_step_model

    # ---  LSTM direct forecast---
    print('LSTM_direct_Stat_Model')
    if calc_goal == 'TEC':
        lstm_multi_results, lstm_multi_metrics = train_lstm_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index, calc_goal, cliping_and_customLoss)
    else :
        lstm_multi_results, lstm_multi_metrics = train_lstm_meta_direct_multistep(data_path, checkpoint_dir, n_out, test_start_index, results,
                                     best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal, hierarchical_features, cliping_and_customLoss)
    lstm_multi_results = np.array(lstm_multi_results).reshape(-1, 14)
    lstm_multi_metrics = np.array(lstm_multi_metrics).reshape(-1, 4)
    # --- Boosting with window direct forecast---
    print('CBR_direct_Stat_Model')
    if calc_goal == 'TEC':
        cbr_multi_results, cbr_multi_metrics = train_cbr_direct_multistep(data_path, checkpoint_dir, calc_goal, n_out, test_start_index, cliping_and_customLoss)
    else:
        cbr_multi_results, cbr_multi_metrics = train_cbr_meta_direct_multistep(data_path, checkpoint_dir, n_out,
                                                                         test_start_index, results,
                                     best_window_model_name, test_idx, best_step_model, best_stat_model_name, calc_goal, hierarchical_features, cliping_and_customLoss)
    cbr_multi_results = np.array(cbr_multi_results).reshape(-1, 14)
    cbr_multi_metrics = np.array(cbr_multi_metrics).reshape(-1, 4)

    best_step_model_name = print_step_results(lstm_multi_metrics, cbr_multi_metrics)
    if best_step_model_name == 'lstm': best_step_model = lstm_multi_results
    if best_step_model_name == 'CBR': best_step_model = cbr_multi_results
    save_step_predictions(
        reports_dir,
        calc_goal,
        lstm_multi_results,
        cbr_multi_results,
        lstm_multi_metrics,
        cbr_multi_metrics,
        best_step_model_name,
    )

    return results, constraints, test_idx, best_window_model_name, best_stat_model_name, best_step_model


def _normalize_calibration_start(calibration_start_index, train_size):
    if calibration_start_index is None:
        return None
    try:
        value = int(calibration_start_index)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value >= train_size:
        return None
    return value


def _get_loss_types(cliping_and_customLoss):
    if cliping_and_customLoss == 0:
        return ['custom']
    if cliping_and_customLoss == 1:
        return ['mae']
    return ['custom', 'mae']


def _apply_bounds(yhat, n_min_scaled, n_max_scaled, scaler_y):
    n_min = scaler_y.inverse_transform(n_min_scaled[:, None])
    n_max = scaler_y.inverse_transform(n_max_scaled[:, None])
    yhat = yhat.copy()
    yhat[yhat < n_min] = 0
    yhat = np.where(yhat > n_max, n_max, yhat)
    return np.maximum(yhat, 0)


def _clip_predictions(yhat, cliping_and_customLoss, n_min, n_max):
    if cliping_and_customLoss != 0:
        return yhat
    yhat = yhat.copy()
    yhat[yhat < n_min] = 0
    yhat = np.where(yhat > n_max, n_max, yhat)
    return np.maximum(yhat, 0)
