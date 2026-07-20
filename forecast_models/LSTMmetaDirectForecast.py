import gc

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as keras_backend
import os
import optuna
import json
import tensorflow as tf

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from sklearn import metrics
from tensorflow.keras.utils import timeseries_dataset_from_array

from utils import prepare_meta_step_data, prepare_meta_direct_data, custom_loss

tf.random.set_seed(42)


def _build_step_model(params, sequence_length, n_features, model_loss=custom_loss):
    model = Sequential([
        LSTM(params['n_lstm'], input_shape=(sequence_length, n_features), unroll=True, dtype='float16'),
        Dense(params['n_dense'], activation='relu'),
        Dense(1, dtype='float32')
    ])
    model.compile(loss=model_loss, optimizer=Adam(learning_rate=params['lr']), metrics=['mse'])
    return model


def train_lstm_meta_direct_multistep(data, checkpoint_dir, n_out, train_size, results, best_window_model_name, test_idx,
                                     best_step_model, best_stat_model_name, calc_goal, hierarchical_features, cliping_and_customLoss):

    results_list = []
    metrics_list = []

    checkpoint_dir = os.path.join(checkpoint_dir, 'multistep/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_base = os.path.join(checkpoint_dir, f"{calc_goal}_LSTM_LAG_model_step_")
    params_path = os.path.join(checkpoint_dir, f"{calc_goal}_best_params.json")

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            all_steps_params = json.load(f)
    else:
        all_steps_params = {}

    data_with_lag, base_features, weights_train = prepare_meta_step_data(
        data, train_size, n_out, results, best_window_model_name, best_stat_model_name, best_step_model, test_idx,
        calc_goal, hierarchical_features, cliping_and_customLoss
    )

    max_possible_window = 3
    current_batch_size = 4096
    use_custom_loss = hierarchical_features == 0 or cliping_and_customLoss == 0
    model_loss = custom_loss if use_custom_loss else "mae"

    for i in range(n_out):
        step = i + 1
        step_key = str(i)
        current_checkpoint = f"{checkpoint_base}{i}.keras"
        print(f"\n=== Step training {step} ===")

        x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, scaler_y, y_test, y_train_s_combined, y_test_s_combined, y_test_combined = prepare_meta_direct_data(
            data, data_with_lag, step, base_features, train_size, calc_goal, cliping_and_customLoss, hierarchical_features)

        if os.path.exists(current_checkpoint):
            print(f"--- Step {step}: Checkpoint found. Loading... ---")
            saved_window = all_steps_params.get(step_key, {}).get('sequence_length', 1)
            final_sequence_length = saved_window
            try:
                if use_custom_loss:
                    model = load_model(current_checkpoint, custom_objects={'custom_loss': custom_loss})
                else:
                    model = load_model(current_checkpoint)
                model.optimizer.learning_rate.assign(1e-4)
                current_epochs, patience = 50, 10
            except Exception as exc:
                print(f"--- Step {step}: Checkpoint incompatible. Rebuilding from parameters: {exc} ---")
                if step_key not in all_steps_params:
                    raise
                params = all_steps_params[step_key]
                final_sequence_length = params['sequence_length']
                model = _build_step_model(params, final_sequence_length, x_train_scaled.shape[1], model_loss)
                current_epochs, patience = 1000, 100
        else:
            print(f"--- Step {step}: No checkpoint.")
            current_epochs, patience = 1000, 100

            if step_key not in all_steps_params:
                print(f"--- Step {step}: Tuning hyperparameters with Optuna... ---")

                def objective(trial):
                    seq_len = trial.suggest_categorical('sequence_length', [1, 2, 3])

                    n_lstm = trial.suggest_int('n_lstm', 50, 200)
                    n_dense = trial.suggest_int('n_dense', 20, 100)
                    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

                    def set_shapes(x, y):
                        x.set_shape((None, seq_len, x_train_scaled.shape[1]))
                        y.set_shape((None, 3))
                        return x, y

                    train_dataset_optuna = timeseries_dataset_from_array(
                        data=x_train_scaled,
                        targets=y_train_s_combined[seq_len - 1:],
                        sequence_length=seq_len,
                        batch_size=current_batch_size
                    ).map(set_shapes).shuffle(buffer_size=len(x_train_scaled)).prefetch(tf.data.AUTOTUNE)

                    val_dataset_optuna = timeseries_dataset_from_array(
                        data=x_test_scaled,
                        targets=y_test_s_combined[seq_len - 1:],
                        sequence_length=seq_len,
                        batch_size=current_batch_size
                    ).map(set_shapes).prefetch(tf.data.AUTOTUNE)

                    m = Sequential([
                        LSTM(n_lstm, input_shape=(seq_len, x_train_scaled.shape[1]), unroll=True, dtype='float16'),
                        Dense(n_dense, activation='relu'),
                        Dense(1, dtype='float32')
                    ])
                    m.compile(loss=model_loss, optimizer=Adam(learning_rate=lr))

                    m.fit(
                        train_dataset_optuna,
                        validation_data=val_dataset_optuna,
                        epochs=25,
                        verbose=0,
                        callbacks=[optuna.integration.TFKerasPruningCallback(trial, 'val_loss')]
                    )

                    yp_raw = m.predict(val_dataset_optuna, verbose=0)
                    yp_raw = np.nan_to_num(yp_raw, nan=0.0, posinf=0.0, neginf=0.0)
                    yp = scaler_y.inverse_transform(yp_raw)
                    y_test_current_window = y_test[seq_len - 1:]
                    slice_offset = max_possible_window - seq_len
                    yp_aligned = yp[slice_offset:]
                    y_test_aligned = y_test_current_window[slice_offset:]

                    if np.any(np.isnan(yp_aligned)) or np.any(np.isnan(y_test_aligned)):
                        return float('inf')

                    mae = metrics.mean_absolute_error(y_test_aligned, yp_aligned)

                    del m, train_dataset_optuna, val_dataset_optuna
                    keras_backend.clear_session()
                    gc.collect()

                    return mae

                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=100)
                all_steps_params[step_key] = study.best_params
                with open(params_path, 'w') as f:
                    json.dump(all_steps_params, f)

            params = all_steps_params[step_key]
            final_sequence_length = params['sequence_length']

            model = _build_step_model(params, final_sequence_length, x_train_scaled.shape[1], model_loss)

            if i > 0:
                prev_path = f"{checkpoint_base}{i - 1}.keras"
                if os.path.exists(prev_path):
                    try:
                        model.load_weights(prev_path, by_name=True, skip_mismatch=True)
                        print(f"--- Step {step}: Weights initialized from step {i} ---")
                    except: pass

        def set_final_shapes(x, y):
            x.set_shape((None, final_sequence_length, x_train_scaled.shape[1]))
            y.set_shape((None, 3))
            return x, y

        train_dataset_full = timeseries_dataset_from_array(
            data=x_train_scaled,
            targets=y_train_s_combined[final_sequence_length - 1:],
            sequence_length=final_sequence_length,
            batch_size=current_batch_size
        ).map(set_final_shapes).prefetch(tf.data.AUTOTUNE)

        val_dataset_full = timeseries_dataset_from_array(
            data=x_test_scaled,
            targets=y_test_s_combined[final_sequence_length - 1:],
            sequence_length=final_sequence_length,
            batch_size=current_batch_size
        ).map(set_final_shapes).prefetch(tf.data.AUTOTUNE)

        checkpoint_callback = ModelCheckpoint(filepath=current_checkpoint, save_best_only=True, monitor='val_loss')
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        model.fit(train_dataset_full,
                  epochs=current_epochs,
                  validation_data=val_dataset_full,
                  callbacks=[early_stop_callback, checkpoint_callback],
                  verbose=0)

        yhat_s = model.predict(val_dataset_full, verbose=0)
        yhat_raw = scaler_y.inverse_transform(yhat_s)

        slice_offset = max_possible_window - final_sequence_length
        yhat = yhat_raw[slice_offset:]

        y_test_combined_trimmed = y_test_combined[max_possible_window - 1:]
        y_test_trimmed = y_test[max_possible_window - 1:]

        if cliping_and_customLoss == 0:
            yhat[yhat < y_test_combined_trimmed[:, 2][:, None]] = 0
            n_max = y_test_combined_trimmed[:, 1][:, None]
            yhat = np.where(yhat > n_max, n_max, yhat)
            yhat = np.maximum(yhat, 0)

        results_list.append(yhat)
        metrics_list.append([
            metrics.mean_absolute_error(y_test_trimmed, yhat),
            metrics.mean_squared_error(y_test_trimmed, yhat),
            np.sum(np.abs(y_test_trimmed - yhat)) / np.sum(np.abs(y_test_trimmed)) * 100,
            metrics.r2_score(y_test_trimmed, yhat)
        ])
        print(f"Финальный MAE Шага {step} (Окно {final_sequence_length}):",
              metrics.mean_absolute_error(y_test_trimmed, yhat))

        if 'train_dataset_full' in locals(): del train_dataset_full
        if 'val_dataset_full' in locals(): del val_dataset_full

        del model

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        gc.collect()

    return np.hstack(results_list), np.array(metrics_list)
