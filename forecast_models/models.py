import os

import json
import optuna
import tensorflow as tf

from catboost import CatBoostRegressor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression

from utils import optuna_cbr_search, optuna_rfr_search, optuna_lstm_search, scale_combined, custom_loss, \
    optuna_mlp_search

tf.random.set_seed(42)


def get_catboost(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y):

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: optuna_cbr_search(trial, X_train_s, y_train_s,
                                               X_test_s, y_test_s, scaler_y),
                   n_trials=40)

    best_params = study.best_params

    model = CatBoostRegressor(iterations=2000,
                              depth=best_params['depth'],
                              learning_rate=best_params['lr'],
                              loss_function='MAE',
                              verbose=0)
    return model


def _compile_lstm_from_params(input_shape, best_params, model_loss):
    model = Sequential([
        LSTM(best_params['n_units_lstm'], input_shape=input_shape, unroll=True),
        Dense(best_params['n_units_dense']),
        Dense(1, dtype='float32')
    ])
    optimizer = Adam(learning_rate=best_params['lr'])
    model.compile(optimizer=optimizer, loss=model_loss)
    return model


def _compile_mlp_from_params(best_params):
    model = Sequential()
    for i in range(best_params['n_layers']):
        model.add(Dense(best_params[f'units_l{i}'], activation='relu'))

    model.add(Dense(1, dtype='float32'))

    optimizer = Adam(learning_rate=best_params['lr'])
    model.compile(optimizer=optimizer, loss=custom_loss)
    return model


def get_lstm(input_shape, X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y,
             y_train_s_combined, y_test_s_combined, checkpoint_dir, calc_goal, loss_type='custom'):

    checkpoint_filepath = f"{checkpoint_dir}{calc_goal}_LSTM_{loss_type}.keras"
    params_filepath = f"{checkpoint_dir}{calc_goal}_LSTM_params_{loss_type}.json"

    model_loss = custom_loss if loss_type == 'custom' else 'mae'
    custom_objs = {'custom_loss': custom_loss} if loss_type == 'custom' else None

    if os.path.exists(checkpoint_filepath) and os.path.exists(params_filepath):
        print(f"Loading model and best parameters for {loss_type} loss...")
        try:
            model = load_model(checkpoint_filepath, custom_objects=custom_objs)
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs = 50
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True,
                                                min_delta=0.0001)
        except Exception as exc:
            print(f"Checkpoint is incompatible, rebuilding {loss_type} LSTM from saved parameters: {exc}")
            with open(params_filepath, 'r') as f:
                best_params = json.load(f)
            current_epochs = 1000
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1,
                                                restore_best_weights=True, min_delta=0.0001)
            model = _compile_lstm_from_params(input_shape, best_params, model_loss)
    else:
        print(f"Starting hyperparameter optimization for {loss_type} loss...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optuna_lstm_search(trial, X_train_s, y_train_s,
                                                        X_test_s, y_test_s, scaler_y, input_shape,
                                                        y_train_s_combined, y_test_s_combined, loss_type=loss_type),
                       n_trials=40)
        best_params = study.best_params

        with open(params_filepath, 'w') as f:
            json.dump(best_params, f)

        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

        model = _compile_lstm_from_params(input_shape, best_params, model_loss)
        is_mixed = isinstance(model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
        print(f"--- Аппаратный Loss Scale для float16 активен: {is_mixed} ---")

    return model, early_stop_callback, current_epochs, checkpoint_filepath
def get_linear(): return LinearRegression()
def get_mlp(input_shape, X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y, y_train_s_combined, y_test_s_combined, checkpoint_filepath, params_filepath):
    if os.path.exists(checkpoint_filepath) and os.path.exists(params_filepath):
        print("Loading model and best parameters...")
        try:
            model = load_model(checkpoint_filepath, custom_objects={'custom_loss': custom_loss})
            model.optimizer.learning_rate.assign(1e-4)
            current_epochs = 50
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True,
                                                min_delta=0.0001)
        except Exception as exc:
            print(f"Checkpoint is incompatible, rebuilding MLP from saved parameters: {exc}")
            with open(params_filepath, 'r') as f:
                best_params = json.load(f)
            current_epochs = 1000
            early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1,
                                                restore_best_weights=True, min_delta=0.0001)
            model = _compile_mlp_from_params(best_params)
    else:
        print("Starting hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optuna_mlp_search(trial, X_train_s, y_train_s,
                                                   X_test_s, y_test_s, scaler_y, y_train_s_combined, y_test_s_combined), n_trials=40)
        best_params = study.best_params

        with open(params_filepath, 'w') as f:json.dump(best_params, f)

        current_epochs = 1000
        early_stop_callback = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True,
                                            min_delta=0.0001)

        model = _compile_mlp_from_params(best_params)
    return model, early_stop_callback, current_epochs
def get_rfr(X_train_s, X_test_s, y_train_s, y_test_s, y_train, y_test, scaler_y):
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: optuna_rfr_search(trial, X_train_s, y_train_s, X_test_s, y_test_s, scaler_y),
                   n_trials=40)

    best_params = study.best_params
    params = {
        "n_estimators": best_params['n_estimators'],
        "max_depth": best_params['max_depth'],
        "min_samples_split": best_params['min_samples_split'],
        "min_samples_leaf": best_params['min_samples_leaf'],
        "max_features": best_params['max_features'],
        "n_jobs": -1
    }
    return RandomForestRegressor(**params)
