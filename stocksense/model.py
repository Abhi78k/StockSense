"""LSTM model: building, training, evaluation and forecasting.

This module has no Streamlit dependency; the UI wrapper lives in training.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_model(lookback: int):
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.Input(shape=(lookback, 1)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_lstm(close: np.ndarray, dates: pd.DatetimeIndex, lookback: int, epochs: int,
               test_frac: float, on_epoch=None) -> dict:
    """Train on daily log returns and evaluate on a chronological hold-out set.

    Each sample is the previous `lookback` daily log returns; the target is the
    next day's log return. Predicted prices are previous close * exp(return),
    so the model never sees test-period prices during training.
    """
    import tensorflow as tf
    from numpy.lib.stride_tricks import sliding_window_view

    tf.keras.utils.set_random_seed(42)

    rets = np.diff(np.log(close))                 # rets[i] is the return into day i + 1
    X = sliding_window_view(rets, lookback)[:-1]  # X[k] = rets[k : k + lookback]
    y = rets[lookback:]                           # y[k] = rets[k + lookback]
    target_day = np.arange(lookback, len(rets)) + 1

    split_day = int(len(close) * (1 - test_frac))
    train = target_day < split_day
    if train.sum() < 100 or (~train).sum() < 20:
        raise ValueError("Not enough data for this lookback / test split. Choose a longer date range.")

    scale = float(rets[: split_day - 1].std()) or 1e-4
    Xs, ys = (X / scale)[..., None].astype("float32"), (y / scale).astype("float32")

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if on_epoch:
                on_epoch(epoch + 1, epochs, logs or {})

    model = build_model(lookback)
    history = model.fit(
        Xs[train], ys[train],
        validation_split=0.1,  # last 10% of the training period, still chronological
        epochs=epochs, batch_size=32, shuffle=True, verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            Progress(),
        ],
    )

    pred_r = model.predict(Xs[~train], verbose=0, batch_size=512).ravel() * scale
    actual_r = y[~train]
    days = target_day[~train]
    prev, actual = close[days - 1], close[days]
    pred = prev * np.exp(pred_r)

    err = pred - actual
    naive_mae = float(np.mean(np.abs(prev - actual)))
    mae = float(np.mean(np.abs(err)))
    metrics = {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err / actual)) * 100),
        "dir_acc": float(np.mean(np.sign(pred_r) == np.sign(actual_r)) * 100),
        "naive_mae": naive_mae,
        "skill": (1 - mae / naive_mae) * 100 if naive_mae else 0.0,
        "n_train": int(train.sum()),
        "n_test": int((~train).sum()),
        "test_start": dates[days[0]],
    }
    return {
        "model": model,
        "scale": scale,
        "resid": (actual_r - pred_r).astype("float64"),
        "test_dates": dates[days],
        "test_actual": actual,
        "test_pred": pred,
        "metrics": metrics,
        "loss": history.history.get("loss", []),
        "val_loss": history.history.get("val_loss", []),
        "epochs_run": len(history.history.get("loss", [])),
        "forecasts": {},
    }


def forecast(result: dict, close: np.ndarray, last_date: pd.Timestamp, lookback: int,
             horizon: int, n_paths: int = 400, seed: int = 7) -> dict:
    """Recursive multi-step forecast plus a Monte Carlo cone.

    Path 0 is the plain LSTM forecast. The other paths add residuals bootstrapped
    from the test set at every step, giving an empirical uncertainty band.
    """
    model, scale, resid = result["model"], result["scale"], result["resid"]
    rng = np.random.default_rng(seed)

    window = np.tile(np.diff(np.log(close))[-lookback:], (n_paths, 1))
    path_r = np.empty((n_paths, horizon))
    for h in range(horizon):
        mu = model((window / scale)[..., None].astype("float32"), training=False).numpy().ravel() * scale
        noise = rng.choice(resid, size=n_paths)
        noise[0] = 0.0
        step = mu + noise
        path_r[:, h] = step
        if h == 0:
            first_mu = float(mu[0])
        window = np.concatenate([window[:, 1:], step[:, None]], axis=1)

    paths = close[-1] * np.exp(np.cumsum(path_r, axis=1))
    return {
        "dates": pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon),
        "point": paths[0],
        "p10": np.percentile(paths[1:], 10, axis=0),
        "p50": np.percentile(paths[1:], 50, axis=0),
        "p90": np.percentile(paths[1:], 90, axis=0),
        # P(next return > 0) = P(mu + residual > 0), evaluated over all test residuals.
        "p_up": float(np.mean(first_mu + resid > 0)),
        "next_price": float(close[-1] * np.exp(first_mu)),
    }
