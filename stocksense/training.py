"""Training the LSTM inside the Streamlit UI, with a process-wide model cache."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from .config import Settings
from .model import train_lstm


@st.cache_resource(show_spinner=False)
def model_store() -> dict:
    """Process-wide store of trained models, so reruns and new sessions reuse them."""
    return {}


def get_trained(df: pd.DataFrame, ticker: str, source: str, s: Settings) -> dict | None:
    store = model_store()
    close = df["Close"].to_numpy(dtype="float64")
    key = (ticker, source, str(df.index[0].date()), str(df.index[-1].date()), len(df),
           round(float(close[-1]), 4), s.lookback, s.epochs, s.test_pct)
    if s.retrain:
        store.pop(key, None)
    if key in store:
        return store[key]

    status = st.status(f"Training LSTM on {len(df):,} trading days of {ticker}…", expanded=True)
    t0 = time.time()
    try:
        with status:
            st.write("Loading TensorFlow and preparing sliding windows…")
            bar = st.progress(0.0, text="Starting training")

            def on_epoch(ep: int, total: int, logs: dict) -> None:
                bar.progress(
                    min(ep / total, 1.0),
                    text=f"Epoch {ep}/{total} · loss {logs.get('loss', 0):.4f} · val_loss {logs.get('val_loss', 0):.4f}",
                )

            result = train_lstm(close, df.index, s.lookback, s.epochs, s.test_pct / 100, on_epoch)
            bar.progress(1.0, text="Training complete")
    except ImportError:
        status.update(label="TensorFlow is not installed", state="error", expanded=True)
        st.error("TensorFlow could not be imported. Run `pip install -r requirements.txt` and restart the app.")
        return None
    except Exception as exc:
        status.update(label="Training failed", state="error", expanded=True)
        st.error(f"Model training failed: {exc}")
        return None

    result["train_seconds"] = time.time() - t0
    status.update(
        label=f"Model ready · {result['epochs_run']} epochs in {result['train_seconds']:.1f}s",
        state="complete", expanded=False,
    )
    if len(store) >= 8:
        store.pop(next(iter(store)))
    store[key] = result
    return result
