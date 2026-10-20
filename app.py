"""StockSense: an LSTM stock market prediction dashboard.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Quieten TensorFlow before it is (lazily) imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="StockSense · LSTM Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
FALLBACK_SAMPLE = "AAPL"

POPULAR = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys Ltd.",
    "HDFCBANK.NS": "HDFC Bank",
}
CUSTOM = "Custom ticker…"

UP, DOWN = "#00d4a0", "#ff4d6d"
ACCENT, CYAN, AMBER = "#7c5cff", "#22d3ee", "#fbbf24"
TEXT, MUTED = "#e5e7eb", "#8b95a7"
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{
  background:
    radial-gradient(1100px 600px at 0% -10%, rgba(124,92,255,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,.09), transparent 60%),
    #0b0f19;
}}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #111827 0%, #0d1220 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}}
.brand {{ font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em;
  background: linear-gradient(90deg, {ACCENT}, {CYAN}); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }}
.brand-sub {{ color: {MUTED}; font-size: .85rem; margin-top: -4px; margin-bottom: .6rem; }}
.hero {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap;
  gap: 1rem; margin-bottom: 1.2rem; }}
.hero h1 {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -.03em; margin: 0; padding: 0; color: {TEXT}; }}
.hero .ticker {{ color: {MUTED}; font-weight: 600; }}
.badges {{ display: flex; gap: .5rem; flex-wrap: wrap; }}
.badge {{ font-size: .75rem; font-weight: 600; padding: .3rem .7rem; border-radius: 999px;
  border: 1px solid rgba(255,255,255,.1); background: rgba(255,255,255,.04); color: {TEXT}; }}
.badge.live {{ color: {UP}; border-color: rgba(0,212,160,.35); background: rgba(0,212,160,.08); }}
.badge.sample {{ color: {AMBER}; border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.08); }}
.metric-card {{
  background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
  border: 1px solid rgba(255,255,255,.08); border-radius: 16px; padding: 1rem 1.2rem;
  box-shadow: 0 8px 30px rgba(0,0,0,.25); height: 100%;
}}
.metric-label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; color: {MUTED}; font-weight: 600; }}
.metric-value {{ font-size: 1.65rem; font-weight: 750; color: {TEXT}; margin-top: .25rem; line-height: 1.2; }}
.metric-sub {{ font-size: .82rem; color: {MUTED}; margin-top: .2rem; }}
.up {{ color: {UP} !important; }}
.down {{ color: {DOWN} !important; }}
.trend-card {{ border-radius: 18px; padding: 1.4rem 1.5rem; border: 1px solid rgba(255,255,255,.08); }}
.trend-card.up {{ background: linear-gradient(145deg, rgba(0,212,160,.16), rgba(0,212,160,.03)); border-color: rgba(0,212,160,.3); }}
.trend-card.down {{ background: linear-gradient(145deg, rgba(255,77,109,.16), rgba(255,77,109,.03)); border-color: rgba(255,77,109,.3); }}
.trend-dir {{ font-size: 2.8rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; margin: .4rem 0 .2rem; }}
.trend-prob {{ font-size: 1.05rem; color: {TEXT}; font-weight: 600; }}
.trend-line {{ color: {TEXT}; font-size: .92rem; margin-top: .55rem; }}
.disclaimer {{ font-size: .75rem; color: {MUTED}; margin-top: .9rem; padding-top: .7rem;
  border-top: 1px dashed rgba(255,255,255,.12); }}
.section-title {{ font-size: 1.05rem; font-weight: 700; color: {TEXT}; margin: 1.2rem 0 .5rem; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{ background: rgba(255,255,255,.03); border-radius: 10px 10px 0 0; padding: .5rem 1rem; }}
.footer-note {{ text-align: center; color: {MUTED}; font-size: .78rem; margin-top: 2rem; }}
</style>
"""


@dataclass(frozen=True)
class Settings:
    ticker: str
    start: dt.date
    end: dt.date
    horizon: int
    lookback: int
    epochs: int
    test_pct: int
    offline: bool
    retrain: bool


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def currency(ticker: str) -> str:
    return "₹" if ticker.upper().endswith((".NS", ".BO")) else "$"


def fmt_volume(v: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= div:
            return f"{v / div:.2f}{unit}"
    return f"{v:,.0f}"


def tone(x: float) -> str:
    return "up" if x >= 0 else "down"


def card(label: str, value: str, sub: str = "", value_tone: str = "", sub_tone: str = "") -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {value_tone}">{value}</div>'
        f'<div class="metric-sub {sub_tone}">{sub}</div></div>'
    )


def render_cards(cards: list[str]) -> None:
    for col, html in zip(st.columns(len(cards)), cards):
        col.markdown(html, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", color="#d1d5db"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False)
    return fig


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf

    # yfinance treats `end` as exclusive, so ask for one extra day.
    raw = yf.Ticker(ticker).history(start=start, end=end + dt.timedelta(days=1), auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for '{ticker}'")
    return _clean(raw)


@st.cache_data(show_spinner=False)
def load_sample(ticker: str) -> tuple[pd.DataFrame, str]:
    path = DATA_DIR / f"{ticker.replace('.', '_')}.csv"
    used = ticker
    if not path.exists():
        path, used = DATA_DIR / f"{FALLBACK_SAMPLE}.csv", FALLBACK_SAMPLE
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return _clean(df), used


def get_data(s: Settings, min_rows: int) -> tuple[pd.DataFrame, str, str, str | None]:
    """Return (data, source, ticker actually shown, note for the user)."""
    note = None
    if not s.offline:
        try:
            df = fetch_live(s.ticker, s.start, s.end)
            if len(df) >= min_rows:
                return df, "live", s.ticker, None
            note = (
                f"Only {len(df)} trading days of live data for **{s.ticker}** in this range "
                f"(need at least {min_rows}); pick a longer date range or use the sample below."
            )
        except Exception as exc:  # network down, bad ticker, rate limit, ...
            note = f"Live data for **{s.ticker}** is unavailable ({exc})."

    df, used = load_sample(s.ticker)
    sliced = df.loc[pd.Timestamp(s.start): pd.Timestamp(s.end)]
    msgs = [note] if note else []
    if used != s.ticker:
        msgs.append(f"No bundled sample exists for {s.ticker}, so the **{used}** sample is shown.")
    else:
        msgs.append("Showing bundled sample data.")
    if len(sliced) < min_rows:
        msgs.append("The selected range is too short for the sample, so its full history is used.")
        sliced = df
    return sliced, "sample", used, " ".join(msgs)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    return df


# ----------------------------------------------------------------------------
# LSTM model
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker,
            increasing=dict(line=dict(color=UP), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA 20", line=dict(color=AMBER, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA 50", line=dict(color=ACCENT, width=1.8)), row=1, col=1)
    vol_colors = np.where(df["Close"] >= df["Open"], "rgba(0,212,160,.55)", "rgba(255,77,109,.55)")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, showlegend=False), row=2, col=1)

    # Open on the last year by default; the range buttons zoom in or out.
    start = max(df.index[0], df.index[-1] - pd.DateOffset(years=1))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], range=[start, df.index[-1] + pd.Timedelta(days=2)])
    fig.update_xaxes(
        row=1, col=1, rangeslider_visible=False,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="rgba(255,255,255,.06)", activecolor=ACCENT, font=dict(color=TEXT),
            x=1, xanchor="right", y=1.02, yanchor="bottom",
        ),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    style_fig(fig, 620)
    fig.update_layout(legend=dict(x=0, xanchor="left"))
    return fig


def forecast_chart(df: pd.DataFrame, res: dict, fc: dict) -> go.Figure:
    start = res["metrics"]["test_start"] - pd.Timedelta(days=90)
    hist = df.loc[start:]
    last_date, last_close = df.index[-1], float(df["Close"].iloc[-1])
    fc_x = [last_date, *fc["dates"]]

    fig = go.Figure()
    fig.add_vrect(x0=res["metrics"]["test_start"], x1=last_date, fillcolor="rgba(255,255,255,.035)", line_width=0)
    fig.add_annotation(x=res["metrics"]["test_start"], y=1, yref="paper", text="Test period", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=MUTED, size=11))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Historical close", line=dict(color="#e5e7eb", width=1.8)))
    fig.add_trace(go.Scatter(x=res["test_dates"], y=res["test_pred"], name="LSTM prediction (test)",
                             line=dict(color=CYAN, width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=fc_x, y=[last_close, *fc["p90"]], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fc_x, y=[last_close, *fc["p10"]], fill="tonexty", fillcolor="rgba(124,92,255,.22)",
                             line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc_x, y=[last_close, *fc["point"]], name="Forecast",
                             line=dict(color=ACCENT, width=2.6), mode="lines+markers", marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[last_date], y=[last_close], mode="markers", name="Last close",
                             marker=dict(color=AMBER, size=10, line=dict(color="#0b0f19", width=2))))
    return style_fig(fig, 480)


def gauge(p_up: float) -> go.Figure:
    color = UP if p_up >= 0.5 else DOWN
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p_up * 100, number=dict(suffix="%", font=dict(size=34, color=TEXT)),
        title=dict(text="Probability of UP move", font=dict(size=13, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickwidth=1),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(255,255,255,.03)", borderwidth=0,
            steps=[dict(range=[0, 50], color="rgba(255,77,109,.12)"), dict(range=[50, 100], color="rgba(0,212,160,.12)")],
            threshold=dict(line=dict(color=TEXT, width=2), thickness=0.8, value=50),
        ),
    ))
    style_fig(fig, 230)
    fig.update_layout(margin=dict(l=25, r=25, t=50, b=0))
    return fig


def test_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["test_dates"], y=res["test_actual"], name="Actual", line=dict(color="#e5e7eb", width=1.8)))
    fig.add_trace(go.Scatter(x=res["test_dates"], y=res["test_pred"], name="Predicted", line=dict(color=CYAN, width=1.6)))
    return style_fig(fig, 380)


def loss_chart(res: dict) -> go.Figure:
    ep = list(range(1, len(res["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=res["loss"], name="Training loss", line=dict(color=ACCENT, width=2)))
    if res["val_loss"]:
        fig.add_trace(go.Scatter(x=ep, y=res["val_loss"], name="Validation loss", line=dict(color=AMBER, width=2)))
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="MSE (scaled)")
    return style_fig(fig, 300)


def error_hist(res: dict) -> go.Figure:
    pct_err = (res["test_pred"] - res["test_actual"]) / res["test_actual"] * 100
    fig = go.Figure(go.Histogram(x=pct_err, nbinsx=40, marker_color=CYAN, opacity=0.8, name="Error"))
    fig.update_xaxes(title_text="Prediction error (%)")
    fig.update_yaxes(title_text="Days")
    style_fig(fig, 300)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
def sidebar() -> Settings:
    with st.sidebar:
        st.markdown('<div class="brand">📈 StockSense</div><div class="brand-sub">Deep-learning price forecasting</div>',
                    unsafe_allow_html=True)

        choice = st.selectbox("Stock ticker", [*POPULAR, CUSTOM],
                              format_func=lambda t: f"{t} · {POPULAR[t]}" if t in POPULAR else t)
        ticker = choice
        if choice == CUSTOM:
            ticker = st.text_input("Yahoo Finance symbol", "AMD", help="e.g. AMD, NFLX, SBIN.NS, ^NSEI").strip().upper() or "AAPL"

        today = dt.date.today()
        c1, c2 = st.columns(2)
        start = c1.date_input("Start date", today - dt.timedelta(days=365 * 5),
                              min_value=dt.date(2000, 1, 1), max_value=today)
        end = c2.date_input("End date", today, min_value=dt.date(2000, 1, 1), max_value=today)
        horizon = st.slider("Prediction period (trading days)", 1, 30, 7)

        with st.expander("⚙️ Model settings"):
            lookback = st.slider("Lookback window (days)", 20, 120, 60, 5,
                                 help="How many past days the LSTM sees for each prediction.")
            epochs = st.slider("Max training epochs", 5, 60, 20, 5,
                               help="Early stopping ends training sooner if validation loss stops improving.")
            test_pct = st.slider("Test set size (%)", 10, 30, 20, 5,
                                 help="The most recent part of the data, held out for evaluation.")

        offline = st.toggle("Offline demo mode", value=False,
                            help="Use the bundled sample datasets instead of fetching live data.")
        retrain = st.button("🔄 Retrain model", width="stretch")

        st.markdown("---")
        st.caption("Data: Yahoo Finance via yfinance. Model: 2-layer LSTM (TensorFlow/Keras). "
                   "Educational project, not financial advice.")

    return Settings(ticker, start, end, horizon, lookback, epochs, test_pct, offline, retrain)


def hero(ticker: str, source: str, df: pd.DataFrame) -> None:
    name = POPULAR.get(ticker, ticker)
    src = '<span class="badge live">● LIVE DATA</span>' if source == "live" else '<span class="badge sample">● SAMPLE DATA</span>'
    st.markdown(
        f"""<div class="hero"><div><h1>{name} <span class="ticker">({ticker})</span></h1></div>
        <div class="badges">{src}
        <span class="badge">Last close · {df.index[-1]:%d %b %Y}</span>
        <span class="badge">{len(df):,} trading days</span></div></div>""",
        unsafe_allow_html=True,
    )


def market_cards(df: pd.DataFrame, cur: str) -> None:
    last, prev = df.iloc[-1], df.iloc[-2]
    chg = last["Close"] - prev["Close"]
    pct = chg / prev["Close"] * 100
    avg_vol = df["Volume"].tail(20).mean()
    vol_vs = (last["Volume"] / avg_vol - 1) * 100 if avg_vol else 0.0
    year = df.loc[df.index[-1] - pd.DateOffset(years=1):]
    arrow = "▲" if chg >= 0 else "▼"
    render_cards([
        card("Latest close", f"{cur}{last['Close']:,.2f}", f"as of {df.index[-1]:%d %b %Y}"),
        card("Daily change", f"{arrow} {cur}{abs(chg):,.2f}", f"vs prev close {cur}{prev['Close']:,.2f}", tone(chg)),
        card("Percent change", f"{pct:+.2f}%", "1-day return", tone(pct)),
        card("Volume", fmt_volume(last["Volume"]), f"{vol_vs:+.0f}% vs 20-day avg", sub_tone=tone(vol_vs)),
        card("52-week range", f"{cur}{year['Low'].min():,.0f} – {cur}{year['High'].max():,.0f}",
             f"MA20 {cur}{last['MA20']:,.2f} · MA50 {cur}{last['MA50']:,.2f}"),
    ])


def forecast_tab(df: pd.DataFrame, res: dict, fc: dict, cur: str, horizon: int) -> None:
    last_close = float(df["Close"].iloc[-1])
    p_up = fc["p_up"]
    is_up = p_up >= 0.5
    exp_chg = (fc["next_price"] / last_close - 1) * 100
    end_chg = (fc["point"][-1] / last_close - 1) * 100

    left, right = st.columns([1, 2.1], gap="large")
    with left:
        st.markdown(
            f"""<div class="trend-card {'up' if is_up else 'down'}">
              <div class="metric-label">Next-day trend · {fc['dates'][0]:%a %d %b %Y}</div>
              <div class="trend-dir {'up' if is_up else 'down'}">{'▲ UP' if is_up else '▼ DOWN'}</div>
              <div class="trend-prob">{max(p_up, 1 - p_up) * 100:.1f}% estimated probability</div>
              <div class="trend-line">Predicted close: <b>{cur}{fc['next_price']:,.2f}</b>
                <span class="{tone(exp_chg)}">({exp_chg:+.2f}%)</span></div>
              <div class="trend-line">{horizon}-day outlook: <b>{cur}{fc['point'][-1]:,.2f}</b>
                <span class="{tone(end_chg)}">({end_chg:+.2f}%)</span></div>
              <div class="disclaimer">⚠️ This is a statistical model <b>estimate</b> for educational purposes only.
                It is <b>not financial advice</b>. Markets are driven by news and events the model cannot see.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(gauge(p_up), config=PLOTLY_CONFIG)
    with right:
        st.markdown('<div class="section-title">Predicted vs historical prices</div>', unsafe_allow_html=True)
        st.plotly_chart(forecast_chart(df, res, fc), config=PLOTLY_CONFIG)

    st.markdown(f'<div class="section-title">{horizon}-day forecast</div>', unsafe_allow_html=True)
    table = pd.DataFrame({
        "Date": fc["dates"].strftime("%a %d %b %Y"),
        "Forecast": fc["point"],
        "Low (10th pct)": fc["p10"],
        "High (90th pct)": fc["p90"],
        "Change vs last close (%)": (fc["point"] / last_close - 1) * 100,
    })
    money = {"format": f"{cur}%.2f"}
    st.dataframe(
        table, hide_index=True,
        column_config={
            "Forecast": st.column_config.NumberColumn(**money),
            "Low (10th pct)": st.column_config.NumberColumn(**money),
            "High (90th pct)": st.column_config.NumberColumn(**money),
            "Change vs last close (%)": st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )
    st.download_button("⬇️ Download forecast (CSV)", table.to_csv(index=False), file_name="forecast.csv", mime="text/csv")


def performance_tab(res: dict, cur: str, s: Settings) -> None:
    m = res["metrics"]
    st.caption(
        f"Evaluated on the most recent {m['n_test']} trading days "
        f"({m['test_start']:%d %b %Y} onward), which the model never saw during training."
    )
    render_cards([
        card("MAE", f"{cur}{m['mae']:,.2f}", "mean absolute error"),
        card("RMSE", f"{cur}{m['rmse']:,.2f}", "root mean squared error"),
        card("MAPE", f"{m['mape']:.2f}%", "mean absolute % error"),
        card("Directional accuracy", f"{m['dir_acc']:.1f}%", "up/down called correctly",
             "up" if m["dir_acc"] >= 50 else "down"),
        card("Skill vs naive", f"{m['skill']:+.1f}%",
             f"naive MAE {cur}{m['naive_mae']:,.2f} (tomorrow = today)", tone(m["skill"])),
    ])

    st.markdown('<div class="section-title">Actual vs predicted (test set)</div>', unsafe_allow_html=True)
    st.plotly_chart(test_chart(res), config=PLOTLY_CONFIG)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Training history</div>', unsafe_allow_html=True)
        st.plotly_chart(loss_chart(res), config=PLOTLY_CONFIG)
    with c2:
        st.markdown('<div class="section-title">Error distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(error_hist(res), config=PLOTLY_CONFIG)

    with st.expander("🧠 How the model works"):
        st.markdown(f"""
- **Input:** the last **{s.lookback}** daily log returns (scaled by their training-period standard deviation).
- **Network:** LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, ReLU) → Dense(1).
- **Target:** the next day's log return, converted back to a price via `previous close × exp(return)`.
- **Training:** Adam (lr 1e-3), MSE loss, up to {s.epochs} epochs with early stopping
  ({res['epochs_run']} epochs run on {m['n_train']:,} samples in {res['train_seconds']:.1f}s).
- **Evaluation:** strictly chronological split. The last {s.test_pct}% of days is held out, so there is no look-ahead leakage.
- **Forecast:** predictions are fed back in recursively. The shaded band comes from 400 Monte Carlo paths that
  add bootstrapped test-set errors, and the UP probability is the share of those outcomes above today's close.
- **Reading the results:** daily returns are very noisy, so a directional accuracy a little above 50% and a
  skill near 0% against the naive baseline are normal, honest results for price-only models.
""")


def data_tab(df: pd.DataFrame, ticker: str) -> None:
    view = df.sort_index(ascending=False).round(2)
    st.dataframe(view, height=460, column_config={"Volume": st.column_config.NumberColumn(format="%d")})
    st.download_button("⬇️ Download data (CSV)", view.to_csv(), file_name=f"{ticker}_history.csv", mime="text/csv")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    s = sidebar()
    if s.start >= s.end:
        st.error("The start date must be before the end date.")
        st.stop()

    min_rows = s.lookback + 200
    with st.spinner(f"Fetching market data for {s.ticker}…"):
        try:
            df, source, ticker, note = get_data(s, min_rows)
        except Exception as exc:
            st.error(f"Could not load live or sample data: {exc}. Check that the `data/` folder is next to app.py.")
            st.stop()

    df = add_indicators(df)
    cur = currency(ticker)
    hero(ticker, source, df)
    if note:
        st.warning(note, icon="⚠️")
    market_cards(df, cur)
    st.write("")

    res = get_trained(df, ticker, source, s) if len(df) >= min_rows else None
    fc = None
    if res is not None:
        fc = res["forecasts"].get(s.horizon)
        if fc is None:
            close = df["Close"].to_numpy(dtype="float64")
            fc = res["forecasts"][s.horizon] = forecast(res, close, df.index[-1], s.lookback, s.horizon)

    tabs = st.tabs(["📈 Price chart", "🔮 Forecast", "📊 Model performance", "🗂️ Data"])
    with tabs[0]:
        st.plotly_chart(candlestick_chart(df, ticker), config=PLOTLY_CONFIG)
    with tabs[1]:
        if fc is None:
            st.info(f"The model needs at least {min_rows} trading days. Choose a longer date range.")
        else:
            forecast_tab(df, res, fc, cur, s.horizon)
    with tabs[2]:
        if res is None:
            st.info("Model performance appears once a model has been trained.")
        else:
            performance_tab(res, cur, s)
    with tabs[3]:
        data_tab(df, ticker)

    st.markdown(
        '<div class="footer-note">StockSense · Built with Streamlit, TensorFlow and Plotly · '
        "Predictions are estimates for educational purposes only, not financial advice.</div>",
        unsafe_allow_html=True,
    )


main()
