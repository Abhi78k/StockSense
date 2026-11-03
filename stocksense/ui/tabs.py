"""Contents of the Forecast, Model performance and Data tabs."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..charts import error_hist, forecast_chart, gauge, loss_chart, test_chart
from ..config import PLOTLY_CONFIG, Settings
from ..formatting import tone
from .components import card, render_cards


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
