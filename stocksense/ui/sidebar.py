"""Sidebar controls."""
from __future__ import annotations

import datetime as dt

import streamlit as st

from ..config import CUSTOM, POPULAR, Settings


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
