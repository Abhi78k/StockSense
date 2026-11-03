"""Page assembly: wires data, model and UI together on every Streamlit rerun."""
from __future__ import annotations

import streamlit as st

from .charts import candlestick_chart
from .config import PLOTLY_CONFIG
from .data import add_indicators, get_data
from .formatting import currency
from .model import forecast
from .styles import CSS
from .training import get_trained
from .ui.header import hero, market_cards
from .ui.sidebar import sidebar
from .ui.tabs import data_tab, forecast_tab, performance_tab


def main() -> None:
    st.set_page_config(
        page_title="StockSense · LSTM Forecasting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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
            st.error(f"Could not load live or sample data: {exc}. Check that the `data/` folder is in the project root.")
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
