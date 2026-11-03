"""Loading price data from Yahoo Finance or the bundled samples."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from .config import DATA_DIR, FALLBACK_SAMPLE, Settings


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        # Drop the timezone but keep the exchange's local date. tz_convert(None)
        # would shift to UTC first, moving e.g. NSE dates one day earlier.
        idx = idx.tz_localize(None)
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
