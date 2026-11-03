"""Refresh the bundled offline datasets in data/.

Usage:  python scripts/build_sample_data.py

Downloads daily history from Yahoo Finance. If a download fails (for example,
with no internet), a realistic synthetic series is written instead, so the app
always has something to show in offline demo mode.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = {
    "AAPL": 25.0, "MSFT": 55.0, "GOOGL": 37.0, "NVDA": 0.8, "TSLA": 15.0,
    "RELIANCE.NS": 450.0, "TCS.NS": 1100.0,
}
START = dt.date(2016, 1, 1)
OUT = Path(__file__).resolve().parent.parent / "data"


def download(ticker: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(start=START, end=dt.date.today() + dt.timedelta(days=1), auto_adjust=True)
    if df.empty:
        raise ValueError("empty response")
    # tz_localize(None) keeps the exchange's local date; tz_convert(None) would
    # shift to UTC first and move e.g. NSE dates one day earlier.
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def synthetic(ticker: str, start_price: float) -> pd.DataFrame:
    """Geometric Brownian motion with OHLC and volume derived from it."""
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    dates = pd.bdate_range(START, dt.date.today())
    rets = rng.normal(0.0006, 0.018, len(dates))
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, 0.006, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, len(dates))))
    volume = rng.lognormal(17, 0.35, len(dates)).astype("int64")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for ticker, start_price in TICKERS.items():
        try:
            df, kind = download(ticker), "live"
        except Exception as exc:
            print(f"{ticker}: download failed ({exc}); writing synthetic data")
            df, kind = synthetic(ticker, start_price), "synthetic"
        df.index.name = "Date"
        path = OUT / f"{ticker.replace('.', '_')}.csv"
        df.round(4).to_csv(path)
        print(f"{ticker}: {len(df)} rows ({kind}) -> {path.name}")


if __name__ == "__main__":
    main()
