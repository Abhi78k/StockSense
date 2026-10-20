# StockSense: LSTM Stock Market Prediction Dashboard

A dark-themed Streamlit dashboard that fetches stock data from Yahoo Finance, draws interactive
candlestick charts with moving averages, trains an LSTM to forecast closing prices, and reports
honest metrics on a chronological test set.

## Features

- Sidebar controls for ticker (popular US and NSE stocks, or any Yahoo Finance symbol), date range,
  prediction period (1 to 30 trading days) and model settings
- Metric cards for latest close, daily change, percent change, volume and 52-week range
- Plotly candlestick chart with 20-day and 50-day moving averages, a volume panel and range buttons
- 2-layer LSTM (TensorFlow/Keras) trained with a live progress bar and early stopping
- Predicted vs historical chart with a recursive forecast and an 80% Monte Carlo uncertainty band
- Next-day UP/DOWN indicator with a probability gauge, clearly labelled as an estimate and not financial advice
- Model performance section: MAE, RMSE, MAPE, directional accuracy, skill against a naive baseline,
  the loss curve and the error distribution
- Caching (data for 1 hour, trained models for the whole server session), so switching tabs or
  forecast length is instant
- Offline fallback: if live data fails, the app switches to bundled CSVs in `data/` and says so.
  The **Offline demo mode** toggle forces this, which is useful on unreliable Wi-Fi.

## Setup

Requires Python 3.10 to 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Presentation tips

- Open the app once before presenting. The first model trains in about 20 to 40 seconds on a laptop CPU,
  and it is cached afterwards. Pre-train each ticker you plan to show.
- If the venue Wi-Fi is unreliable, turn on **Offline demo mode** in the sidebar. Bundled samples:
  AAPL, MSFT, GOOGL, NVDA, TSLA, RELIANCE.NS and TCS.NS.
- Refresh the bundled samples with `python build_sample_data.py`.

## How the model works

1. Prices are converted to daily log returns. Each sample is the previous *lookback* (default 60)
   returns, and the target is the next day's return.
2. The last 20% of days is held out as a strictly chronological test set. Scaling uses training data only.
3. Network: LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1), Adam optimizer, MSE loss.
4. Predicted price = previous close × exp(predicted return).
5. Multi-day forecasts feed predictions back in recursively. 400 Monte Carlo paths that add
   bootstrapped test-set errors produce the uncertainty band and the UP probability.

Daily stock returns are close to random, so expect directional accuracy slightly above 50% and a
skill score near 0% against the naive "tomorrow = today" baseline. These are realistic results, and
the app reports them as they are. It is an educational project, **not financial advice**.

## Project structure

```
app.py                  # Streamlit dashboard (data, model, charts, layout)
build_sample_data.py    # Refreshes the offline CSVs in data/
data/                   # Bundled sample datasets
requirements.txt
.streamlit/config.toml  # Dark theme
```
