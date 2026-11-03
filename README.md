# StockSense: LSTM Stock Market Prediction Dashboard

StockSense is a single-page Streamlit web app that downloads daily stock prices from Yahoo Finance,
shows them as an interactive candlestick chart, trains a small LSTM neural network (TensorFlow/Keras)
to predict the next day's closing price, produces a multi-day forecast with an uncertainty band, and
reports evaluation metrics on a strictly chronological hold-out set.

It is an educational deep-learning project. It is **not financial advice**, and the app says so
wherever it shows a prediction.

This README is written to be a complete, standalone description of the project: what every file does,
how data flows through the app, how the model is built, trained, evaluated and used for forecasting,
how caching works, and what the known limitations are.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Repository layout](#2-repository-layout)
3. [Tech stack and dependencies](#3-tech-stack-and-dependencies)
4. [How the app runs, end to end](#4-how-the-app-runs-end-to-end)
5. [User inputs (sidebar)](#5-user-inputs-sidebar)
6. [Data layer](#6-data-layer)
7. [The LSTM model](#7-the-lstm-model)
8. [Forecasting and uncertainty](#8-forecasting-and-uncertainty)
9. [Evaluation metrics](#9-evaluation-metrics)
10. [Caching and state](#10-caching-and-state)
11. [User interface](#11-user-interface)
12. [Theming and styling](#12-theming-and-styling)
13. [Offline sample data](#13-offline-sample-data)
14. [Error handling](#14-error-handling)
15. [Module map](#15-module-map)
16. [Known limitations and quirks](#16-known-limitations-and-quirks)
17. [Extending the project](#17-extending-the-project)
18. [Presentation tips](#18-presentation-tips)
19. [Disclaimer](#19-disclaimer)

---

## 1. Quick start

Requires Python 3.10 to 3.12 (the range TensorFlow 2.16+ supports).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. With the default settings it loads five years of AAPL data,
trains a model (about 20 to 40 seconds on a laptop CPU the first time) and then shows the dashboard.

No API keys, accounts, databases or environment variables are needed. Yahoo Finance data is fetched
anonymously through the `yfinance` library. If there is no internet connection, the app falls back to
the bundled CSVs in `data/` automatically.

---

## 2. Repository layout

```
.
├── app.py                  # Streamlit entry point: imports stocksense.dashboard.main and calls it
├── stocksense/             # The application package
│   ├── __init__.py         # Sets TensorFlow logging env vars before TF is ever imported
│   ├── config.py           # Paths, ticker list, colors, Plotly config, Settings dataclass
│   ├── styles.py           # Custom CSS injected into the page
│   ├── formatting.py       # currency(), fmt_volume(), tone()
│   ├── data.py             # Yahoo Finance download, sample CSV fallback, cleaning, moving averages
│   ├── model.py            # LSTM build, training, evaluation, Monte Carlo forecast (no Streamlit)
│   ├── training.py         # Streamlit wrapper: progress UI and process-wide trained-model cache
│   ├── charts.py           # All Plotly figures and the shared style_fig()
│   ├── dashboard.py        # main(): page config and assembly of the whole page
│   └── ui/
│       ├── __init__.py
│       ├── components.py   # card() and render_cards() HTML metric cards
│       ├── sidebar.py      # sidebar() -> Settings
│       ├── header.py       # hero() title/badges and market_cards()
│       └── tabs.py         # forecast_tab(), performance_tab(), data_tab()
├── scripts/
│   └── build_sample_data.py  # Regenerates the offline CSVs in data/
├── data/                   # Bundled daily OHLCV history used in offline / fallback mode
│   ├── AAPL.csv
│   ├── GOOGL.csv
│   ├── MSFT.csv
│   ├── NVDA.csv
│   ├── RELIANCE_NS.csv     # RELIANCE.NS (dots in tickers become underscores in file names)
│   ├── TCS_NS.csv          # TCS.NS
│   └── TSLA.csv
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignores __pycache__/ and .venv/
├── .streamlit/
│   └── config.toml         # Dark theme, headless server, telemetry off, minimal toolbar
└── README.md
```

There are no tests and no build step. `streamlit run app.py` puts the project root on `sys.path`
(Streamlit inserts the script's directory), so `app.py` can import the `stocksense` package without
installing it. Streamlit re-executes `app.py`, and therefore `main()`, on every rerun; the package
modules themselves are imported once per server process.

Dependency direction inside the package (arrows mean "imports"):

```
app.py -> dashboard -> ui.sidebar, ui.header, ui.tabs -> ui.components, charts, formatting, config
                    -> data, training -> model
                    -> charts, styles, formatting, config
```

`model.py`, `formatting.py`, `config.py`, `styles.py` and `charts.py` do not import Streamlit, so the
model and charts can be used from a notebook or script. `data.py` and `training.py` use Streamlit
for caching and progress display.

---

## 3. Tech stack and dependencies

From `requirements.txt`:

| Package       | Minimum | Used for |
|---------------|---------|----------|
| `streamlit`   | 1.49    | Web UI, widgets, caching (`st.cache_data`, `st.cache_resource`), status/progress. 1.49+ is needed for `width="stretch"` on buttons. |
| `tensorflow`  | 2.16    | Keras LSTM model, training, early stopping, inference |
| `yfinance`    | 0.2.54  | Downloading daily OHLCV history from Yahoo Finance |
| `plotly`      | 5.22    | Candlestick, line, histogram and gauge charts |
| `pandas`      | 2.1     | Time series handling, CSV I/O, rolling averages |
| `numpy`       | 1.26    | Returns, sliding windows, metrics, Monte Carlo simulation |

TensorFlow and yfinance are imported lazily inside the functions that need them, so the page shell
renders before the heavy imports finish. Two environment variables are set in `stocksense/__init__.py`
(which runs before any submodule) before TensorFlow loads, to keep the console quiet:

- `TF_CPP_MIN_LOG_LEVEL=2` hides TensorFlow info and warning logs.
- `TF_ENABLE_ONEDNN_OPTS=0` disables the oneDNN notice (and makes CPU results more reproducible).

The UI font, Inter, is loaded from Google Fonts by the injected CSS. Without internet the browser
falls back to the system sans-serif font.

---

## 4. How the app runs, end to end

Streamlit reruns `app.py` from top to bottom on every widget interaction. `app.py` calls
`stocksense.dashboard.main()`, which does the following on each run:

```
st.set_page_config(...)       -> title "StockSense · LSTM Forecasting", 📈 icon, wide layout
inject CSS                    -> styles.CSS
sidebar()                     -> Settings (ticker, dates, horizon, lookback, epochs, test %, offline, retrain)
validate start < end          -> error and stop otherwise
get_data(settings, min_rows)  -> (df, source "live"|"sample", ticker shown, optional warning note)
add_indicators(df)            -> adds MA20 and MA50 columns
hero() + warning note + market_cards()
get_trained(df, ...)          -> trained model + test predictions + metrics (cached in-process)
forecast(...)                 -> point forecast, 10/50/90 percentile band, P(up), next price
                                 (cached per horizon inside the trained-model result)
tabs: Price chart | Forecast | Model performance | Data
footer
```

`min_rows = lookback + 200`. If the loaded data has fewer rows than this, no model is trained and the
Forecast and Model performance tabs show an explanatory message instead. The Price chart and Data
tabs always work.

---

## 5. User inputs (sidebar)

All inputs are collected by `sidebar()` (`stocksense/ui/sidebar.py`) into an immutable `Settings`
dataclass (`stocksense/config.py`).

| Control | Default | Range / options | Effect |
|---------|---------|-----------------|--------|
| Stock ticker | AAPL | AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, or "Custom ticker…" | Symbol to load |
| Yahoo Finance symbol | AMD | Any Yahoo symbol (e.g. `NFLX`, `SBIN.NS`, `^NSEI`) | Shown only for "Custom ticker…". Upper-cased and trimmed; empty falls back to AAPL |
| Start date | today minus 5 years | 2000-01-01 to today | Start of the history window |
| End date | today | 2000-01-01 to today | End of the window (inclusive) |
| Prediction period | 7 | 1 to 30 trading days | Forecast horizon. Changing it does not retrain |
| Lookback window | 60 | 20 to 120, step 5 | Number of past daily returns fed to the LSTM per sample |
| Max training epochs | 20 | 5 to 60, step 5 | Upper bound; early stopping usually ends sooner |
| Test set size | 20% | 10% to 30%, step 5 | Most recent share of days held out for evaluation |
| Offline demo mode | off | toggle | Skip Yahoo Finance and use the bundled CSVs |
| Retrain model | button | | Discards the cached model for the current configuration and trains again |

The three model settings live in a collapsed "Model settings" expander.

Currency is inferred from the ticker: symbols ending in `.NS` or `.BO` (Indian exchanges) display `₹`,
everything else displays `$`.

---

## 6. Data layer

All functions in this section live in `stocksense/data.py`.

### 6.1 Live data: `fetch_live(ticker, start, end)`

- Calls `yf.Ticker(ticker).history(start=start, end=end + 1 day, auto_adjust=True)`. yfinance treats
  `end` as exclusive, hence the extra day. `auto_adjust=True` means prices are adjusted for splits and
  dividends.
- Raises `ValueError` if the response is empty (bad symbol, delisted, rate-limited, etc.).
- Cached with `st.cache_data(ttl=3600)`, so the same ticker and range is downloaded at most once per hour.

### 6.2 Cleaning: `_clean(df)`

Applied to both live and sample data:

1. Keep only `Open, High, Low, Close, Volume`.
2. Convert the index to datetimes, drop the timezone with `tz_localize(None)`, and normalize to
   midnight. `tz_localize(None)` keeps the exchange's local calendar date. (Using `tz_convert(None)`
   would convert to UTC first, which turns midnight IST into 18:30 UTC on the previous day and shifts
   every NSE/BSE date one day early.)
3. Name the index `Date`, drop duplicate dates (keeping the last), sort ascending.
4. Drop rows with a missing `Close`.

### 6.3 Sample data: `load_sample(ticker)`

Reads `data/<TICKER with '.' replaced by '_'>.csv`. If no file exists for the requested ticker, it
loads `data/AAPL.csv` instead (`FALLBACK_SAMPLE`) and reports which ticker was actually used. Cached
with `st.cache_data` (no expiry).

### 6.4 Choosing the source: `get_data(settings, min_rows)`

Returns `(df, source, ticker_shown, note)`:

1. If offline mode is off, try `fetch_live`. If it succeeds **and** returns at least `min_rows` rows,
   use it (`source = "live"`, no note).
2. Otherwise (offline mode, a network error, a bad ticker, or too few rows) load the bundled sample,
   slice it to the selected date range, and build a user-facing note explaining why. If the slice is
   still shorter than `min_rows`, the full sample history is used and the note says so.

The note is shown as a yellow warning under the page title, and the title badge switches from
"LIVE DATA" (green) to "SAMPLE DATA" (amber).

### 6.5 Indicators: `add_indicators(df)`

Adds 20-day and 50-day simple moving averages of `Close` (`MA20`, `MA50`). They are drawn on the
candlestick chart and summarised in the 52-week range card. They are not model inputs.

---

## 7. The LSTM model

The model code is `build_model()` and `train_lstm()` in `stocksense/model.py`, which has no Streamlit
dependency. The UI wrapper `get_trained()` is in `stocksense/training.py`.

### 7.1 Why log returns instead of prices

Raw prices trend and are non-stationary: a model trained on 2018 prices has never seen 2025 price
levels, and a model that predicts prices directly tends to learn "tomorrow ≈ today". The project
instead models **daily log returns**, `r_t = ln(Close_t / Close_{t-1})`, which are roughly stationary
and comparable across years and tickers. A predicted return is turned back into a price with
`predicted_close_t = Close_{t-1} × exp(predicted r_t)`.

### 7.2 Building the dataset

Given the closing price array `close` of length N:

```
rets        = diff(log(close))                         # length N-1; rets[i] is the return into day i+1
X[k]        = rets[k : k + lookback]                   # sliding windows (numpy sliding_window_view)
y[k]        = rets[k + lookback]                       # the next return after each window
target_day  = k + lookback + 1                         # index into `close` of the day each y[k] belongs to
```

The model therefore sees only past returns when predicting each day; there is a single input feature
(the return) per time step.

### 7.3 Chronological split and scaling

- `split_day = int(N × (1 − test_pct/100))`. Samples whose target day is before `split_day` are
  training samples; the rest form the **test set**. No shuffling across the split, so there is no
  look-ahead leakage.
- Training requires at least 100 training samples and 20 test samples; otherwise a `ValueError`
  ("Not enough data for this lookback / test split…") is shown.
- Scaling: inputs and targets are divided by the standard deviation of returns **in the training
  period only** (`scale`, with a floor of `1e-4` if it is zero). The test period never influences the
  scaler.
- Keras `validation_split=0.1` takes the **last** 10% of the training samples (Keras slices before
  shuffling), so validation is also chronological. `shuffle=True` only shuffles batches inside the
  remaining 90%.

### 7.4 Architecture

```
Input(shape=(lookback, 1))
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32)
Dropout(0.2)
Dense(16, activation="relu")
Dense(1)                         # scaled next-day log return
```

- Optimizer: Adam, learning rate `1e-3`
- Loss: mean squared error (on scaled returns)
- Batch size: 32
- Epochs: up to the "Max training epochs" setting
- Early stopping: monitors `val_loss`, `patience=5`, `restore_best_weights=True`
- Seed: `tf.keras.utils.set_random_seed(42)` before each training run, for repeatable results on the
  same data

### 7.5 Training flow in the UI: `get_trained()`

1. Builds a cache key (see [Caching](#10-caching-and-state)). Returns the cached result if present,
   unless "Retrain model" was just clicked.
2. Opens an `st.status` box with a progress bar. A custom Keras callback updates it after every epoch
   with the epoch number, training loss and validation loss.
3. On success, the status collapses to "Model ready · N epochs in X s".
4. On `ImportError` it shows "TensorFlow is not installed" with install instructions; on any other
   exception it shows "Model training failed: …". In both cases the forecast and performance tabs show
   a placeholder message instead of crashing.

### 7.6 What `train_lstm` returns

A dict holding: the Keras `model`, the `scale`, test-set residuals in return space (`resid`, actual
minus predicted), `test_dates`, `test_actual` and `test_pred` prices, the `metrics` dict, the `loss`
and `val_loss` histories, `epochs_run`, and an empty `forecasts` dict that is filled lazily per
horizon. `get_trained` adds `train_seconds`.

---

## 8. Forecasting and uncertainty

`forecast(result, close, last_date, lookback, horizon, n_paths=400, seed=7)` in `stocksense/model.py`:

1. Start from the last `lookback` returns of the full history and replicate them into 400 parallel
   windows ("paths").
2. For each step `h` of the horizon:
   - Run the model on all 400 windows at once to get the predicted return `mu` for each.
   - Add noise sampled with replacement from the **test-set residuals** (empirical bootstrap).
     Path 0 gets no noise, so it is the plain deterministic LSTM forecast.
   - Append that return to each window and drop the oldest one (recursive multi-step forecasting).
3. Convert cumulative returns to prices: `last_close × exp(cumsum(path returns))`.

Outputs:

| Key | Meaning |
|-----|---------|
| `dates` | The next `horizon` business days after the last data date (`pd.bdate_range`) |
| `point` | Path 0, the noise-free LSTM forecast. Shown as the main forecast line and table column |
| `p10`, `p50`, `p90` | 10th, 50th and 90th percentiles across the 399 noisy paths. `p10`–`p90` is the shaded "80% range" |
| `p_up` | Probability that the next day's return is positive, computed as the share of all test residuals `e` for which `mu_first + e > 0` |
| `next_price` | `last_close × exp(mu_first)`, the next-day point prediction |

The next-day trend card shows **UP** if `p_up ≥ 0.5`, otherwise **DOWN**, with the probability of the
chosen direction (`max(p_up, 1 − p_up)`).

Forecasts are stored in `result["forecasts"][horizon]`, so moving the horizon slider back and forth
only runs inference once per horizon value and never retrains.

---

## 9. Evaluation metrics

All metrics are computed on the held-out test set, in price space, in `train_lstm()`
(`stocksense/model.py`):

| Metric | Definition | How to read it |
|--------|------------|----------------|
| MAE | mean(\|pred − actual\|) | Average error in currency units |
| RMSE | sqrt(mean((pred − actual)²)) | Penalises large misses more than MAE |
| MAPE | mean(\|pred − actual\| / actual) × 100 | Average error as a percentage of price |
| Directional accuracy | share of days where sign(predicted return) = sign(actual return) | 50% is a coin flip |
| Naive MAE | mean(\|Close_{t−1} − Close_t\|) | Error of the "tomorrow = today" baseline |
| Skill vs naive | (1 − MAE / naive MAE) × 100 | Positive means the LSTM beats the naive baseline; 0% means no better |

Also recorded: `n_train`, `n_test` and `test_start` (first test date).

**Expected results.** Daily stock returns are close to a random walk, so for a price-only model,
directional accuracy slightly above 50% and a skill score near 0% are normal and honest. Price-level
metrics such as MAPE look small (often 1–2%) mainly because each prediction starts from the actual
previous close. The app presents these numbers without spin and explains them in the "How the model
works" expander.

---

## 10. Caching and state

| What | Mechanism | Key | Lifetime |
|------|-----------|-----|----------|
| Live downloads | `st.cache_data(ttl=3600)` on `data.fetch_live` | ticker, start, end | 1 hour |
| Sample CSVs | `st.cache_data` on `data.load_sample` | ticker | Server process |
| Trained models | Plain dict returned by `st.cache_resource` `training.model_store()` | `(ticker, source, first date, last date, row count, last close rounded to 4 dp, lookback, epochs, test_pct)` | Server process, shared across all browser sessions |
| Forecasts | `result["forecasts"][horizon]` inside each cached model | horizon | Same as the model |

The model store holds at most **8** models. When a ninth is added, the oldest inserted entry is
evicted (FIFO). Nothing is written to disk; restarting the server clears every cache.

The "Retrain model" button is true only for the rerun triggered by the click, so it removes the
current key from the store once and trains a fresh model.

Because the key includes the data's last date and last close, a new trading day of live data (after
the 1-hour data cache expires) automatically leads to a new model.

---

## 11. User interface

The header and metric cards are in `stocksense/ui/header.py`, the tab contents in
`stocksense/ui/tabs.py`, and the figures in `stocksense/charts.py`.

### Header

- Company name (from the `POPULAR` map, or the raw symbol for custom tickers) and the ticker.
- Badges: `LIVE DATA` or `SAMPLE DATA`, last close date, number of trading days loaded.
- Optional yellow warning with the data-source note.

### Market metric cards (always visible)

1. **Latest close** with its date.
2. **Daily change** in currency, with an up/down arrow, versus the previous close.
3. **Percent change** (1-day return).
4. **Volume** (formatted as K/M/B) and its % difference from the 20-day average.
5. **52-week range** (min Low to max High over the last year) with the current MA20 and MA50.

Positive values are green (`#00d4a0`), negative values red (`#ff4d6d`).

### Tabs

1. **📈 Price chart** — Plotly candlestick with MA20 (amber) and MA50 (violet) overlays and a volume
   bar panel colored by up/down day. Weekends are removed from the x-axis (`rangebreaks`). The view
   opens on the last year; range buttons switch between 1M, 3M, 6M, YTD, 1Y and All.
2. **🔮 Forecast**
   - Next-day trend card (UP/DOWN, probability, predicted next close and % change, N-day outlook,
     and a "not financial advice" disclaimer).
   - Probability gauge (0–100%, red half below 50, green half above).
   - "Predicted vs historical" chart: actual close from 90 days before the test period onward, the
     dotted LSTM test-set predictions, a shaded test-period region, the forecast line with the 80%
     band, and the last close marker.
   - Forecast table (date, forecast, 10th and 90th percentile, % change vs last close) with a
     **Download forecast (CSV)** button (`forecast.csv`).
3. **📊 Model performance**
   - Caption stating the number of test days and the test start date.
   - Cards for MAE, RMSE, MAPE, directional accuracy and skill vs naive.
   - Actual vs predicted chart for the test set.
   - Training and validation loss curves per epoch.
   - Histogram of percentage prediction errors (40 bins).
   - "How the model works" expander that fills in the live settings (lookback, epochs run, sample
     count, training time, test %).
4. **🗂️ Data** — Table of OHLCV plus MA20/MA50, newest first, rounded to 2 dp, with a
   **Download data (CSV)** button (`<TICKER>_history.csv`).

A footer repeats the educational-use disclaimer.

---

## 12. Theming and styling

`.streamlit/config.toml` sets:

- Dark base theme, primary color `#7c5cff`, background `#0b0f19`, secondary background `#141a2a`,
  text `#e5e7eb`, sans-serif font.
- `server.headless = true` (no browser auto-open prompt), `runOnSave = false`.
- `browser.gatherUsageStats = false` (no Streamlit telemetry).
- `client.toolbarMode = "minimal"`.

`dashboard.main()` also injects a `<style>` block (`CSS` in `stocksense/styles.py`) that adds the Inter font, radial-gradient page
background, gradient brand title, rounded glass-style metric cards, the green/red trend card,
badges, tab styling, and hides Streamlit's main menu and footer. All charts go through `style_fig()`,
which applies `plotly_dark`, transparent backgrounds, faint grid lines, a horizontal legend and
unified x-hover. Plotly's logo and lasso/box-select tools are removed via `PLOTLY_CONFIG`.

Color constants (in `stocksense/config.py`): `UP #00d4a0`, `DOWN #ff4d6d`, `ACCENT #7c5cff`, `CYAN #22d3ee`, `AMBER #fbbf24`,
`TEXT #e5e7eb`, `MUTED #8b95a7`.

---

## 13. Offline sample data

### Bundled files

Seven CSVs in `data/` with columns `Date,Open,High,Low,Close,Volume`, split- and dividend-adjusted,
rounded to 4 decimal places:

| File | Ticker | Rows | Date range |
|------|--------|------|------------|
| `AAPL.csv` | AAPL | 2,715 | 2016-01-04 to 2026-10-20 |
| `MSFT.csv` | MSFT | 2,715 | 2016-01-04 to 2026-10-20 |
| `GOOGL.csv` | GOOGL | 2,715 | 2016-01-04 to 2026-10-20 |
| `NVDA.csv` | NVDA | 2,715 | 2016-01-04 to 2026-10-20 |
| `TSLA.csv` | TSLA | 2,715 | 2016-01-04 to 2026-10-20 |
| `RELIANCE_NS.csv` | RELIANCE.NS | 2,667 | 2016-01-01 to 2026-10-19 |
| `TCS_NS.csv` | TCS.NS | 2,667 | 2016-01-01 to 2026-10-19 |

The current files contain real Yahoo Finance data. The NSE files contain three weekend dates on
purpose: 2019-10-27 and 2020-11-14 (Diwali Muhurat trading) and 2025-02-01 (the Union Budget
Saturday session).

AMZN, META, INFY.NS and HDFCBANK.NS are in the ticker dropdown but have no bundled file. In offline
mode, or when live data fails for them, the app shows the AAPL sample and says so.

### Refreshing: `scripts/build_sample_data.py`

```bash
python scripts/build_sample_data.py
```

The script is standalone (it does not import `stocksense`) and writes to the `data/` folder in the
project root, so it can be run from any working directory.

For each ticker in its `TICKERS` dict, it downloads adjusted daily history from 2016-01-01 to today
and writes `data/<TICKER>.csv`. If a download fails (no internet, rate limit), it writes a **synthetic**
series instead so offline mode always has data: geometric Brownian motion with a drift of 0.06% and
volatility of 1.8% per day, starting from a hard-coded realistic 2016 price, with Open/High/Low and
log-normal Volume derived from it. The script prints whether each file is `live` or `synthetic`.

To bundle another ticker, add it to `TICKERS` in `scripts/build_sample_data.py` (value = a plausible
2016 starting price for the synthetic fallback) and rerun the script. To make it selectable without
typing it as a custom ticker, also add it to `POPULAR` in `stocksense/config.py`.

---

## 14. Error handling

| Situation | What the user sees |
|-----------|--------------------|
| Start date ≥ end date | Red error, app stops |
| No internet / Yahoo error / bad symbol | Yellow warning, sample data shown, `SAMPLE DATA` badge |
| Live data shorter than `lookback + 200` days | Yellow warning suggesting a longer range, sample data shown |
| No sample file for the ticker | Warning that the AAPL sample is shown instead |
| `data/` folder missing and live data unavailable | Red error "Could not load live or sample data… Check that the `data/` folder is in the project root.", app stops |
| Data too short to train | Info message in the Forecast and Performance tabs; chart and data tabs still work |
| TensorFlow not installed | Error in the training status box with the `pip install` command |
| Training raises an exception | "Model training failed: …" error; other tabs keep working |

---

## 15. Module map

| File | Responsibility | Public names |
|------|----------------|--------------|
| `app.py` | Entry point for `streamlit run` | calls `main()` |
| `stocksense/__init__.py` | TensorFlow log env vars | |
| `stocksense/config.py` | Constants and settings | `DATA_DIR`, `FALLBACK_SAMPLE`, `POPULAR`, `CUSTOM`, `UP`, `DOWN`, `ACCENT`, `CYAN`, `AMBER`, `TEXT`, `MUTED`, `PLOTLY_CONFIG`, `Settings` |
| `stocksense/styles.py` | Page CSS | `CSS` |
| `stocksense/formatting.py` | Text helpers | `currency`, `fmt_volume`, `tone` |
| `stocksense/data.py` | Data loading | `_clean`, `fetch_live`, `load_sample`, `get_data`, `add_indicators` |
| `stocksense/model.py` | LSTM and forecasting (no Streamlit) | `build_model`, `train_lstm`, `forecast` |
| `stocksense/training.py` | Training UI and model cache | `model_store`, `get_trained` |
| `stocksense/charts.py` | Plotly figures | `style_fig`, `candlestick_chart`, `forecast_chart`, `gauge`, `test_chart`, `loss_chart`, `error_hist` |
| `stocksense/dashboard.py` | Page assembly | `main` |
| `stocksense/ui/components.py` | HTML metric cards | `card`, `render_cards` |
| `stocksense/ui/sidebar.py` | Sidebar controls | `sidebar` |
| `stocksense/ui/header.py` | Title, badges, market cards | `hero`, `market_cards` |
| `stocksense/ui/tabs.py` | Tab contents | `forecast_tab`, `performance_tab`, `data_tab` |
| `scripts/build_sample_data.py` | Offline CSV refresh (standalone) | `download`, `synthetic`, `main` |

To exercise the app headlessly (for example in a smoke test), use Streamlit's `AppTest` with the project
root on `PYTHONPATH`, because `AppTest`, unlike `streamlit run`, does not add the script directory to
`sys.path`:

```bash
PYTHONPATH=. python -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('app.py', default_timeout=300); at.run(); print(at.exception)"
```

---

## 16. Known limitations and quirks

- **Price-only, single-feature model.** The LSTM sees only past closing-price returns. It has no
  volume, fundamentals, news, macro data or cross-asset information, so it cannot anticipate events.
- **Near-random-walk target.** Expect skill near 0% against the naive baseline. This is a property of
  the problem, not a bug.
- **Recursive forecasts degrade.** Multi-day forecasts feed predictions back as inputs. The point
  forecast tends to flatten toward the mean return, and the band widens with the horizon.
- **Uncertainty band assumptions.** The band bootstraps test residuals independently at each step, so
  it ignores volatility clustering and fat-tail dependence. Treat it as indicative.
- **Forecast dates ignore exchange holidays.** `pd.bdate_range` skips weekends only, so a forecast
  date may fall on a market holiday.
- **Synthetic data is not reproducible across runs.** `build_sample_data.synthetic` seeds its RNG
  with Python's `hash(ticker)`, which is randomised per process unless `PYTHONHASHSEED` is set.
- **In-memory model cache.** Trained models are not saved to disk and are shared by all users of one
  server process. A restart retrains everything; more than 8 configurations evicts the oldest.
- **Training blocks the session.** Training runs synchronously in the Streamlit script thread.
- **Custom ticker text is placed into HTML** (`unsafe_allow_html`) in the header. It is only reflected
  back to the same user, but escape it if the app is ever deployed publicly.
- **Yahoo Finance is unofficial.** yfinance scrapes Yahoo endpoints, which can change or rate-limit
  without notice; the offline fallback exists for this reason.

---

## 17. Extending the project

- **Add tickers to the dropdown:** edit `POPULAR` in `stocksense/config.py` (symbol → display name).
- **Change the network:** edit `build_model(lookback)` in `stocksense/model.py`. Inputs are
  `(batch, lookback, 1)` scaled returns; the output must be one scaled return.
- **Add input features** (e.g. volume change, MA spreads): build a `(samples, lookback, n_features)`
  array in `train_lstm`, update the `Input` shape, and update the window that `forecast` rolls forward
  (it must be able to generate future values of every feature).
- **Different metrics:** add entries to the `metrics` dict in `train_lstm` (`stocksense/model.py`) and
  a `card(...)` in `performance_tab` (`stocksense/ui/tabs.py`).
- **Persist models:** save `result["model"]` with `model.save(...)` keyed by the same tuple used in
  `get_trained` (`stocksense/training.py`), and load it before training.
- **New chart:** add a figure function to `stocksense/charts.py` and pass the figure through
  `style_fig()` so it matches the theme.
- **New tab:** add a render function to `stocksense/ui/tabs.py`, then add a label to the
  `st.tabs([...])` list in `stocksense/dashboard.py` and a corresponding `with tabs[i]:` block.

---

## 18. Presentation tips

- Open the app once before presenting. The first model for each configuration trains in about 20 to
  40 seconds on a laptop CPU and is cached afterwards. Pre-train every ticker you plan to show.
- If the venue Wi-Fi is unreliable, turn on **Offline demo mode** in the sidebar. Bundled samples:
  AAPL, MSFT, GOOGL, NVDA, TSLA, RELIANCE.NS and TCS.NS.
- Changing the prediction period is instant; changing lookback, epochs, test size, ticker or dates
  trains a new model.
- Refresh the bundled samples with `python scripts/build_sample_data.py` while online.
- When presenting the metrics, frame directional accuracy around 50% and skill near 0% as the honest
  expected outcome for price-only daily forecasting.

---

## 19. Disclaimer

StockSense is an educational deep-learning project. Its predictions are statistical estimates based
only on past prices and are **not financial advice**. Do not make investment decisions based on it.
