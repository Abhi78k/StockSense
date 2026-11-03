"""Constants and the user settings shared across the app."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
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
