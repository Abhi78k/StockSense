"""Small text formatting helpers."""
from __future__ import annotations


def currency(ticker: str) -> str:
    return "₹" if ticker.upper().endswith((".NS", ".BO")) else "$"


def fmt_volume(v: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if v >= div:
            return f"{v / div:.2f}{unit}"
    return f"{v:,.0f}"


def tone(x: float) -> str:
    return "up" if x >= 0 else "down"
