"""Page title, data-source badges and market metric cards."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..config import POPULAR
from ..formatting import fmt_volume, tone
from .components import card, render_cards


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
