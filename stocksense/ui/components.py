"""Reusable HTML building blocks."""
from __future__ import annotations

import streamlit as st


def card(label: str, value: str, sub: str = "", value_tone: str = "", sub_tone: str = "") -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {value_tone}">{value}</div>'
        f'<div class="metric-sub {sub_tone}">{sub}</div></div>'
    )


def render_cards(cards: list[str]) -> None:
    for col, html in zip(st.columns(len(cards)), cards):
        col.markdown(html, unsafe_allow_html=True)
