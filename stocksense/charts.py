"""Plotly figures used by the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ACCENT, AMBER, CYAN, DOWN, MUTED, TEXT, UP


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", color="#d1d5db"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False)
    return fig


def candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker,
            increasing=dict(line=dict(color=UP), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN),
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA 20", line=dict(color=AMBER, width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA 50", line=dict(color=ACCENT, width=1.8)), row=1, col=1)
    vol_colors = np.where(df["Close"] >= df["Open"], "rgba(0,212,160,.55)", "rgba(255,77,109,.55)")
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, showlegend=False), row=2, col=1)

    # Open on the last year by default; the range buttons zoom in or out.
    start = max(df.index[0], df.index[-1] - pd.DateOffset(years=1))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], range=[start, df.index[-1] + pd.Timedelta(days=2)])
    fig.update_xaxes(
        row=1, col=1, rangeslider_visible=False,
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="rgba(255,255,255,.06)", activecolor=ACCENT, font=dict(color=TEXT),
            x=1, xanchor="right", y=1.02, yanchor="bottom",
        ),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    style_fig(fig, 620)
    fig.update_layout(legend=dict(x=0, xanchor="left"))
    return fig


def forecast_chart(df: pd.DataFrame, res: dict, fc: dict) -> go.Figure:
    start = res["metrics"]["test_start"] - pd.Timedelta(days=90)
    hist = df.loc[start:]
    last_date, last_close = df.index[-1], float(df["Close"].iloc[-1])
    fc_x = [last_date, *fc["dates"]]

    fig = go.Figure()
    fig.add_vrect(x0=res["metrics"]["test_start"], x1=last_date, fillcolor="rgba(255,255,255,.035)", line_width=0)
    fig.add_annotation(x=res["metrics"]["test_start"], y=1, yref="paper", text="Test period", showarrow=False,
                       xanchor="left", yanchor="top", font=dict(color=MUTED, size=11))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Historical close", line=dict(color="#e5e7eb", width=1.8)))
    fig.add_trace(go.Scatter(x=res["test_dates"], y=res["test_pred"], name="LSTM prediction (test)",
                             line=dict(color=CYAN, width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=fc_x, y=[last_close, *fc["p90"]], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fc_x, y=[last_close, *fc["p10"]], fill="tonexty", fillcolor="rgba(124,92,255,.22)",
                             line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc_x, y=[last_close, *fc["point"]], name="Forecast",
                             line=dict(color=ACCENT, width=2.6), mode="lines+markers", marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[last_date], y=[last_close], mode="markers", name="Last close",
                             marker=dict(color=AMBER, size=10, line=dict(color="#0b0f19", width=2))))
    return style_fig(fig, 480)


def gauge(p_up: float) -> go.Figure:
    color = UP if p_up >= 0.5 else DOWN
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p_up * 100, number=dict(suffix="%", font=dict(size=34, color=TEXT)),
        title=dict(text="Probability of UP move", font=dict(size=13, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickwidth=1),
            bar=dict(color=color, thickness=0.28),
            bgcolor="rgba(255,255,255,.03)", borderwidth=0,
            steps=[dict(range=[0, 50], color="rgba(255,77,109,.12)"), dict(range=[50, 100], color="rgba(0,212,160,.12)")],
            threshold=dict(line=dict(color=TEXT, width=2), thickness=0.8, value=50),
        ),
    ))
    style_fig(fig, 230)
    fig.update_layout(margin=dict(l=25, r=25, t=50, b=0))
    return fig


def test_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["test_dates"], y=res["test_actual"], name="Actual", line=dict(color="#e5e7eb", width=1.8)))
    fig.add_trace(go.Scatter(x=res["test_dates"], y=res["test_pred"], name="Predicted", line=dict(color=CYAN, width=1.6)))
    return style_fig(fig, 380)


def loss_chart(res: dict) -> go.Figure:
    ep = list(range(1, len(res["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=res["loss"], name="Training loss", line=dict(color=ACCENT, width=2)))
    if res["val_loss"]:
        fig.add_trace(go.Scatter(x=ep, y=res["val_loss"], name="Validation loss", line=dict(color=AMBER, width=2)))
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="MSE (scaled)")
    return style_fig(fig, 300)


def error_hist(res: dict) -> go.Figure:
    pct_err = (res["test_pred"] - res["test_actual"]) / res["test_actual"] * 100
    fig = go.Figure(go.Histogram(x=pct_err, nbinsx=40, marker_color=CYAN, opacity=0.8, name="Error"))
    fig.update_xaxes(title_text="Prediction error (%)")
    fig.update_yaxes(title_text="Days")
    style_fig(fig, 300)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig
