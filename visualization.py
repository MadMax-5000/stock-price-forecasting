"""
Visualization Module for Stock Price Prediction.

This module handles:
- Plotting historical stock data
- Overlaying future predictions
- Interactive charts with Plotly

Author: madmax
Version: 2.0
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_historical_with_predictions(
    historical: pd.DataFrame,
    predictions: pd.DataFrame,
    ticker: str = "STOCK",
    show_volume: bool = False,
    title: str | None = None,
) -> Any:
    """
    Plot historical stock data with future predictions.

    Args:
        historical: DataFrame with 'Close' column and DatetimeIndex.
        predictions: DataFrame with 'Close' and 'Prediction' columns.
        ticker: Stock ticker symbol.
        show_volume: Whether to show volume subplot.
        title: Custom title. If None, uses default.

    Returns:
        Plotly figure object.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization. Install with: pip install plotly")

    if title is None:
        title = f"{ticker} - Historical Data with Predictions"

    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1 if show_volume else 0,
        subplot_titles=("Price", "Volume") if show_volume else None,
    )

    historical = historical.copy()
    if isinstance(historical.index, pd.DatetimeIndex):
        hist_dates = historical.index.tolist()
    else:
        hist_dates = list(historical.index)

    hist_close = historical["Close"].values

    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_close,
            mode="lines",
            name="Historical",
            line=dict(color="#2196F3", width=1.5),
            hovertemplate="$%{y:.2f}<extra>Historical</extra>",
        ),
        row=1, col=1,
    )

    if predictions is not None and len(predictions) > 0:
        predictions = predictions.copy()
        if isinstance(predictions.index, pd.DatetimeIndex):
            pred_dates = predictions.index.tolist()
        else:
            pred_dates = list(predictions.index)

        pred_close = predictions["Close"].values

        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_close,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#FF5722", width=2),
                marker=dict(size=4, symbol="diamond"),
                hovertemplate="$%{y:.2f}<extra>Predicted</extra>",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[hist_dates[-1], pred_dates[0]],
                y=[hist_close[-1], pred_close[0]],
                mode="lines",
                name="Connection",
                line=dict(color="#FF5722", width=1, dash="dot"),
                showlegend=True,
            ),
            row=1, col=1,
        )

    if show_volume and "Volume" in historical.columns:
        volumes = historical["Volume"].values

        fig.add_trace(
            go.Bar(
                x=hist_dates,
                y=volumes,
                name="Volume",
                marker_color="#90A4AE",
                opacity=0.5,
                hovertemplate="%{y:,}<extra>Volume</extra>",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=500 if show_volume else 400,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=80, b=60),
    )

    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")

    return fig


def plot_candlestick(
    data: pd.DataFrame,
    ticker: str = "STOCK",
    title: str | None = None,
) -> Any:
    """
    Create candlestick chart for OHLC data.

    Args:
        data: DataFrame with Open, High, Low, Close columns.
        ticker: Stock ticker symbol.
        title: Custom title.

    Returns:
        Plotly figure object.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization")

    if title is None:
        title = f"{ticker} - Candlestick Chart"

    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker,
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")

    fig.update_layout(xaxis_rangeslider_visible=False)

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "Precision",
    title: str | None = None,
) -> Any:
    """
    Plot model comparison as horizontal bar chart.

    Args:
        comparison_df: DataFrame with Model and metric columns.
        metric: Metric to plot (Precision, Accuracy, Recall, F1).
        title: Custom title.

    Returns:
        Plotly figure object.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization")

    if title is None:
        title = f"Model Comparison by {metric}"

    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison_df")

    df_sorted = comparison_df.sort_values(by=metric, ascending=True)

    colors = ["#FF5722" if x == df_sorted[metric].max() else "#2196F3" 
             for x in df_sorted[metric]]

    fig = go.Figure(data=[
        go.Bar(
            x=df_sorted[metric],
            y=df_sorted["Model"],
            orientation="h",
            marker_color=colors,
            text=df_sorted[metric].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title=metric,
        yaxis_title="Model",
        template="plotly_white",
        height=max(300, len(df_sorted) * 30),
        hovermode="y unified",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0", 
                  range=[0, max(df_sorted[metric]) * 1.2])

    return fig


def plot_technical_indicators(
    data: pd.DataFrame,
    ticker: str = "STOCK",
) -> Any:
    """
    Plot technical indicators (RSI, MACD, Bollinger Bands).

    Args:
        data: DataFrame with technical indicator columns.
        ticker: Stock ticker symbol.

    Returns:
        Plotly figure object with subplots.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price with Bollinger Bands", "RSI", "MACD"),
        row_heights=[0.5, 0.2, 0.2],
    )

    if "Close" in data.columns:
        close = data["Close"]
        
        fig.add_trace(
            go.Scatter(
                x=data.index, y=close,
                mode="lines", name="Close",
                line=dict(color="#2196F3", width=1.5),
            ),
            row=1, col=1
        )

        if "BB_Upper" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data["BB_Upper"],
                    mode="lines", name="BB Upper",
                    line=dict(color="#90A4AE", width=1, dash="dash"),
                    showlegend=False,
                ),
                row=1, col=1
            )

        if "BB_Lower" in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data["BB_Lower"],
                    mode="lines", name="BB Lower",
                    line=dict(color="#90A4AE", width=1, dash="dash"),
                    showlegend=False,
                ),
                row=1, col=1
            )

    if "RSI" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["RSI"],
                mode="lines", name="RSI",
                line=dict(color="#9C27B0", width=1.5),
            ),
            row=2, col=1
        )

        fig.add_hline(y=70, line_dash="dot", line_color="red", 
                     row=2, col=1, annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dot", line_color="green", 
                     row=2, col=1, annotation_text="Oversold")

    if "MACD" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["MACD"],
                mode="lines", name="MACD",
                line=dict(color="#FF5722", width=1.5),
            ),
            row=3, col=1
        )

    if "Signal_Line" in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data["Signal_Line"],
                mode="lines", name="Signal",
                line=dict(color="#4CAF50", width=1.5),
            ),
            row=3, col=1
        )

    fig.update_layout(
        title=dict(text=f"{ticker} - Technical Indicators", font=dict(size=20)),
        xaxis_title="Date",
        template="plotly_white",
        height=700,
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


def plot_returns_distribution(
    data: pd.DataFrame,
    ticker: str = "STOCK",
) -> Any:
    """
    Plot returns distribution histogram.

    Args:
        data: DataFrame with 'Returns' column.
        ticker: Stock ticker symbol.

    Returns:
        Plotly figure object.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for visualization")

    if "Returns" not in data.columns:
        raise ValueError("DataFrame must contain 'Returns' column")

    returns = data["Returns"].dropna()

    fig = go.Figure(data=[
        go.Histogram(
            x=returns,
            nbinsx=50,
            marker_color="#2196F3",
            opacity=0.7,
            name="Returns",
        )
    ])

    fig.update_layout(
        title=dict(text=f"{ticker} - Daily Returns Distribution", font=dict(size=20)),
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
    )

    fig.add_vline(x=returns.mean(), line_dash="dot", line_color="red",
                  annotation_text=f"Mean: {returns.mean():.4f}")
    fig.add_vline(x=0, line_dash="solid", line_color="black", annotation_text="0%")

    return fig


def save_plot(fig: Any, filename: str, format: str = "png") -> None:
    """
    Save plot to file.

    Args:
        fig: Plotly figure object.
        filename: Output filename.
        format: Output format (png, html, svg, pdf).
    """
    if format == "html":
        fig.write_html(filename)
    else:
        fig.write_image(filename, width=1200, height=800, scale=2)


if __name__ == "__main__":
    print("Visualization module loaded.")
    print("Available functions:")
    print("  - plot_historical_with_predictions()")
    print("  - plot_candlestick()")
    print("  - plot_model_comparison()")
    print("  - plot_technical_indicators()")
    print("  - plot_returns_distribution()")
    print("  - save_plot()")