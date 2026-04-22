"""
Data collection module for downloading stock data from Yahoo Finance.

The main goal of this project is to answer this question:
Can I predict whether Apple's stock will go up or down tomorrow and explain the factors driving it?

Author: madmax
Version: 1.0
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf

STOCKS: list[str] = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NFLX",
    "NVDA",
    "AMD",
    "TSLA",
    "JPM",
    "GS",
    "WMT",
    "KO",
    "NKE",
    "XOM",
    "BA",
]


def download_stock_data(
    symbol: str,
    start_date: str = (date.today().replace(year=date.today().year - 10)).isoformat(),
    end_date: str = date.today().isoformat(),
    output_dir: str = "data",
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance for a given symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        output_dir: Directory to save the CSV file.

    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume).

    Raises:
        ValueError: If no data is returned for the symbol.
        yfinance.exceptions.YFinanceException: If download fails.
    """
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.reset_index()

    output_path = f"{output_dir}/{symbol.lower()}_stock_data.csv"
    data.to_csv(output_path, index=False)

    return data


def download_all_stocks(
    start_date: str = (date.today().replace(year=date.today().year - 10)).isoformat(),
    end_date: str = date.today().isoformat(),
    output_dir: str = "data",
) -> dict[str, pd.DataFrame]:
    """
    Download stock data for all predefined stock symbols.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        output_dir: Directory to save CSV files.

    Returns:
        Dictionary mapping stock symbols to their respective DataFrames.
    """
    results: dict[str, pd.DataFrame] = {}

    for symbol in STOCKS:
        data = download_stock_data(symbol, start_date, end_date, output_dir)
        results[symbol] = data
        print(f"{symbol}: {len(data)} rows saved")

    print("All stock data downloaded successfully!")
    return results


if __name__ == "__main__":
    download_all_stocks()
