"""
Feature engineering module for stock price prediction.

This module generates technical indicators and features for ML models including:
- Price-based features (returns, log returns, volatility)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Rolling horizon features (price ratios, trend signals)
- Calendar features (day of week, month)

Author: madmax
Version: 1.0
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_HORIZONS: list[int] = [2, 5, 60, 250]


def validate_datetime_index(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has a DatetimeIndex.

    Args:
        df: DataFrame to validate.

    Returns:
        True if index is DatetimeIndex, False otherwise.
    """
    return isinstance(df.index, pd.DatetimeIndex)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add base price-based and technical indicator features.

    Features added:
        - Tomorrow: Next day's closing price (for target calculation)
        - Target: Binary target (1 if price goes up, 0 otherwise)
        - returns: Daily percentage returns
        - Log_Returns: Logarithmic returns
        - High_Low_Pct: Normalized high-low range
        - Gap_Pct: Gap percentage from previous close
        - Volume_Pct_Change: Volume percentage change
        - RSI: Relative Strength Index (14-period)
        - EMA_12: 12-period Exponential Moving Average
        - EMA_26: 26-period Exponential Moving Average
        - MACD: Moving Average Convergence Divergence
        - Signal_line: MACD signal line (9-period EMA)
        - BB_Middle: Bollinger Bands middle band (20-period SMA)
        - BB_Std: Bollinger Bands standard deviation
        - BB_Upper: Bollinger Bands upper band
        - BB_Lower: Bollinger Bands lower band
        - BB_Position: Normalized position within Bollinger Bands
        - Volatility_10: 10-day rolling volatility

    Args:
        df: Input DataFrame with Close, High, Low, Open, Volume columns.

    Returns:
        DataFrame with base features added.
    """
    result_df = df.copy()

    result_df["Target"] = (result_df["Close"].shift(-1) > result_df["Close"]).astype(int)

    result_df["returns"] = result_df["Close"].pct_change()
    result_df["Log_Returns"] = np.log(result_df["Close"] / result_df["Close"].shift(1))

    result_df["High_Low_Pct"] = (result_df["High"] - result_df["Low"]) / result_df["Close"]
    result_df["Gap_Pct"] = (result_df["Open"] - result_df["Close"].shift(1)) / result_df[
        "Close"
    ].shift(1)

    result_df["Volume_Pct_Change"] = result_df["Volume"].pct_change()

    delta = result_df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result_df["RSI"] = 100 - (100 / (1 + rs))

    result_df["EMA_12"] = result_df["Close"].ewm(span=12, adjust=False).mean()
    result_df["EMA_26"] = result_df["Close"].ewm(span=26, adjust=False).mean()
    result_df["MACD"] = result_df["EMA_12"] - result_df["EMA_26"]
    result_df["Signal_line"] = result_df["MACD"].ewm(span=9, adjust=False).mean()

    result_df["BB_Middle"] = result_df["Close"].rolling(window=20).mean()
    result_df["BB_Std"] = result_df["Close"].rolling(window=20).std()
    result_df["BB_Upper"] = result_df["BB_Middle"] + (2 * result_df["BB_Std"])
    result_df["BB_Lower"] = result_df["BB_Middle"] - (2 * result_df["BB_Std"])
    result_df["BB_Position"] = (result_df["Close"] - result_df["BB_Lower"]) / (
        result_df["BB_Upper"] - result_df["BB_Lower"]
    )

    result_df["Volatility_10"] = result_df["returns"].rolling(window=10).std()

    return result_df


def add_rolling_horizon_features(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling horizon features for multiple time windows.

    For each horizon, adds:
        - Close_Ratio_{horizon}: Price / rolling average ratio
        - Trend_{horizon}: Sum of previous target values (shifted by 1)

    Args:
        df: Input DataFrame with Close and Target columns.
        horizons: List of horizon values. Defaults to DEFAULT_HORIZONS.

    Returns:
        DataFrame with rolling horizon features added.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    result_df = df.copy()

    for horizon in horizons:
        rolling_average = result_df["Close"].rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        result_df[ratio_column] = result_df["Close"] / rolling_average

        trend_column = f"Trend_{horizon}"
        result_df[trend_column] = result_df["Target"].shift(2).rolling(horizon).sum()

    return result_df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features from datetime index.

    Features added:
        - Day_of_Week: Day of week (0=Monday, 6=Sunday)
        - Month: Month of year (1-12)

    Args:
        df: Input DataFrame with DatetimeIndex.

    Returns:
        DataFrame with calendar features added.
    """
    result_df = df.copy()
    result_df["Day_of_Week"] = result_df.index.to_series().dt.dayofweek
    result_df["Month"] = result_df.index.to_series().dt.month
    return result_df


def add_features(
    dataframe: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline.

    Pipeline steps:
        1. Set Date as index if present
        2. Add base features (returns, technical indicators)
        3. Add rolling horizon features
        4. Add calendar features
        5. Remove intermediate columns (EMA_12, EMA_26, BB_Upper, BB_Lower, Tomorrow)
        6. Drop rows with NaN values

    Args:
        dataframe: Input DataFrame with OHLCV columns.
        horizons: List of horizon values for rolling features.

    Returns:
        Feature-engineered DataFrame ready for modeling.
    """
    df = dataframe.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

    if not validate_datetime_index(df):
        print("Warning: Index is not Datetime. Date features may fail.")

    df = add_base_features(df)
    df = add_rolling_horizon_features(df, horizons)
    df = add_calendar_features(df)

    cols_to_drop = ["EMA_12", "EMA_26", "BB_Upper", "BB_Lower"]
    df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)

    df.dropna(inplace=True)

    return df


def get_processed_data(
    source_path: str = "data/apple_stock_data.csv",
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load, clean, and process stock data.

    Args:
        source_path: Path to the raw stock data CSV.
        horizons: List of horizon values for rolling features.

    Returns:
        Processed DataFrame with all features.
    """
    from preparing.data_cleaning import load_and_clean_data

    raw_df = load_and_clean_data(source_path, remove_outliers=False)
    return add_features(raw_df, horizons)


if __name__ == "__main__":
    final_df = get_processed_data()
    print("Feature Engineering Complete")
    print(f"Data Shape: {final_df.shape}")
    print(final_df.head())
    final_df.to_csv("ready_data.csv")
