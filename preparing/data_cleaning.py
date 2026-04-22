"""
Data cleaning module for stock price data.

This module handles:
- Missing or Null Values
- Correct Data Types
- Duplicates
- Sort by Date
- Outlier Detection (IQR and Rolling IQR methods)
- Set Date as Index

Author: madmax
Version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd


class MissingValuesInfo(TypedDict):
    """Type definition for missing values information."""

    count: pd.Series
    has_missing: bool
    total_missing: int


class OutlierInfo(TypedDict):
    """Type definition for outlier information."""

    static_outliers: pd.DataFrame
    rolling_outliers: pd.DataFrame
    inconsistent_rows: pd.DataFrame


@dataclass
class DataQualityReport:
    """Data quality report containing validation results."""

    missing_values: MissingValuesInfo
    type_mismatches: dict[str, tuple[str, str]]
    duplicate_count: int
    outlier_count: int
    inconsistent_count: int
    is_clean: bool


PRICE_COLUMNS: list[str] = ["Open", "High", "Low", "Close"]
EXPECTED_TYPES: dict[str, str] = {
    "Date": "datetime64[ns]",
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Volume": "int64",
}


def get_missing_values_info(df: pd.DataFrame) -> MissingValuesInfo:
    """
    Analyze missing values in the DataFrame.

    Args:
        df: Input DataFrame to analyze.

    Returns:
        Dictionary containing:
            - count: Series with missing value counts per column
            - has_missing: Boolean indicating if any missing values exist
            - total_missing: Total count of missing values
    """
    return {
        "count": df.isna().sum(),
        "has_missing": bool(df.isna().any().any()),
        "total_missing": int(df.isna().sum().sum()),
    }


def validate_data_types(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """
    Validate that columns have expected data types.

    Args:
        df: Input DataFrame to validate.

    Returns:
        Dictionary mapping column names to (actual_type, expected_type) tuples
        for mismatched columns only.
    """
    mismatches: dict[str, tuple[str, str]] = {}
    for col, expected in EXPECTED_TYPES.items():
        actual = str(df[col].dtype)
        if actual != expected:
            mismatches[col] = (actual, expected)
    return mismatches


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Args:
        df: Input DataFrame.
        columns: List of columns to check for outliers. Defaults to PRICE_COLUMNS.

    Returns:
        DataFrame containing only outlier rows.
    """
    if columns is None:
        columns = PRICE_COLUMNS

    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[columns] < lower_bound) | (df[columns] > upper_bound)
    return df[outliers.any(axis=1)]


def detect_outliers_rolling_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    window: int = 252,
) -> pd.DataFrame:
    """
    Detect outliers using a rolling Interquartile Range (IQR) method.

    Uses approximately 1 trading year (252 days) as the window size.

    Args:
        df: Input DataFrame.
        columns: List of columns to check for outliers. Defaults to PRICE_COLUMNS.
        window: Rolling window size in days. Defaults to 252 (~1 trading year).

    Returns:
        DataFrame with outlier flag columns added.
    """
    if columns is None:
        columns = PRICE_COLUMNS

    result_df = df.copy()

    for col in columns:
        rolling_Q1 = df[col].rolling(window).quantile(0.25)
        rolling_Q3 = df[col].rolling(window).quantile(0.75)
        rolling_IQR = rolling_Q3 - rolling_Q1

        outliers_col = (
            (df[col] < rolling_Q1 - 1.5 * rolling_IQR)
            | (df[col] > rolling_Q3 + 1.5 * rolling_IQR)
        )
        result_df[f"{col}_outlier"] = outliers_col

    return result_df


def get_outlier_info(df: pd.DataFrame) -> OutlierInfo:
    """
    Get comprehensive outlier information using both static and rolling IQR methods.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing static outliers, rolling outliers, and inconsistent rows.
    """
    df_with_flags = detect_outliers_rolling_iqr(df)

    outlier_cols = [f"{col}_outlier" for col in PRICE_COLUMNS]
    rolling_outliers = df_with_flags[df_with_flags[outlier_cols].any(axis=1)]

    static_outliers = detect_outliers_iqr(df)

    inconsistent = df[
        (df["Low"] > df["Open"])
        | (df["High"] < df["Open"])
        | (df["Low"] > df["Close"])
        | (df["High"] < df["Close"])
    ]

    return {
        "static_outliers": static_outliers,
        "rolling_outliers": rolling_outliers,
        "inconsistent_rows": inconsistent,
    }


def get_data_quality_report(df: pd.DataFrame) -> DataQualityReport:
    """
    Generate a comprehensive data quality report.

    Args:
        df: Input DataFrame to analyze.

    Returns:
        DataQualityReport object containing all validation results.
    """
    missing_info = get_missing_values_info(df)
    type_mismatches = validate_data_types(df)
    outlier_info = get_outlier_info(df)
    outlier_cols = [f"{col}_outlier" for col in PRICE_COLUMNS]

    return DataQualityReport(
        missing_values=missing_info,
        type_mismatches=type_mismatches,
        duplicate_count=df.duplicated(subset=["Date"]).sum(),
        outlier_count=len(outlier_info["rolling_outliers"]),
        inconsistent_count=len(outlier_info["inconsistent_rows"]),
        is_clean=(
            not missing_info["has_missing"]
            and len(type_mismatches) == 0
            and len(outlier_info["inconsistent_rows"]) == 0
        ),
    )


def clean_data(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Clean the stock data DataFrame.

    Args:
        df: Input DataFrame to clean.
        remove_outliers: Whether to remove outlier rows. Defaults to True.
        inplace: Whether to modify the DataFrame in place. Defaults to False.

    Returns:
        Cleaned DataFrame with Date as index.
    """
    if not inplace:
        df = df.copy()

    df_with_flags = detect_outliers_rolling_iqr(df)
    outlier_cols = [f"{col}_outlier" for col in PRICE_COLUMNS]

    if remove_outliers:
        df = df_with_flags[~df_with_flags[outlier_cols].any(axis=1)].copy()
    else:
        df = df_with_flags.drop(columns=outlier_cols)

    df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)
    df.set_index("Date", inplace=True)

    return df


def load_and_clean_data(
    filepath: str = "data/apple_stock_data.csv",
    remove_outliers: bool = True,
) -> pd.DataFrame:
    """
    Load stock data from CSV and apply cleaning pipeline.

    Args:
        filepath: Path to the CSV file.
        remove_outliers: Whether to remove outlier rows. Defaults to True.

    Returns:
        Cleaned DataFrame with Date as index.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    return clean_data(df, remove_outliers=remove_outliers)


if __name__ == "__main__":
    df = pd.read_csv("data/apple_stock_data.csv", parse_dates=["Date"])

    report = get_data_quality_report(df)

    print("Missing Values:")
    print(report.missing_values["count"])
    print(f"\nTotal missing: {report.missing_values['total_missing']}")

    print("\nData Type Mismatches:")
    if report.type_mismatches:
        for col, (actual, expected) in report.type_mismatches.items():
            print(f" - {col}: expected {expected}, got {actual}")
    else:
        print("All columns have correct data types.")

    print(f"\nDuplicates: {report.duplicate_count}")

    print("\nOutlier Detection (Rolling IQR):")
    print(f"Outlier rows: {report.outlier_count}")

    print(f"\nInconsistent rows (OHLC violations): {report.inconsistent_count}")
