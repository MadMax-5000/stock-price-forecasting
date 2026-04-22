"""
Naive Baseline Stock Price Prediction Model.

Simple baseline that predicts the next day's direction based on the previous
day's movement. Uses walk-forward validation.

Author: madmax
Version: 1.0
"""

from __future__ import annotations

import pandas as pd

from models.utils import print_benchmark_table
from preparing.feature_engineering import get_processed_data

DECISION_THRESHOLD: float = 0.51
BACKTEST_STEP: int = 250
TRAIN_SPLIT_RATIO: float = 0.7


def predict_signal(test: pd.DataFrame) -> pd.DataFrame:
    """
    Return predictions based on previous day's close direction.

    Args:
        test: Test DataFrame with Target column.

    Returns:
        DataFrame with Target and Predictions columns.
    """
    preds = test["Target"].shift(1).fillna(0).astype(int)

    return pd.concat([
        test["Target"],
        pd.Series(preds.values, index=test.index, name="Predictions"),
    ], axis=1)


def run_backtest(data: pd.DataFrame, start: int, step: int) -> pd.DataFrame:
    """
    Execute walk-forward validation backtest.

    Args:
        data: Feature-engineered DataFrame with Target column.
        start: Starting index for testing.
        step: Number of samples per test window.

    Returns:
        DataFrame with Target and Predictions columns.
    """
    all_predictions: list[pd.DataFrame] = []

    for i in range(start, data.shape[0], step):
        test = data.iloc[i : (i + step)].copy()

        batch_preds = predict_signal(test)
        all_predictions.append(batch_preds)

    return pd.concat(all_predictions)


def train_and_evaluate(
    raw_df: pd.DataFrame | None = None,
    step: int = BACKTEST_STEP,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    show_extended: bool = False,
) -> pd.DataFrame:
    """
    Run naive baseline backtest.

    Args:
        raw_df: Input DataFrame. If None, loads from get_processed_data().
        step: Backtest window size.
        train_ratio: Train/test split ratio.
        show_extended: Whether to show extended metrics.

    Returns:
        Predictions DataFrame.
    """
    if raw_df is None:
        raw_df = get_processed_data()

    raw_df = raw_df.sort_index()
    df = raw_df.dropna()

    start_index = int(len(df) * train_ratio)
    results = run_backtest(df, start=start_index, step=step)

    print_benchmark_table(results, "Naive Baseline", show_extended=show_extended)

    return results


if __name__ == "__main__":
    train_and_evaluate()
