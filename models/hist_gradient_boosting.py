"""
Histogram Gradient Boosting Stock Price Prediction Model.

Histogram-based gradient boosting classifier for predicting stock price direction
using trend-based features and walk-forward validation.

Author: madmax
Version: 1.0
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from models.utils import engineer_features, print_benchmark_table, run_backtest
from preparing.feature_engineering import get_processed_data

MODEL_PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 1,
}

TREND_HORIZONS: list[int] = [2, 5, 60, 250, 1000]
DECISION_THRESHOLD: float = 0.51
BACKTEST_STEP: int = 250
TRAIN_SPLIT_RATIO: float = 0.7


def create_model(**kwargs: Any) -> HistGradientBoostingClassifier:
    """
    Create a HistGradientBoostingClassifier with given or default parameters.

    Args:
        **kwargs: Model parameters to override defaults.

    Returns:
        Configured HistGradientBoostingClassifier instance.
    """
    params = MODEL_PARAMS.copy()
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)


def train_and_evaluate(
    raw_df: pd.DataFrame | None = None,
    horizons: list[int] | None = None,
    threshold: float = DECISION_THRESHOLD,
    step: int = BACKTEST_STEP,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    show_extended: bool = False,
) -> tuple[pd.DataFrame, HistGradientBoostingClassifier]:
    """
    Train Histogram Gradient Boosting model and run backtest.

    Args:
        raw_df: Input DataFrame. If None, loads from get_processed_data().
        horizons: Trend horizons for feature engineering.
        threshold: Decision threshold for predictions.
        step: Backtest window size.
        train_ratio: Train/test split ratio.
        show_extended: Whether to show extended metrics.

    Returns:
        Tuple of (predictions DataFrame, trained model).
    """
    if raw_df is None:
        raw_df = get_processed_data()

    raw_df = raw_df.sort_index()

    if horizons is None:
        horizons = TREND_HORIZONS

    df, features = engineer_features(raw_df, horizons)
    model = create_model()

    start_index = int(len(df) * train_ratio)
    results = run_backtest(
        df, model, features, start=start_index, step=step, threshold=threshold
    )

    print_benchmark_table(results, "Hist Gradient Boosting", show_extended=show_extended)

    return results, model


if __name__ == "__main__":
    train_and_evaluate()
