"""
Elastic Net Stock Price Prediction Model.

Elastic Net regression classifier for predicting stock price direction using
trend-based features and walk-forward validation.

Author: madmax
Version: 1.0
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.linear_model import ElasticNet

from models.utils import engineer_features, print_benchmark_table, run_backtest
from preparing.feature_engineering import get_processed_data

MODEL_PARAMS: dict[str, Any] = {
    "alpha": 0.001,
    "l1_ratio": 0.5,
    "max_iter": 1000,
    "random_state": 1,
}

TREND_HORIZONS: list[int] = [2, 5, 60, 250, 1000]
DECISION_THRESHOLD: float = 0.51
BACKTEST_STEP: int = 250
TRAIN_SPLIT_RATIO: float = 0.7


def create_model(**kwargs: Any) -> ElasticNet:
    """
    Create an ElasticNet model with given or default parameters.

    Args:
        **kwargs: Model parameters to override defaults.

    Returns:
        Configured ElasticNet instance.
    """
    params = MODEL_PARAMS.copy()
    params.update(kwargs)
    return ElasticNet(**params)


def train_and_evaluate(
    raw_df: pd.DataFrame | None = None,
    horizons: list[int] | None = None,
    threshold: float = DECISION_THRESHOLD,
    step: int = BACKTEST_STEP,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    show_extended: bool = False,
) -> tuple[pd.DataFrame, ElasticNet]:
    """
    Train Elastic Net model and run backtest.

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
        df, model, features, start=start_index, step=step, threshold=threshold, use_proba=False
    )

    print_benchmark_table(results, "Elastic Net", show_extended=show_extended)

    return results, model


if __name__ == "__main__":
    train_and_evaluate()
