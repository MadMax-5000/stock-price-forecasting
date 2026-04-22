"""
Hidden Markov Model Stock Price Prediction Model.

Gaussian HMM classifier for predicting stock price direction using
trend-based features and walk-forward validation.

Author: madmax
Version: 1.0
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from hmmlearn.hmm import GaussianHMM

from models.utils import engineer_features, print_benchmark_table
from preparing.feature_engineering import get_processed_data

MODEL_PARAMS: dict[str, Any] = {
    "n_components": 3,
    "covariance_type": "full",
    "n_iter": 100,
    "random_state": 1,
}

TREND_HORIZONS: list[int] = [2, 5, 60, 250, 1000]
DECISION_THRESHOLD: float = 0.51
BACKTEST_STEP: int = 250
TRAIN_SPLIT_RATIO: float = 0.7


def create_model(**kwargs: Any) -> GaussianHMM:
    """
    Create a GaussianHMM with given or default parameters.

    Args:
        **kwargs: Model parameters to override defaults.

    Returns:
        Configured GaussianHMM instance.
    """
    params = MODEL_PARAMS.copy()
    params.update(kwargs)
    return GaussianHMM(**params)


def run_backtest_hmm(
    data: pd.DataFrame,
    model: GaussianHMM,
    predictors: list[str],
    start: int,
    step: int,
    threshold: float = DECISION_THRESHOLD,
) -> pd.DataFrame:
    """
    Execute walk-forward validation for Hidden Markov Model.

    Args:
        data: Feature-engineered DataFrame with Target column.
        model: GaussianHMM model instance.
        predictors: List of feature column names.
        start: Starting index for testing.
        step: Number of samples per test window.
        threshold: Decision threshold for predictions.

    Returns:
        DataFrame with Target and Predictions columns.
    """
    all_predictions: list[pd.DataFrame] = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : (i + step)].copy()

        model.fit(train[predictors].values)

        probs = model.predict_proba(test[predictors].values)[:, -1]
        preds = (probs >= threshold).astype(int)

        batch_preds = pd.concat([
            test["Target"],
            pd.Series(preds, index=test.index, name="Predictions"),
        ], axis=1)
        all_predictions.append(batch_preds)

    return pd.concat(all_predictions)


def train_and_evaluate(
    raw_df: pd.DataFrame | None = None,
    horizons: list[int] | None = None,
    threshold: float = DECISION_THRESHOLD,
    step: int = BACKTEST_STEP,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    show_extended: bool = False,
) -> tuple[pd.DataFrame, GaussianHMM]:
    """
    Train Hidden Markov Model and run backtest.

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
    results = run_backtest_hmm(
        df, model, features, start=start_index, step=step, threshold=threshold
    )

    print_benchmark_table(results, "HMM", show_extended=show_extended)

    return results, model


if __name__ == "__main__":
    train_and_evaluate()
