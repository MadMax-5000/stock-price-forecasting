"""
SARIMAX Stock Price Prediction Model.

SARIMAX time series model with exogenous variables for predicting stock price
direction using walk-forward validation.

Author: madmax
Version: 1.0
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.utils import print_benchmark_table
from preparing.feature_engineering import get_processed_data

MODEL_PARAMS: dict[str, tuple[int, int, int] | tuple[int, int, int, int] | str | bool] = {
    "order": (5, 1, 0),
    "seasonal_order": (1, 1, 1, 5),
    "trend": "c",
    "enforce_stationarity": False,
    "enforce_invertibility": False,
}

DECISION_THRESHOLD: float = 0.51
BACKTEST_STEP: int = 250
TRAIN_SPLIT_RATIO: float = 0.7


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare exogenous features for SARIMAX.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with NaN rows dropped.
    """
    result_df = df.copy()
    return result_df.dropna()


def predict_signal(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Train SARIMAX with exogenous variables and return predictions.

    Args:
        train: Training DataFrame.
        test: Test DataFrame.

    Returns:
        DataFrame with Target and Predictions columns.
    """
    try:
        exog_train = train.drop(columns=["Close", "Target"], errors="ignore")
        exog_test = test.drop(columns=["Close", "Target"], errors="ignore")

        model = SARIMAX(train["Close"], exog=exog_train, **MODEL_PARAMS)
        fitted = model.fit(disp=False)

        forecast = fitted.forecast(steps=len(test), exog=exog_test)
        preds = (forecast >= DECISION_THRESHOLD).astype(int)

    except Exception:
        preds = pd.Series(0, index=test.index)

    return pd.concat([
        test["Target"],
        pd.Series(preds.values, index=test.index, name="Predictions"),
    ], axis=1)


def run_backtest(data: pd.DataFrame, start: int, step: int) -> pd.DataFrame:
    """
    Execute walk-forward validation backtest.

    Args:
        data: Feature-engineered DataFrame with Target and Close columns.
        start: Starting index for testing.
        step: Number of samples per test window.

    Returns:
        DataFrame with Target and Predictions columns.
    """
    all_predictions: list[pd.DataFrame] = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : (i + step)].copy()

        batch_preds = predict_signal(train, test)
        all_predictions.append(batch_preds)

    return pd.concat(all_predictions)


def train_and_evaluate(
    raw_df: pd.DataFrame | None = None,
    step: int = BACKTEST_STEP,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    show_extended: bool = False,
) -> pd.DataFrame:
    """
    Train SARIMAX model and run backtest.

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
    df = engineer_features(raw_df)

    start_index = int(len(df) * train_ratio)
    results = run_backtest(df, start=start_index, step=step)

    print_benchmark_table(results, "SARIMAX", show_extended=show_extended)

    return results


if __name__ == "__main__":
    train_and_evaluate()
