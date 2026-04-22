"""Shared utilities for ML model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    total_trades: int
    confusion_matrix: list[list[int]]


def engineer_features(df: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """
    Generate rolling trend and ratio features for stock prediction.

    For each horizon, creates:
        - Close_Ratio_{horizon}: Current close / rolling average
        - Trend_{horizon}: Sum of previous target values over the horizon

    Args:
        df: Input DataFrame with Close and Target columns.
        horizons: List of time horizons for rolling calculations.

    Returns:
        Tuple of (processed DataFrame with NaN rows dropped, list of predictor column names).
    """
    predictors: list[str] = []

    for horizon in horizons:
        rolling_avg = df.rolling(horizon).mean()

        ratio_col = f"Close_Ratio_{horizon}"
        df[ratio_col] = df["Close"] / rolling_avg["Close"]

        trend_col = f"Trend_{horizon}"
        df[trend_col] = df.shift(1).rolling(horizon).sum()["Target"]

        predictors.extend([ratio_col, trend_col])

    return df.dropna(), predictors


def run_backtest(
    data: pd.DataFrame,
    model,
    predictors: list[str],
    start: int,
    step: int,
    use_proba: bool = True,
    threshold: float = 0.51,
) -> pd.DataFrame:
    """
    Execute walk-forward validation backtest.

    Args:
        data: Feature-engineered DataFrame with Target column.
        model: Scikit-learn compatible model instance.
        predictors: List of feature column names.
        start: Starting index for testing.
        step: Number of samples per test window.
        use_proba: Whether to use predict_proba for predictions.
        threshold: Decision threshold for classification.

    Returns:
        DataFrame with Target and Predictions columns.
    """
    all_predictions: list[pd.DataFrame] = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : (i + step)].copy()

        model.fit(train[predictors], train["Target"])

        if use_proba:
            probs = model.predict_proba(test[predictors])[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            preds = model.predict(test[predictors])

        batch_preds = pd.concat([
            test["Target"],
            pd.Series(preds, index=test.index, name="Predictions"),
        ], axis=1)
        all_predictions.append(batch_preds)

    return pd.concat(all_predictions)


def calculate_metrics(predictions: pd.DataFrame) -> ModelMetrics:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        predictions: DataFrame with Target and Predictions columns.

    Returns:
        ModelMetrics dataclass with all evaluation metrics.
    """
    y_true = predictions["Target"]
    y_pred = predictions["Predictions"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = None
    try:
        if hasattr(y_pred, "astype"):
            roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = None

    return ModelMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        roc_auc=roc_auc,
        total_trades=int(y_pred.sum()),
        confusion_matrix=cm.tolist(),
    )


def print_benchmark_table(
    predictions: pd.DataFrame,
    model_name: str = "Model",
    show_extended: bool = False,
) -> None:
    """
    Print formatted benchmark report to console.

    Args:
        predictions: DataFrame with Target and Predictions columns.
        model_name: Name of the model for display.
        show_extended: Whether to show extended metrics (F1, ROC-AUC, confusion matrix).
    """
    metrics = calculate_metrics(predictions)

    if show_extended:
        header = (
            "| {:^15} | {:^10} | {:^11} | {:^10} | {:^10} | {:^10} | {:^12} |"
        )
        row = (
            "| {:^15} | {:^10.4f} | {:^11.4f} | {:^10.4f} | {:^10.4f} | "
            "| {:^10.4f} | {:^12} |"
        )
        separator = (
            "+" + "-" * 17 + "+" + "-" * 12 + "+" + "-" * 13 + "+"
            + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 14 + "+"
        )

        print("\nEXTENDED BENCHMARK REPORT")
        print(separator)
        print(
            header.format(
                "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Total Trades"
            )
        )
        print(separator)
        print(
            row.format(
                model_name,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.roc_auc or 0.0,
                metrics.total_trades,
            )
        )
        print(separator)
    else:
        header = "| {:<15} | {:<10} | {:<11} | {:<10} | {:<12} |"
        row = "| {:<15} | {:<10.4f} | {:<11.4f} | {:<10.4f} | {:<12} |"
        separator = (
            "+" + "-" * 17 + "+" + "-" * 12 + "+" + "-" * 13 + "+"
            + "-" * 12 + "+" + "-" * 14 + "+"
        )

        print("\nBENCHMARK REPORT")
        print(separator)
        print(
            header.format("Model", "Accuracy", "Precision", "Recall", "Total Trades")
        )
        print(separator)
        print(
            row.format(
                model_name,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.total_trades,
            )
        )
        print(separator + "\n")
