"""
Model Comparison Runner - Run all models and compare performance.

This module runs all available models against the stock data
and returns a sorted comparison table based on Precision.

Author: madmax
Version: 2.0
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


AVAILABLE_MODELS: list[tuple[str, type, dict]] = [
    (
        "RandomForest",
        RandomForestClassifier,
        {
            "n_estimators": 300,
            "max_depth": 18,
            "min_samples_split": 30,
            "min_samples_leaf": 15,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    (
        "GradientBoosting",
        GradientBoostingClassifier,
        {
            "n_estimators": 200,
            "max_depth": 10,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    ),
    (
        "ExtraTrees",
        ExtraTreesClassifier,
        {
            "n_estimators": 300,
            "max_depth": 18,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    (
        "HistGradientBoosting",
        HistGradientBoostingClassifier,
        {
            "max_iter": 250,
            "max_depth": 12,
            "learning_rate": 0.1,
            "random_state": 42,
        },
    ),
    (
        "LogisticRegression",
        LogisticRegression,
        {
            "max_iter": 2000,
            "random_state": 42,
            "C": 1.0,
        },
    ),
    (
        "RidgeClassifier",
        RidgeClassifier,
        {
            "alpha": 1.0,
            "random_state": 42,
        },
    ),
    (
        "KNN",
        KNeighborsClassifier,
        {
            "n_neighbors": 15,
            "weights": "distance",
            "n_jobs": -1,
        },
    ),
    (
        "SVC",
        SVC,
        {
            "kernel": "rbf",
            "C": 1.0,
            "probability": True,
            "random_state": 42,
        },
    ),
]

if HAS_XGB:
    AVAILABLE_MODELS.append(
        (
            "XGBoost",
            XGBClassifier,
            {
                "n_estimators": 250,
                "max_depth": 10,
                "learning_rate": 0.1,
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "verbosity": 0,
                "n_jobs": -1,
            },
        )
    )

if HAS_LGB:
    AVAILABLE_MODELS.append(
        (
            "LightGBM",
            LGBMClassifier,
            {
                "n_estimators": 250,
                "max_depth": 10,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            },
        )
    )

if HAS_CAT:
    AVAILABLE_MODELS.append(
        (
            "CatBoost",
            CatBoostClassifier,
            {
                "iterations": 250,
                "depth": 10,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": 0,
            },
        )
    )


def engineer_features(df: pd.DataFrame, horizons: list[int] = None) -> tuple[pd.DataFrame, list[str]]:
    """Engineer features for modeling."""
    if horizons is None:
        horizons = [2, 5, 60, 250]

    predictors = []
    for horizon in horizons:
        rolling_avg = df.rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_avg["Close"]
        df[f"Trend_{horizon}"] = df.shift(1).rolling(horizon).sum()["Target"]
        predictors.extend([f"Close_Ratio_{horizon}", f"Trend_{horizon}"])

    return df.dropna(), predictors


def run_backtest(
    data: pd.DataFrame,
    model,
    predictors: list[str],
    start_idx: int,
    step: int,
    threshold: float = 0.51,
) -> pd.DataFrame:
    """Run walk-forward validation backtest."""
    all_preds = []

    for i in range(start_idx, len(data), step):
        train = data.iloc[:i].copy()
        test = data.iloc[i : min(i + step, len(data))].copy()

        if len(train) < 50 or len(test) < 10:
            continue

        try:
            model.fit(train[predictors], train["Target"])
            probs = model.predict_proba(test[predictors])[:, 1]
            preds = (probs >= threshold).astype(int)
        except Exception:
            preds = model.predict(test[predictors])

        batch = pd.concat([
            test[["Target"]],
            pd.Series(preds, index=test.index, name="Predictions"),
        ], axis=1)
        all_preds.append(batch)

    if not all_preds:
        return pd.DataFrame(columns=["Target", "Predictions"])

    return pd.concat(all_preds)


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    y_true = predictions["Target"]
    y_pred = predictions["Predictions"]

    if len(y_true) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "total_trades": 0}

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "total_trades": int(y_pred.sum()),
    }


def run_all_models(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    backtest_step: int = 250,
    threshold: float = 0.51,
    sort_by: str = "precision",
) -> pd.DataFrame:
    """
    Run all models and return comparison table.

    Args:
        data: Feature-engineered DataFrame with Target column.
        train_ratio: Train/test split ratio.
        backtest_step: Step size for walk-forward validation.
        threshold: Decision threshold for predictions.
        sort_by: Metric to sort by (default: 'precision').

    Returns:
        DataFrame with model comparison results sorted by the specified metric.
    """
    df_sorted = data.sort_index()
    df, predictors = engineer_features(df_sorted.copy())

    if len(df) < 200:
        raise ValueError("Not enough data for modeling")

    results = []

    print(f"Running {len(AVAILABLE_MODELS)} models...")
    print("-" * 60)

    for idx, (model_name, model_class, params) in enumerate(AVAILABLE_MODELS):
        try:
            model = model_class(**params)
            start_idx = int(len(df) * train_ratio)

            preds = run_backtest(
                df,
                model,
                predictors,
                start_idx,
                backtest_step,
                threshold,
            )

            if len(preds) > 0:
                metrics = calculate_metrics(preds)
                results.append({
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "Total_Trades": metrics["total_trades"],
                })
                print(f"[{idx + 1}/{len(AVAILABLE_MODELS)}] {model_name}: Precision={metrics['precision']:.4f}, Accuracy={metrics['accuracy']:.4f}")
            else:
                print(f"[{idx + 1}/{len(AVAILABLE_MODELS)}] {model_name}: No predictions")

        except Exception as e:
            print(f"[{idx + 1}/{len(AVAILABLE_MODELS)}] {model_name}: Error - {str(e)[:40]}")

    if not results:
        raise ValueError("No models succeeded")

    results_df = pd.DataFrame(results)

    if sort_by not in results_df.columns:
        sort_by = "Precision"

    results_df = results_df.sort_values(by=sort_by, ascending=False)
    results_df = results_df.reset_index(drop=True)

    return results_df


def get_best_model(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    backtest_step: int = 250,
    threshold: float = 0.51,
) -> tuple[str, type, dict]:
    """
    Get the best performing model based on Precision.

    Args:
        data: Feature-engineered DataFrame with Target column.
        train_ratio: Train/test split ratio.
        backtest_step: Step size for walk-forward validation.
        threshold: Decision threshold for predictions.

    Returns:
        Tuple of (model_name, model_class, params).
    """
    df_sorted = data.sort_index()
    df, predictors = engineer_features(df_sorted.copy())

    results = []

    for model_name, model_class, params in AVAILABLE_MODELS:
        try:
            model = model_class(**params)
            start_idx = int(len(df) * train_ratio)

            preds = run_backtest(
                df,
                model,
                predictors,
                start_idx,
                backtest_step,
                threshold,
            )

            if len(preds) > 0:
                metrics = calculate_metrics(preds)
                results.append({
                    "name": model_name,
                    "class": model_class,
                    "params": params,
                    "precision": metrics["precision"],
                })
        except Exception:
            continue

    if not results:
        raise ValueError("No models succeeded")

    results.sort(key=lambda x: x["precision"], reverse=True)

    return results[0]["name"], results[0]["class"], results[0]["params"]


if __name__ == "__main__":
    from pipeline import download_stock_data, clean_stock_data, generate_features, PipelineConfig

    config = PipelineConfig(ticker="AAPL")

    print("Downloading data...")
    raw_df = download_stock_data(config.ticker, config.start_date, config.end_date)

    print("Cleaning data...")
    clean_df = clean_stock_data(raw_df)

    print("Generating features...")
    feature_df = generate_features(clean_df)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)

    comparison = run_all_models(
        feature_df,
        train_ratio=config.train_ratio,
        backtest_step=config.backtest_step,
        threshold=config.decision_threshold,
        sort_by="Precision",
    )

    print("\n" + "=" * 60)
    print("SORTED RESULTS (by Precision)")
    print("=" * 60)
    print(comparison.to_string(index=False))

    best = comparison.iloc[0]
    print(f"\n>>> Best Model: {best['Model']} (Precision: {best['Precision']:.4f})")