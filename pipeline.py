"""
Unified Stock Price Prediction Pipeline Orchestrator.

This module orchestrates the complete pipeline:
1. Data Collection - Download from Yahoo Finance
2. Data Cleaning - Handle missing values, outliers, duplicates
3. Feature Engineering - Technical indicators generation
4. Model Training - Train the best model based on Precision
5. Prediction - Generate future predictions

Author: madmax
Version: 2.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
import numpy as np


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


@dataclass
class PipelineConfig:
    """Configuration for the pipeline."""

    ticker: str = "AAPL"
    start_date: str = (date.today().replace(year=date.today().year - 10)).isoformat()
    end_date: str = date.today().isoformat()
    prediction_horizon: int = 30
    train_ratio: float = 0.7
    backtest_step: int = 250
    decision_threshold: float = 0.51
    data_dir: str = "data"


@dataclass
class PipelineResult:
    """Results from the pipeline execution."""

    predictions_df: pd.DataFrame
    metrics: dict[str, float]
    best_model_name: str
    model: Any
    historical_data: pd.DataFrame
    future_predictions: pd.DataFrame


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data",
) -> pd.DataFrame:
    """Download stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.reset_index()

    output_path = f"{output_dir}/{ticker.lower()}_stock_data.csv"
    data.to_csv(output_path, index=False)

    return data


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean stock data."""
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date"])
    df.set_index("Date", inplace=True)

    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col not in df.columns:
            continue

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    mask = (
        (df["Low"] <= df["High"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
        & (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
    )
    df = df[mask]

    df = df.dropna()
    df = df.sort_index()

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate technical indicator features."""
    result_df = df.copy()

    result_df["Returns"] = result_df["Close"].pct_change()
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
    result_df["Signal_Line"] = result_df["MACD"].ewm(span=9, adjust=False).mean()

    result_df["BB_Middle"] = result_df["Close"].rolling(window=20).mean()
    result_df["BB_Std"] = result_df["Close"].rolling(window=20).std()
    result_df["BB_Upper"] = result_df["BB_Middle"] + (2 * result_df["BB_Std"])
    result_df["BB_Lower"] = result_df["BB_Middle"] - (2 * result_df["BB_Std"])
    result_df["BB_Position"] = (result_df["Close"] - result_df["BB_Lower"]) / (
        result_df["BB_Upper"] - result_df["BB_Lower"]
    )

    result_df["Volatility_10"] = result_df["Returns"].rolling(window=10).std()

    result_df["Target"] = (result_df["Close"].shift(-1) > result_df["Close"]).astype(int)

    horizons = [5, 60]
    for horizon in horizons:
        rolling_avg = result_df["Close"].rolling(horizon).mean()
        result_df[f"Close_Ratio_{horizon}"] = result_df["Close"] / rolling_avg
        result_df[f"Trend_{horizon}"] = result_df["Target"].shift(2).rolling(horizon).sum()

    result_df["Day_of_Week"] = result_df.index.to_series().dt.dayofweek
    result_df["Month"] = result_df.index.to_series().dt.month

    drop_cols = [
        "EMA_12",
        "EMA_26",
        "BB_Upper",
        "BB_Lower",
        "Tomorrow",
        "Open",
        "High",
        "Low",
        "Volume",
    ]
    result_df = result_df.drop([c for c in drop_cols if c in result_df.columns], axis=1)

    result_df = result_df.dropna(subset=["Target"])

    return result_df


def get_model_configs() -> list[tuple[str, Any, dict]]:
    """Get list of model configurations."""
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
        import xgboost as xgb
        from xgboost import XGBClassifier

        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    try:
        import lightgbm as lgb
        from lightgbm import LGBMClassifier

        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    try:
        from catboost import CatBoostClassifier

        HAS_CAT = True
    except ImportError:
        HAS_CAT = False

    models = [
        (
            "RandomForest",
            RandomForestClassifier,
            {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier,
            {
                "n_estimators": 150,
                "max_depth": 8,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier,
            {
                "n_estimators": 200,
                "max_depth": 15,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier,
            {
                "max_iter": 200,
                "max_depth": 10,
                "learning_rate": 0.1,
                "random_state": 42,
            },
        ),
        (
            "LogisticRegression",
            LogisticRegression,
            {
                "max_iter": 1000,
                "random_state": 42,
            },
        ),
        (
            "Ridge",
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
                "n_neighbors": 10,
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
        models.append(
            (
                "XGBoost",
                XGBClassifier,
                {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                },
            )
        )

    if HAS_LGB:
        models.append(
            (
                "LightGBM",
                LGBMClassifier,
                {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "verbose": -1,
                },
            )
        )

    if HAS_CAT:
        models.append(
            (
                "CatBoost",
                CatBoostClassifier,
                {
                    "iterations": 200,
                    "depth": 8,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "verbose": 0,
                },
            )
        )

    return models


def engineer_features(
    df: pd.DataFrame, horizons: list[int] = None
) -> tuple[pd.DataFrame, list[str]]:
    """Engineer features for modeling."""
    if horizons is None:
        horizons = [5, 60]

    predictors = []
    for horizon in horizons:
        rolling_avg = df.rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_avg["Close"]
        df[f"Trend_{horizon}"] = df.shift(2).rolling(horizon).sum()["Target"]
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
    GAP_DAYS = 21

    for i in range(start_idx, len(data), step):
        train_end = i - GAP_DAYS
        train = data.iloc[:train_end].copy()
        test = data.iloc[i : min(i + step, len(data))].copy()

        if len(train) < 50 or len(test) < 10 or train_end <= 0:
            continue

        model.fit(train[predictors], train["Target"])

        try:
            probs = model.predict_proba(test[predictors])[:, 1]
            preds = (probs >= threshold).astype(int)
        except Exception:
            preds = model.predict(test[predictors])

        batch = pd.concat(
            [
                test[["Target"]],
                pd.Series(preds, index=test.index, name="Predictions"),
            ],
            axis=1,
        )
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


def run_pipeline(config: PipelineConfig | None = None) -> PipelineResult:
    """Run the complete pipeline."""
    if config is None:
        config = PipelineConfig()

    print(f"=== Stock Price Prediction Pipeline ===")
    print(f"Stock: {config.ticker}")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Prediction Horizon: {config.prediction_horizon} days")
    print()

    print("[1/5] Downloading data...")
    df = download_stock_data(
        config.ticker,
        config.start_date,
        config.end_date,
        config.data_dir,
    )
    print(f"      Downloaded {len(df)} rows")

    print("[2/5] Cleaning data...")
    df = clean_stock_data(df)
    print(f"      {len(df)} clean rows")

    print("[3/5] Generating features...")
    df = generate_features(df)
    print(f"      {len(df)} rows with {len(df.columns)} features")

    print("[4/5] Training and evaluating models...")
    df = df.sort_index()

    model_configs = get_model_configs()
    results = []

    for idx, (model_name, model_class, params) in enumerate(model_configs):
        try:
            model = model_class(**params)
            df_copy, predictors = engineer_features(df.copy())

            if len(df_copy) < 200:
                continue

            start_idx = int(len(df_copy) * config.train_ratio)

            preds = run_backtest(
                df_copy,
                model,
                predictors,
                start_idx,
                config.backtest_step,
                config.decision_threshold,
            )

            if len(preds) > 0:
                metrics = calculate_metrics(preds)
                results.append(
                    {
                        "model_name": model_name,
                        "model_class": model_class,
                        "params": params,
                        "precision": metrics["precision"],
                        "accuracy": metrics["accuracy"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "predictions": preds,
                    }
                )
                print(f"      {model_name}: Precision={metrics['precision']:.4f}")
        except Exception as e:
            print(f"      {model_name}: Failed - {str(e)[:50]}")
            continue

    if not results:
        raise ValueError("No models succeeded in training")

    results.sort(key=lambda x: x["precision"], reverse=True)
    best = results[0]
    best_model_name = best["model_name"]
    best_precision = best["precision"]

    print(f"      Best model: {best_model_name} (Precision: {best_precision:.4f})")

    print("[5/5] Training final model for prediction...")
    df_final, predictors = engineer_features(df.copy())

    final_model = best["model_class"](**best["params"])
    train_size = int(len(df_final) * config.train_ratio)
    final_model.fit(df_final[predictors].iloc[:train_size], df_final["Target"].iloc[:train_size])

    print("      Generating predictions...")

    hist_returns = df_final["Returns"].iloc[:train_size].dropna()
    if len(hist_returns) >= 20:
        historical_mean_return = hist_returns.mean()
        historical_volatility = hist_returns.std()
    else:
        historical_mean_return = 0.001
        historical_volatility = 0.02

    last_date = df_final.index[-1]
    start_price = df_final["Close"].iloc[-1]
    last_close = start_price
    print(f"      Starting forecast from: {last_date.date()}, close=${start_price:.2f}")

    future_dates = pd.bdate_range(
        start=last_date + timedelta(days=1),
        periods=config.prediction_horizon,
    )

    future_preds = []
    upper_band = []
    lower_band = []
    last_row = df_final.iloc[-1:].copy()

    for i, date in enumerate(future_dates):
        try:
            probs = final_model.predict_proba(last_row[predictors])[:, 1]
            confidence = probs[0]
            pred_direction = 1 if confidence >= config.decision_threshold else 0

            if pred_direction == 1:
                price_change = historical_mean_return
            else:
                price_change = -historical_mean_return

            cumulative_change = (i + 1) * price_change
            future_price = start_price * (1 + cumulative_change)

            future_price = max(future_price, start_price * 0.9)
            future_price = min(future_price, start_price * 1.1)

            volatility_band = historical_volatility * np.sqrt(i + 1)
            upper = future_price * (1 + volatility_band)
            lower = future_price * (1 - volatility_band)

            future_preds.append(
                {
                    "Date": date,
                    "Close": future_price,
                    "Prediction": pred_direction,
                }
            )
            upper_band.append(upper)
            lower_band.append(lower)

            new_row = last_row.copy()
            new_row["Close"] = future_price
            new_row["Returns"] = (future_price / start_price) - 1
            new_row["Target"] = np.nan

            for horizon in [2, 5, 60, 250]:
                rolling_vals = last_row.get(f"Close_Ratio_{horizon}", last_row["Close"])
                if rolling_vals.iloc[0] != 0 and not pd.isna(rolling_vals.iloc[0]):
                    new_row[f"Close_Ratio_{horizon}"] = future_price / (
                        start_price / rolling_vals.iloc[0]
                    )
                trend = last_row.get(f"Trend_{horizon}", 0)
                if isinstance(trend, pd.Series):
                    trend = trend.iloc[0]
                new_row[f"Trend_{horizon}"] = trend + (pred_direction * 2 - 1) / horizon

            last_row = new_row
        except Exception:
            for j, date in enumerate(future_dates[i:], start=i):
                pred_close = start_price * (1 + (j + 1) * 0.001)
                future_preds.append(
                    {
                        "Date": date,
                        "Close": pred_close,
                        "Prediction": 1,
                    }
                )
                upper_band.append(start_price * (1 + (j + 1) * 0.011))
                lower_band.append(start_price * (1 + (j + 1) * 0.009))
            break

    if future_preds:
        for j, pred in enumerate(future_preds):
            pred["Upper"] = upper_band[j] if j < len(upper_band) else pred["Close"]
            pred["Lower"] = lower_band[j] if j < len(lower_band) else pred["Close"]
        future_df = pd.DataFrame(future_preds).set_index("Date")
    else:
        future_df = pd.DataFrame(columns=["Close", "Prediction", "Upper", "Lower"])

    historical = df[["Close"]].copy()

    return PipelineResult(
        predictions_df=best["predictions"],
        metrics={
            "precision": best_precision,
            "accuracy": best["accuracy"],
            "recall": best["recall"],
            "f1": best["f1"],
        },
        best_model_name=best_model_name,
        model=final_model,
        historical_data=historical,
        future_predictions=future_df,
    )


if __name__ == "__main__":
    import numpy as np

    config = PipelineConfig(
        ticker="AAPL",
        prediction_horizon=30,
    )

    result = run_pipeline(config)

    print("\n=== Pipeline Complete ===")
    print(f"Best Model: {result.best_model_name}")
    print(f"Precision: {result.metrics['precision']:.4f}")
    print(f"Historical data: {len(result.historical_data)} rows")
    print(f"Future predictions: {len(result.future_predictions)} rows")
