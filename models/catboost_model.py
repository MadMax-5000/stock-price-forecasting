"""
CatBoost Stock Price Prediction Model.

Optimized gradient boosting classifier for predicting stock price direction
using trend-based features and walk-forward validation.

Author: madmax
Version: 1.0
"""

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from preparing.feature_engineering import get_processed_data

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "random_seed": 1,
    "verbose": 0,
    "thread_count": -1,
}

TREND_HORIZONS = [2, 5, 60, 250, 1000]
DECISION_THRESHOLD = 0.51
BACKTEST_STEP = 250
TRAIN_SPLIT_RATIO = 0.7

# ==========================================
# LOGIC
# ==========================================

def engineer_features(df: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Generates rolling trend and ratio features."""
    predictors = []
    
    for horizon in horizons:
        rolling_avg = df.rolling(horizon).mean()
        
        ratio_col = f"Close_Ratio_{horizon}"
        df[ratio_col] = df["Close"] / rolling_avg["Close"]
        
        trend_col = f"Trend_{horizon}"
        df[trend_col] = df.shift(1).rolling(horizon).sum()["Target"]
        
        predictors.extend([ratio_col, trend_col])
        
    return df.dropna(), predictors


def predict_signal(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str], model: CatBoostClassifier) -> pd.DataFrame:
    """Trains model and returns predictions based on optimized threshold."""
    model.fit(train[predictors], train["Target"])
    
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= DECISION_THRESHOLD).astype(int)
    
    return pd.concat([
        test["Target"],
        pd.Series(preds, index=test.index, name="Predictions")
    ], axis=1)


def run_backtest(data: pd.DataFrame, model: CatBoostClassifier, predictors: list[str], start: int, step: int) -> pd.DataFrame:
    """Executes walk-forward validation."""
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : (i + step)].copy()
        
        batch_preds = predict_signal(train, test, predictors, model)
        all_predictions.append(batch_preds)
    
    return pd.concat(all_predictions)


def print_benchmark_table(predictions: pd.DataFrame, model_name: str = "CatBoost") -> None:
    """Calculates metrics and prints the benchmark report."""
    y_true = predictions["Target"]
    y_pred = predictions["Predictions"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    trades = int(y_pred.sum())

    header_fmt = "| {:<15} | {:<10} | {:<11} | {:<10} | {:<12} |"
    row_fmt    = "| {:<15} | {:<10.4f} | {:<11.4f} | {:<10.4f} | {:<12} |"
    separator  = "+" + "-"*17 + "+" + "-"*12 + "+" + "-"*13 + "+" + "-"*12 + "+" + "-"*14 + "+"

    print("\nBENCHMARK REPORT")
    print(separator)
    print(header_fmt.format("Model", "Accuracy", "Precision", "Recall", "Total Trades"))
    print(separator)
    print(row_fmt.format(model_name, acc, prec, rec, trades))
    print(separator + "\n")

# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    raw_df = get_processed_data()
    raw_df = raw_df.sort_index()

    df, features = engineer_features(raw_df, TREND_HORIZONS)

    cb_model = CatBoostClassifier(**MODEL_PARAMS)

    start_index = int(len(df) * TRAIN_SPLIT_RATIO)
    results = run_backtest(df, cb_model, features, start=start_index, step=BACKTEST_STEP)

    print_benchmark_table(results, "CatBoost")
