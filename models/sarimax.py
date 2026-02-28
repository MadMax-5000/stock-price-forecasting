"""
SARIMAX Stock Price Prediction Model.

SARIMAX time series model with exogenous variables for predicting stock price
direction using walk-forward validation.

Author: madmax
Version: 1.0
"""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import accuracy_score, precision_score, recall_score
from preparing.feature_engineering import get_processed_data

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PARAMS = {
    "order": (5, 1, 0),
    "seasonal_order": (1, 1, 1, 5),
    "trend": "c",
    "enforce_stationarity": False,
    "enforce_invertibility": False,
}

DECISION_THRESHOLD = 0.51
BACKTEST_STEP = 250
TRAIN_SPLIT_RATIO = 0.7

# ==========================================
# LOGIC
# ==========================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares exogenous features for SARIMAX."""
    df = df.copy()
    return df.dropna()


def predict_signal(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Trains SARIMAX with exogenous variables and returns predictions."""
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
        pd.Series(preds.values, index=test.index, name="Predictions")
    ], axis=1)


def run_backtest(data: pd.DataFrame, start: int, step: int) -> pd.DataFrame:
    """Executes walk-forward validation."""
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : (i + step)].copy()
        
        batch_preds = predict_signal(train, test)
        all_predictions.append(batch_preds)
    
    return pd.concat(all_predictions)


def print_benchmark_table(predictions: pd.DataFrame, model_name: str = "SARIMAX") -> None:
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

    df = engineer_features(raw_df)

    start_index = int(len(df) * TRAIN_SPLIT_RATIO)
    results = run_backtest(df, start=start_index, step=BACKTEST_STEP)

    print_benchmark_table(results, "SARIMAX")
