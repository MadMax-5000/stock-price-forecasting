"""
===========================================
RANDOM FOREST STOCK PRICE PREDICTION MODEL
===========================================

Description:
    Optimized Random Forest classifier for predicting stock price direction
    (up/down) using trend-based features and technical analysis.

Performance Benchmark (Test Set):
    +-----------------+------------+-------------+------------+--------------+
    | Model           | Accuracy   | Precision   | Recall     | Total Trades |
    +-----------------+------------+-------------+------------+--------------+
    | Random Forest   | 0.5605     | 0.6103      | 0.5667     | 195          |
    +-----------------+------------+-------------+------------+--------------+

Key Features:
    - 10 trend-based features (Close Ratios & Trend Strength)
    - Horizons: 2, 5, 60, 250, 1000 days
    - Walk-forward backtesting (70/30 split)
    - Optimized decision threshold: 0.51

Model Configuration:
    - Algorithm: Random Forest Classifier
    - Trees: 600
    - Max Depth: 22
    - Min Samples Split: 45
    - Min Samples Leaf: 18
    - Max Features: sqrt
    - Class Weight: balanced

Notes:
    - This model uses ONLY trend features (no raw prices or technical indicators)
    - Threshold of 0.51 was empirically optimized for best accuracy
    - Model is trained using walk-forward validation to prevent lookahead bias

Author: madmax
Date: 2025-12-16
Version: 1.0 (Benchmark)

==============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from feature_engineering import get_processed_data

# Load the fully engineered data
df = get_processed_data()
df = df.sort_index()

# ========================
# FEATURE ENGINEERING
# ========================

# Trend Features (proven to be the best predictors)
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_average = df.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    df[ratio_column] = df["Close"] / rolling_average["Close"]
    
    trend_column = f"Trend_{horizon}"
    df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]

df = df.dropna()

# ========================
# OPTIMIZED RANDOM FOREST MODEL
# ========================

model = RandomForestClassifier(
    n_estimators=600,
    min_samples_split=45,
    max_depth=22,
    min_samples_leaf=18,
    max_features='sqrt',
    class_weight='balanced',
    random_state=1,
    n_jobs=-1
)

# ========================
# PREDICTION & BACKTESTING
# ========================

def predict(train, test, predictors, model):
    """Train model and make predictions with optimized threshold"""
    model.fit(train[predictors], train["Target"])
    
    # Get probability predictions
    preds = model.predict_proba(test[predictors])[:, 1]
    
    # Apply optimal threshold (0.51)
    preds = (preds >= 0.51).astype(int)
    
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    """Walk-forward backtesting"""
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

def print_benchmark_table(predictions, model_name="Random Forest"):
    # 1. Calculate Metrics
    acc = accuracy_score(predictions["Target"], predictions["Predictions"])
    prec = precision_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    rec = recall_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    trades = int(predictions["Predictions"].sum())

    # 2. Define Table Formatting
    header_fmt = "| {:<15} | {:<10} | {:<11} | {:<10} | {:<12} |"
    row_fmt    = "| {:<15} | {:<10.4f} | {:<11.4f} | {:<10.4f} | {:<12} |"
    separator  = "+" + "-"*17 + "+" + "-"*12 + "+" + "-"*13 + "+" + "-"*12 + "+" + "-"*14 + "+"

    # 3. Print Table
    print("\nBENCHMARK REPORT")
    print(separator)
    print(header_fmt.format("Model", "Accuracy", "Precision", "Recall", "Total Trades"))
    print(separator)
    print(row_fmt.format(model_name, acc, prec, rec, trades))
    print(separator + "\n")

# ========================
# EXECUTION
# ========================

start = int(len(df) * 0.7)
predictions = backtest(df, model, new_predictors, start=start)

# Execute benchmark
print_benchmark_table(predictions, "Random Forest")