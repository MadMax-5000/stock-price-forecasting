"""
Stock Price Prediction Web Application.

A Streamlit web app that allows users to:
1. Select a stock ticker
2. Run the complete pipeline (collect, clean, feature engineer)
3. Compare all models by Precision
4. Train the best model and generate predictions
5. Visualize historical data + predictions on interactive graphs

Author: madmax
Version: 2.0
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


@st.cache_data(ttl=3600)
def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data = data.reset_index()

    return data


@st.cache_data(ttl=3600)
def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean stock data."""
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Date"])
    df.set_index("Date", inplace=True)

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


@st.cache_data(ttl=3600)
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

    horizons = [2, 5, 60, 250]
    for horizon in horizons:
        rolling_avg = result_df["Close"].rolling(horizon).mean()
        result_df[f"Close_Ratio_{horizon}"] = result_df["Close"] / rolling_avg
        result_df[f"Trend_{horizon}"] = result_df["Target"].shift(1).rolling(horizon).sum()

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

    result_df = result_df.dropna()

    return result_df


def engineer_features(
    df: pd.DataFrame, horizons: list[int] = None
) -> tuple[pd.DataFrame, list[str]]:
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


def get_model_configs():
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

    models = [
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
        models.append(
            (
                "XGBoost",
                XGBClassifier,
                {
                    "n_estimators": 250,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "n_jobs": -1,
                },
            )
        )

    if HAS_LGB:
        models.append(
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
        models.append(
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

    return models


@st.cache_data(ttl=3600)
def run_all_models(
    data: pd.DataFrame, train_ratio: float, backtest_step: int, threshold: float
) -> pd.DataFrame:
    """Run all models and return comparison table."""
    model_configs = get_model_configs()

    df_sorted = data.sort_index()
    df, predictors = engineer_features(df_sorted.copy())

    if len(df) < 200:
        raise ValueError("Not enough data for modeling")

    results = []
    start_idx = int(len(df) * train_ratio)

    for model_name, model_class, params in model_configs:
        try:
            model = model_class(**params)
            preds = run_backtest(df, model, predictors, start_idx, backtest_step, threshold)

            if len(preds) > 0:
                metrics = calculate_metrics(preds)
                results.append(
                    {
                        "Model": model_name,
                        "Accuracy": metrics["accuracy"],
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1": metrics["f1"],
                        "Total_Trades": metrics["total_trades"],
                    }
                )
        except Exception:
            continue

    if not results:
        raise ValueError("No models succeeded")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Precision", ascending=False)
    results_df = results_df.reset_index(drop=True)

    return results_df


def train_best_model_and_predict(
    data: pd.DataFrame, best_model_info: dict, prediction_horizon: int, threshold: float
):
    """Train the best model and generate predictions."""
    from sklearn.ensemble import RandomForestClassifier

    df = data.sort_index()
    df, predictors = engineer_features(df.copy())

    model_name = best_model_info["model_name"]
    precision = best_model_info.get("precision", 0)

    if model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            **{
                "n_estimators": 300,
                "max_depth": 18,
                "min_samples_split": 30,
                "min_samples_leaf": 15,
                "random_state": 42,
                "n_jobs": -1,
            }
        )
    elif model_name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            **{
                "n_estimators": 200,
                "max_depth": 10,
                "learning_rate": 0.1,
                "random_state": 42,
            }
        )
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            **{
                "n_estimators": 250,
                "max_depth": 10,
                "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss",
                "verbosity": 0,
            }
        )
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            **{
                "n_estimators": 250,
                "max_depth": 10,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": -1,
            }
        )
    else:
        model = RandomForestClassifier(n_estimators=300, max_depth=18, random_state=42, n_jobs=-1)

    train_size = int(len(df) * 0.7)
    historical_avg = df["Close"].iloc[:train_size].mean()
    historical_std = df["Close"].iloc[:train_size].std()

    model.fit(df[predictors].iloc[:train_size], df["Target"].iloc[:train_size])

    hist_returns = df["Returns"].iloc[:train_size].dropna()
    if len(hist_returns) >= 20:
        historical_mean_return = hist_returns.mean()
        historical_volatility = hist_returns.std()
    else:
        historical_mean_return = 0.001
        historical_volatility = 0.02

    last_date = df.index[-1]
    start_price = df["Close"].iloc[-1]
    last_close = start_price

    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=prediction_horizon)

    future_preds = []
    upper_band = []
    lower_band = []
    last_row = df.iloc[-1:].copy()

    for i, date in enumerate(future_dates):
        try:
            probs = model.predict_proba(last_row[predictors])[:, 1]
            confidence = probs[0]
            pred_direction = 1 if confidence >= threshold else 0

            if pred_direction == 1:
                price_change = historical_mean_return
            else:
                price_change = -historical_mean_return

            cumulative_change = (i + 1) * price_change
            future_price = start_price * (1 + cumulative_change)

            mean_reversion_strength = 0.05
            future_price = (
                future_price * (1 - mean_reversion_strength)
                + historical_avg * mean_reversion_strength
            )

            future_price = max(future_price, start_price * 0.9)
            future_price = min(future_price, start_price * 1.1)

            volatility_band = historical_volatility * np.sqrt(i + 1)
            upper = future_price * (1 + volatility_band)
            lower = future_price * (1 - volatility_band)

            future_preds.append(
                {
                    "Date": date,
                    "Close": future_price,
                    "Direction": "Up" if pred_direction == 1 else "Down",
                    "Probability": confidence,
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
            last_close = start_price * (1 + (i + 1) * 0.001)
            future_preds.append(
                {
                    "Date": date,
                    "Close": last_close,
                    "Direction": "Up",
                    "Probability": 0.5,
                }
            )
            upper_band.append(start_price * (1 + (i + 1) * 0.011))
            lower_band.append(start_price * (1 + (i + 1) * 0.009))

    if future_preds:
        for j, pred in enumerate(future_preds):
            pred["Upper"] = upper_band[j] if j < len(upper_band) else pred["Close"]
            pred["Lower"] = lower_band[j] if j < len(lower_band) else pred["Close"]
    future_df = pd.DataFrame(future_preds).set_index("Date")
    historical = df[["Close"]].copy()

    return historical, future_df


def plot_chart(historical: pd.DataFrame, predictions: pd.DataFrame, ticker: str):
    """Create plotly chart."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    train_size = int(len(historical) * 0.7)
    train_data = historical.iloc[:train_size]
    test_data = historical.iloc[train_size:]

    fig.add_trace(
        go.Scatter(
            x=train_data.index.tolist(),
            y=train_data["Close"].values,
            mode="lines",
            name="Train data",
            line=dict(color="#4CAF50", width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=test_data.index.tolist(),
            y=test_data["Close"].values,
            mode="lines",
            name="Test data",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=1,
        col=1,
    )

    if predictions is not None and len(predictions) > 0:
        pred_dates = predictions.index.tolist()
        pred_close = predictions["Close"].values

        if "Upper" in predictions.columns and "Lower" in predictions.columns:
            upper = predictions["Upper"].values
            lower = predictions["Lower"].values

            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=upper,
                    mode="lines",
                    name="Upper Band",
                    line=dict(color="#4CAF50", width=1, dash="dash"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=lower,
                    mode="lines",
                    name="Lower Band",
                    line=dict(color="#F44336", width=1, dash="dash"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_close,
                mode="lines+markers",
                name="Predicted",
                line=dict(color="#FF5722", width=2),
                marker=dict(size=5, symbol="diamond"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=[test_data.index.tolist()[-1], pred_close[0]],
                mode="lines",
                line=dict(color="#FF5722", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=dict(text=f"{ticker} - Train/Test Split and Predictions", font=dict(size=20)),
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#E0E0E0")

    return fig


st.title("📈 Stock Price Predictor")

st.sidebar.header("Configuration")

selected_stock = st.sidebar.selectbox("Select Stock", STOCKS, index=0)

start_date = st.sidebar.date_input("Start Date", date.today().replace(year=date.today().year - 10))
end_date = st.sidebar.date_input("End Date", date.today())

prediction_horizon = st.sidebar.slider("Prediction Days", 7, 90, 30)

train_ratio = st.sidebar.slider("Train Ratio", 0.5, 0.9, 0.7)

threshold = st.sidebar.slider("Decision Threshold", 0.3, 0.7, 0.51)

run_button = st.sidebar.button("Run Pipeline", type="primary")

if run_button:
    with st.spinner("Downloading data..."):
        try:
            raw_df = download_stock_data(
                selected_stock, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            st.success(f"Downloaded {len(raw_df)} rows")
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.stop()

    with st.spinner("Cleaning data..."):
        clean_df = clean_stock_data(raw_df)
        st.success(f"Cleaned data: {len(clean_df)} rows")

    with st.spinner("Generating features..."):
        feature_df = generate_features(clean_df)
        st.success(f"Features generated: {len(feature_df)} rows, {len(feature_df.columns)} columns")

    with st.spinner("Running models..."):
        try:
            comparison_df = run_all_models(feature_df, train_ratio, 250, threshold)
            st.success("All models evaluated")
        except Exception as e:
            st.error(f"Error running models: {e}")
            st.stop()

    st.subheader("📊 Model Comparison")
    st.dataframe(
        comparison_df.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    best_model = comparison_df.iloc[0]
    best_model_info = {
        "model_name": best_model["Model"],
        "precision": best_model["Precision"],
    }

    st.subheader(f"🏆 Best Model: {best_model_info['model_name']}")
    st.metric("Precision", f"{best_model_info['precision']:.4f}")

    with st.spinner("Training best model and generating predictions..."):
        historical_data, future_predictions = train_best_model_and_predict(
            feature_df,
            best_model_info,
            prediction_horizon,
            threshold,
        )
        st.success("Predictions generated!")

    st.subheader(f"📈 {selected_stock} - Price Chart")

    fig = plot_chart(historical_data, future_predictions, selected_stock)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Future Predictions")
    pred_display = future_predictions.copy()
    pred_display["Close"] = pred_display["Close"].apply(lambda x: f"${x:.2f}")
    pred_display["Probability"] = pred_display["Probability"].apply(lambda x: f"{x:.2%}")
    st.dataframe(pred_display, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Last Historical Price", f"${historical_data['Close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Predicted Price (End)", f"${future_predictions['Close'].iloc[-1]:.2f}")
    with col3:
        change = (
            future_predictions["Close"].iloc[-1] / historical_data["Close"].iloc[-1] - 1
        ) * 100
        st.metric("Predicted Change", f"{change:+.2f}%")

else:
    st.info("👈 Configure your options in the sidebar and click 'Run Pipeline' to start!")

    st.markdown("""
    ### How to use:
    1. **Select a stock** from the dropdown
    2. **Set date range** for historical data
    3. **Choose prediction horizon** (how many days to predict)
    4. **Adjust train ratio** and decision threshold
    5. **Click Run Pipeline** to execute!

    The app will:
    - Download stock data from Yahoo Finance
    - Clean and preprocess the data
    - Generate technical indicators
    - Run all models and compare by Precision
    - Train the best model
    - Generate and visualize predictions
    """)

    st.markdown("---")
    st.caption("Stock data provided by Yahoo Finance | Built with Streamlit")
