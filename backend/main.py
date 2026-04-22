from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yahooquery as yq
import time
import threading
from functools import lru_cache
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology"},
    {"symbol": "NFLX", "name": "Netflix Inc.", "sector": "Communication Services"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sector": "Technology"},
    {"symbol": "AMD", "name": "Advanced Micro Devices", "sector": "Technology"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Consumer Cyclical"},
    {"symbol": "CRM", "name": "Salesforce Inc.", "sector": "Technology"},
    {"symbol": "ADBE", "name": "Adobe Inc.", "sector": "Technology"},
    {"symbol": "ORCL", "name": "Oracle Corporation", "sector": "Technology"},
    {"symbol": "INTC", "name": "Intel Corporation", "sector": "Technology"},
    {"symbol": "CSCO", "name": "Cisco Systems Inc.", "sector": "Technology"},
    {"symbol": "IBM", "name": "IBM Corporation", "sector": "Technology"},
    {"symbol": "QCOM", "name": "Qualcomm Inc.", "sector": "Technology"},
]


CACHE_TTL_SECONDS = 86400
MAX_RETRIES = 5
BASE_DELAY = 2.0
MIN_REQUEST_INTERVAL = 1.0


class StockCache:
    def __init__(self, maxsize: int = 100, ttl: int = CACHE_TTL_SECONDS):
        self.cache = OrderedDict()
        self.timestamps = {}
        self.maxsize = maxsize
        self.ttl = ttl

    def _make_key(self, ticker: str, start_date: str, end_date: str) -> str:
        return f"{ticker}:{start_date}:{end_date}"

    def get(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        key = self._make_key(ticker, start_date, end_date)
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                self.cache.move_to_end(key)
                return self.cache[key]
            del self.cache[key]
            del self.timestamps[key]
        return None

    def set(self, ticker: str, start_date: str, end_date: str, data: pd.DataFrame):
        key = self._make_key(ticker, start_date, end_date)
        if key in self.cache:
            del self.cache[key]
        while len(self.cache) >= self.maxsize:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            del self.timestamps[oldest]
        self.cache[key] = data
        self.timestamps[key] = time.time()


stock_cache = StockCache()
request_lock = threading.Lock()
last_request_time = 0.0


def rate_limit():
    global last_request_time
    with request_lock:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        last_request_time = time.time()


def download_stock_data_with_retry(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    cached = stock_cache.get(ticker, start_date, end_date)
    if cached is not None:
        logger.info(f"Cache hit for {ticker}")
        return cached

    rate_limit()

    last_exception = None
    rate_limit_hit = False

    for attempt in range(MAX_RETRIES):
        try:
            yf_ticker = TICKER_MAP.get(ticker, ticker)
            ticker_obj = yq.Ticker(yf_ticker)
            data = ticker_obj.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")

            data = data.reset_index()
            data = data[["date", "open", "high", "low", "close", "volume"]]
            data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            stock_cache.set(ticker, start_date, end_date, data)
            return data

        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(
                keyword in error_msg
                for keyword in ["rate", "too many", "429", "timed out", "connect"]
            )

            if not is_rate_limit:
                raise

            rate_limit_hit = True
            last_exception = e
            delay = BASE_DELAY * (2**attempt)
            logger.warning(
                f"Rate limit hit for {ticker}, attempt {attempt + 1}/{MAX_RETRIES}, "
                f"waiting {delay:.1f}s"
            )
            time.sleep(delay)

    raise last_exception or Exception("Failed to fetch stock data")


class PredictionRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    prediction_horizon: int = 30
    train_ratio: float = 0.7
    threshold: float = 0.51


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    historical: list
    predictions: list
    model_comparison: list
    best_model: dict
    metrics: dict


@app.get("/api/stocks")
async def get_stocks():
    return STOCKS


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        data = download_stock_data_with_retry(request.ticker, request.start_date, request.end_date)
        data = clean_stock_data(data)
        data = generate_features(data)

        comparison_df = run_all_models(data, request.train_ratio, 250, request.threshold)

        best_model = comparison_df.iloc[0]
        best_model_info = {
            "model_name": best_model["Model"],
            "precision": float(best_model["Precision"]),
            "accuracy": float(best_model["Accuracy"]),
            "recall": float(best_model["Recall"]),
            "f1": float(best_model["F1"]),
        }

        historical_data, future_predictions = train_best_model_and_predict(
            data, best_model_info, request.prediction_horizon, request.threshold
        )

        historical_list = [
            {"date": str(idx), "close": float(row["Close"])}
            for idx, row in historical_data.iterrows()
        ]

        predictions_list = [
            {
                "date": str(idx),
                "close": float(row["Close"]),
                "direction": row["Direction"],
                "probability": float(row["Probability"]),
            }
            for idx, row in future_predictions.iterrows()
        ]

        model_comparison = [
            {
                "name": row["Model"],
                "precision": float(row["Precision"]),
                "accuracy": float(row["Accuracy"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1"]),
                "total_trades": int(row["Total_Trades"]),
            }
            for _, row in comparison_df.iterrows()
        ]

        metrics = {
            "last_historical_price": float(historical_data["Close"].iloc[-1]),
            "predicted_end_price": float(future_predictions["Close"].iloc[-1]),
            "predicted_change_pct": float(
                (future_predictions["Close"].iloc[-1] / historical_data["Close"].iloc[-1] - 1) * 100
            ),
        }

        return PredictionResponse(
            historical=historical_list,
            predictions=predictions_list,
            model_comparison=model_comparison,
            best_model=best_model_info,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Prediction failed for {request.ticker}: {str(e)}")
        error_detail = {
            "error": "Prediction failed",
            "message": str(e),
            "ticker": request.ticker,
        }
        if "rate" in str(e).lower() or "429" in str(e) or "too many" in str(e).lower():
            error_detail["error"] = "Rate limit exceeded"
            error_detail["message"] = (
                "Too many requests to stock data provider. Please try again later."
            )
        elif "no data" in str(e).lower():
            error_detail["error"] = "Data not found"
        raise HTTPException(status_code=503, detail=error_detail)


TICKER_MAP = {
    "META": "FB",
}


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
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


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    # ============ LAGGED RETURNS (all strictly lagged) ============
    result_df["Returns_1d"] = result_df["Close"].pct_change(1).shift(1)
    result_df["Returns_3d"] = result_df["Close"].pct_change(3).shift(1)
    result_df["Returns_5d"] = result_df["Close"].pct_change(5).shift(1)
    result_df["Returns_10d"] = result_df["Close"].pct_change(10).shift(1)
    result_df["Returns_21d"] = result_df["Close"].pct_change(21).shift(1)

    # ============ LOG RETURNS ============
    result_df["Log_Returns"] = np.log(result_df["Close"] / result_df["Close"].shift(1))

    # ============ ROLLING VOLATILITY (lagged) ============
    result_df["Volatility_5d"] = result_df["Returns_1d"].rolling(window=5).std().shift(1)
    result_df["Volatility_10d"] = result_df["Returns_1d"].rolling(window=10).std().shift(1)
    result_df["Volatility_21d"] = result_df["Returns_1d"].rolling(window=21).std().shift(1)

    # ============ MOMENTUM INDICATORS (lagged) ============
    # RSI (14) - lagged
    delta = result_df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result_df["RSI_14"] = (100 - (100 / (1 + rs))).shift(1)

    # MACD - lagged
    ema_12 = result_df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = result_df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    result_df["MACD"] = macd.shift(1)
    result_df["MACD_Signal"] = signal.shift(1)
    result_df["MACD_Hist"] = (macd - signal).shift(1)

    # Bollinger Bands - lagged
    bb_middle = result_df["Close"].rolling(window=20).mean()
    bb_std = result_df["Close"].rolling(window=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    result_df["BB_Position"] = ((result_df["Close"] - bb_lower) / (bb_upper - bb_lower)).shift(1)
    result_df["BB_Width"] = ((bb_upper - bb_lower) / bb_middle).shift(1)

    # ============ VOLUME FEATURES (lagged) ============
    result_df["Volume_Ratio_20d"] = (
        result_df["Volume"] / result_df["Volume"].rolling(20).mean()
    ).shift(1)

    # OBV (On Balance Volume) - lagged
    result_df["OBV"] = (np.sign(result_df["Close"].diff()) * result_df["Volume"]).fillna(0).cumsum()
    result_df["OBV"] = result_df["OBV"].shift(1)
    result_df["OBV_Change_5d"] = result_df["OBV"].pct_change(5).shift(1)

    # ============ PRICE FEATURES (lagged) ============
    result_df["High_Low_Pct"] = ((result_df["High"] - result_df["Low"]) / result_df["Close"]).shift(
        1
    )
    result_df["Gap_Pct"] = (
        (result_df["Open"] - result_df["Close"].shift(1)) / result_df["Close"].shift(1)
    ).shift(1)

    # Close vs moving averages (lagged)
    for window in [5, 10, 20, 50]:
        result_df[f"Close_MA_{window}_Ratio"] = (
            result_df["Close"] / result_df["Close"].rolling(window).mean()
        ).shift(1)

    # ============ LABEL: NEXT-DAY DIRECTION ============
    # Predict TOMORROW's direction using TODAY's features (no lookahead)
    result_df["Target"] = (result_df["Close"].shift(-1) > result_df["Close"]).astype(int)

    # ============ DATE FEATURES ============
    result_df["Day_of_Week"] = result_df.index.to_series().dt.dayofweek
    result_df["Month"] = result_df.index.to_series().dt.month

    # Drop intermediate columns
    drop_cols = ["Open", "High", "Low", "Volume"]
    result_df = result_df.drop([c for c in drop_cols if c in result_df.columns], axis=1)

    result_df = result_df.dropna()

    return result_df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extract predictor columns from pre-engineered features."""
    exclude_cols = ["Close", "Target", "Day_of_Week", "Month"]
    predictors = [col for col in df.columns if col not in exclude_cols]
    return df, predictors


def run_backtest(
    data: pd.DataFrame,
    model,
    predictors: list[str],
    start_idx: int,
    step: int,
    threshold: float = 0.51,
) -> pd.DataFrame:
    """Walk-forward validation with 21-day gap to prevent leakage."""
    all_preds = []
    GAP_DAYS = 21

    for i in range(start_idx, len(data), step):
        train_end = i - GAP_DAYS
        if train_end < 50:
            continue

        train = data.iloc[:train_end].copy()
        test = data.iloc[i : min(i + step, len(data))].copy()

        if len(train) < 50 or len(test) < 10:
            continue

        # Normalize features using only training window
        train_mean = train[predictors].mean()
        train_std = train[predictors].std() + 1e-8

        train_norm = (train[predictors] - train_mean) / train_std
        test_norm = (test[predictors] - train_mean) / train_std

        model.fit(train_norm, train["Target"])

        try:
            probs = model.predict_proba(test_norm)[:, 1]
            preds = (probs >= threshold).astype(int)
        except Exception:
            preds = model.predict(test_norm)

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


def calculate_metrics(predictions: pd.DataFrame, returns: pd.Series = None) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_true = predictions["Target"]
    y_pred = predictions["Predictions"]

    if len(y_true) == 0:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "total_trades": 0,
            "sharpe_ratio": 0,
            "directional_accuracy": 0,
        }

    # Directional Accuracy: % of correct direction predictions
    directional_accuracy = accuracy_score(y_true, y_pred)

    # Sharpe Ratio calculation if returns are provided
    sharpe_ratio = 0.0
    if returns is not None and len(returns) > 0:
        strategy_returns = returns * (y_pred * 2 - 1)  # Long when pred=1, Short when pred=0
        if strategy_returns.std() > 0:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "total_trades": int(y_pred.sum()),
        "sharpe_ratio": float(sharpe_ratio),
        "directional_accuracy": float(directional_accuracy),
    }


def run_all_models(
    data: pd.DataFrame, train_ratio: float, backtest_step: int, threshold: float
) -> pd.DataFrame:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    model_configs = [
        (
            "RandomForest",
            RandomForestClassifier,
            {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 30,
                "min_samples_leaf": 15,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier,
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42},
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier,
            {"n_estimators": 200, "max_depth": 10, "random_state": 42, "n_jobs": -1},
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier,
            {"max_iter": 100, "max_depth": 8, "learning_rate": 0.1, "random_state": 42},
        ),
        (
            "LogisticRegression",
            LogisticRegression,
            {"max_iter": 2000, "random_state": 42, "C": 1.0},
        ),
        ("RidgeClassifier", RidgeClassifier, {"alpha": 1.0, "random_state": 42}),
        ("KNN", KNeighborsClassifier, {"n_neighbors": 15, "weights": "distance", "n_jobs": -1}),
        ("SVC", SVC, {"kernel": "rbf", "C": 1.0, "probability": True, "random_state": 42}),
        (
            "XGBoost",
            XGBClassifier,
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        ),
        (
            "LightGBM",
            LGBMClassifier,
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": -1,
            },
        ),
        (
            "CatBoost",
            CatBoostClassifier,
            {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": 0,
            },
        ),
    ]

    df_sorted = data.sort_index()
    df, predictors = engineer_features(df_sorted.copy())

    if len(df) < 200:
        raise ValueError("Not enough data for modeling")

    results = []

    for model_name, model_class, params in model_configs:
        try:
            model = model_class(**params)
            start_idx = int(len(df) * train_ratio)

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
    from sklearn.ensemble import RandomForestClassifier

    df = data.sort_index()
    df, predictors = engineer_features(df.copy())

    model_name = best_model_info["model_name"]

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            **{
                "n_estimators": 200,
                "max_depth": 10,
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
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            }
        )
    elif model_name == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            **{
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            }
        )
    elif model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            **{
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbose": -1,
            }
        )
    else:
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

    train_size = int(len(df) * 0.7)
    historical_avg = df["Close"].iloc[:train_size].mean()

    model.fit(df[predictors].iloc[:train_size], df["Target"].iloc[:train_size])

    last_row = df.iloc[-1:][predictors]
    future_dates = pd.bdate_range(
        start=df.index[-1] + timedelta(days=1), periods=prediction_horizon
    )

    future_preds = []
    start_price = df["Close"].iloc[-1]

    for i, date in enumerate(future_dates):
        try:
            probs = model.predict_proba(last_row)[0, 1]
            direction = 1 if probs >= threshold else 0

            avg_return = df["Returns"].dropna().iloc[-20:].mean()
            if pd.isna(avg_return):
                avg_return = 0.01

            if direction == 1:
                change = abs(avg_return) * 1.2
            else:
                change = -abs(avg_return) * 0.8

            change = max(min(change, 0.05), -0.05)

            cumulative_change = (i + 1) * change
            current_close = start_price * (1 + cumulative_change)

            mean_reversion_strength = 0.05
            current_close = (
                current_close * (1 - mean_reversion_strength)
                + historical_avg * mean_reversion_strength
            )

            current_close = max(current_close, start_price * 0.9)
            current_close = min(current_close, start_price * 1.1)

            future_preds.append(
                {
                    "Date": date,
                    "Close": current_close,
                    "Direction": "Up" if direction == 1 else "Down",
                    "Probability": probs,
                }
            )
        except Exception:
            current_close = start_price * (1 + (i + 1) * 0.001)
            future_preds.append(
                {
                    "Date": date,
                    "Close": current_close,
                    "Direction": "Up",
                    "Probability": 0.5,
                }
            )

    future_df = pd.DataFrame(future_preds).set_index("Date")
    historical = df[["Close"]].copy()

    return historical, future_df


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
