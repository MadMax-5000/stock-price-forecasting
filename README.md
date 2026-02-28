# Stock Price Direction Prediction: A Multi-Model Comparative Analysis

## Abstract

This project implements a comprehensive stock price direction prediction system for Apple Inc. (AAPL) common stock. We address the fundamental question: **Can we predict whether Apple's stock will increase or decrease in value on the following trading day?** The system employs multiple machine learning models, advanced feature engineering with technical indicators, and walk-forward validation for robust model evaluation. Our approach demonstrates the application of supervised learning methodologies to financial time series prediction, providing a framework for understanding the factors that influence daily stock price movements.

---

## 1. Introduction

### 1.1 Problem Statement

Stock price prediction remains one of the most challenging problems in financial machine learning due to the inherent stochastic nature of market movements, the presence of noise, and the efficient market hypothesis which suggests that historical prices may already be reflected in current prices. Despite these challenges, this project investigates whether systematic patterns in historical price and volume data can provide predictive signals for daily price direction.

### 1.2 Objectives

1. Collect and preprocess historical Apple stock data (2015-2025)
2. Engineer meaningful predictive features using technical analysis indicators
3. Implement and compare multiple machine learning classification models
4. Evaluate model performance using rigorous walk-forward validation
5. Identify the most influential factors driving stock price direction

### 1.3 Dataset

- **Source**: Yahoo Finance (via `yfinance` API)
- **Ticker**: AAPL (Apple Inc.)
- **Time Period**: January 1, 2015 - October 25, 2025
- **Features**: Open, High, Low, Close, Volume
- **Target Variable**: Binary classification (1 = price increases next day, 0 = price decreases)

---

## 2. Methodology

### 2.1 Data Collection

Historical stock data is fetched programmatically using the `yfinance` library, which provides access to Yahoo Finance's historical data API. The collected data includes daily OHLCV (Open, High, Low, Close, Volume) metrics.

### 2.2 Data Preprocessing

The preprocessing pipeline handles several critical aspects:

| Step | Description |
|------|-------------|
| Missing Values | Detection and handling of null entries |
| Data Types | Verification of correct numerical types |
| Duplicates | Removal of duplicate date entries |
| Outliers | Identification using IQR and rolling IQR methods |
| Indexing | Date column set as DataFrame index |

### 2.3 Feature Engineering

We generate a comprehensive set of technical indicators and derived features:

#### 2.3.1 Price-Based Features
- **Daily Returns**: Percentage change in closing price
- **Log Returns**: Natural logarithm of price ratios
- **High-Low Percentage**: Normalized daily price range
- **Gap Percentage**: Overnight price gap from previous close

#### 2.3.2 Technical Indicators
- **RSI (Relative Strength Index)**: 14-period momentum oscillator
- **MACD**: Moving Average Convergence Divergence with signal line
- **Bollinger Bands**: 20-period bands with position calculation
- **Volatility**: Rolling standard deviation of returns

#### 2.3.3 Trend Features
- Rolling price ratios for horizons: 2, 5, 60, 250 days
- Trend sums (cumulative directional movement)
- Day of week and month encoding

#### 2.3.4 Volume Features
- Volume percentage change
- Volume rolling statistics

### 2.4 Model Architecture

The project implements 20+ classification models spanning multiple paradigms:

| Category | Models |
|----------|--------|
| **Linear** | Linear Regression, Ridge, Lasso, Elastic Net, Logistic Regression |
| **Tree-Based** | Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting |
| **Boosting** | XGBoost, LightGBM, CatBoost |
| **Support Vector** | SVC, SVR, Kernel Ridge |
| **K-Nearest Neighbors** | KNN |
| **Gaussian Processes** | Gaussian Process Classifier |
| **Time Series** | ARIMA, SARIMA, SARIMAX |
| **Markov Models** | Hidden Markov Model |
| **Baseline** | Naive, Moving Average |

### 2.5 Validation Strategy

We employ **walk-forward validation** (also known as expanding window cross-validation):

1. Train on all historical data up to time *t*
2. Test on the subsequent validation window
3. Slide the window forward and repeat
4. Aggregate predictions for final performance metrics

This approach respects temporal ordering and prevents look-ahead bias, making it suitable for time series data.

### 2.6 Evaluation Metrics

- **Accuracy**: Proportion of correct predictions
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **Total Trades**: Number of predicted positive instances

---

## 3. Project Structure

```
stock-prices/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── apple_stock_data.csv          # Raw historical data
│   ├── final_csv.csv                  # Processed dataset
│   └── ready_data.csv                 # Feature-engineered dataset
├── preparing/
│   ├── __init__.py
│   ├── data_collection.py            # Data fetching from Yahoo Finance
│   ├── data_cleaning.py              # Data preprocessing pipeline
│   └── feature_engineering.py        # Technical indicator generation
└── models/
    ├── __init__.py
    ├── linear_regression.py
    ├── ridge_regression.py
    ├── lasso_regression.py
    ├── elastic_net.py
    ├── logistic_regression.py
    ├── random_forest.py
    ├── extra_trees.py
    ├── gradient_boosting.py
    ├── hist_gradient_boosting.py
    ├── xgboost_model.py
    ├── lightgbm_model.py
    ├── catboost_model.py
    ├── knn.py
    ├── svc.py
    ├── svr.py
    ├── kernel_ridge.py
    ├── gaussian_process.py
    ├── arima.py
    ├── sarima.py
    ├── sarimax.py
    ├── hidden_markov_model.py
    ├── baseline_naive.py
    └── baseline_moving_average.py
```

---

## 4. Installation

### 4.1 Prerequisites

- Python 3.10 or higher
- pip package manager

### 4.2 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance data API
- `scikit-learn` - Machine learning utilities
- `xgboost` - XGBoost classifier
- `lightgbm` - LightGBM classifier
- `catboost` - CatBoost classifier
- `statsmodels` - Time series analysis (ARIMA, etc.)
- `hmmlearn` - Hidden Markov Models
- `matplotlib` - Visualization

---

## 5. Usage

### 5.1 Data Collection

Fetch the latest Apple stock data:

```bash
cd preparing
python data_collection.py
```

### 5.2 Data Cleaning

Run the preprocessing pipeline:

```bash
python data_cleaning.py
```

### 5.3 Feature Engineering

Generate technical indicators and features:

```bash
python feature_engineering.py
```

### 5.4 Running Models

Execute any model for training and evaluation:

```bash
cd ..
python models/random_forest.py
```

Replace `random_forest.py` with any other model name to run that specific model.

---

## 6. Results Interpretation

### 6.1 Model Output

Each model produces:
1. **Benchmark Report**: Performance metrics table
2. **Predictions**: Series of buy/sell signals
3. **Feature Importance**: For tree-based models, the most influential predictors

### 6.2 Expected Performance

Stock price prediction is inherently difficult. Expect:
- **Accuracy**: 50-60% (above random 50%)
- **Precision/Recall**: Variable depending on market conditions
- **High variance**: Performance may fluctuate across different time periods

### 6.3 Key Insights

The feature importance analysis reveals which technical indicators and time horizons are most predictive:
- Short-term trends (2-5 days) often carry significant weight
- RSI and MACD provide momentum signals
- Volatility measures capture market uncertainty

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Market Efficiency**: Stock prices may already incorporate available information
2. **Transaction Costs**: Model does not account for brokerage fees
3. **External Factors**: Excludes news sentiment, macroeconomic data, and earnings reports
4. **Single Stock**: Analysis limited to Apple Inc. only
5. **Stationarity**: Assumes historical patterns continue

### 7.2 Future Enhancements

- Incorporate natural language processing for news sentiment
- Add macroeconomic indicators (interest rates, GDP)
- Implement portfolio optimization
- Extend to multiple stocks and asset classes
- Add real-time prediction capability

---

## 8. Academic References

1. Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. *Journal of Finance*, 25(2), 383-417.

2. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.

3. Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques. *Expert Systems with Applications*, 42(1), 259-268.

4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

---

## 9. License

This project is for educational and research purposes.

---

## 10. Acknowledgments

- Yahoo Finance for providing historical market data
- The open-source Python community for the analytical tools
- Various machine learning framework maintainers

---

*Last Updated: February 2026*
