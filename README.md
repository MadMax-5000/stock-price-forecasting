# Stock Price Direction Prediction

A web-based machine learning application for predicting stock price movements. Select a stock, configure parameters, and run the prediction pipeline to forecast whether the price will go up or down.

![Stock Predictor Web App](images/main%20image%20web%20app.jpeg)

---

## Features

- **Multi-Stock Support**: Analyze 16 major stocks including AAPL, MSFT, AMZN, GOOGL, META, NFLX, NVDA, AMD, TSLA, JPM, GS, WMT, KO, NKE, XOM, and BA
- **8+ Machine Learning Models**: Random Forest, Gradient Boosting, Extra Trees, Histogram Gradient Boosting, Logistic Regression, Ridge, KNN, SVC, XGBoost, LightGBM, and CatBoost
- **Automatic Model Selection**: Pipeline selects the best model based on precision score
- **Walk-Forward Validation**: Time-series cross-validation to prevent look-ahead bias
- **Interactive Charts**: Plotly-based visualizations with prediction confidence bands
- **Configurable Parameters**: Adjust prediction horizon, training ratio, and decision threshold

---

## Project Structure

```
stock-prices/
├── pipeline.py              # Main prediction pipeline
├── visualization.py         # Plotly chart utilities
├── generate_images.py      # Report visualization generator
├── run_all_models.py       # Run multiple models
├── requirements.txt       # Python dependencies
├── data/                  # Stock data CSV files
├── models/                # Individual model implementations
├── preparing/             # Data collection and feature engineering
├── frontend/              # Next.js web application
│   ├── src/
│   │   ├── app/           # Next.js app router pages
│   │   ├── components/    # React components
│   │   └── lib/          # Utility functions
│   └── package.json      # Frontend dependencies
└── images/                # Generated visualizations
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
cd frontend
npm install
```

### 2. Run the Web Application

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### 3. Use the Application

1. Select a stock from the dropdown
2. Configure settings (date range, prediction horizon, threshold)
3. Click "Run Prediction"
4. View the results with historical chart and model comparison

---

## Configuration Parameters

| Parameter | Description | Default |
|------------|-------------|---------|
| Stock | Ticker symbol to predict | AAPL |
| Start Date | Beginning of historical data | 2015-01-01 |
| End Date | End of historical data | Today |
| Prediction Horizon | Days to forecast forward | 30 |
| Train Ratio | Training data proportion | 0.7 |
| Decision Threshold | Probability cutoff for predictions | 0.51 |

---

## Pipeline Workflow

1. **Data Collection**: Downloads historical stock data from Yahoo Finance
2. **Data Cleaning**: Removes duplicates, handles missing values, validates price relationships
3. **Feature Engineering**: Generates technical indicators (RSI, MACD, Bollinger Bands, volatility)
4. **Model Training**: Trains multiple models using walk-forward validation
5. **Best Model Selection**: Selects top performer by precision
6. **Prediction**: Generates forecast with confidence bands

---

## Technical Indicators

The pipeline generates these features for prediction:

- **Price Returns**: Daily percentage change
- **RSI (14-period)**: Relative Strength Index momentum oscillator
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: 20-period bands with position calculation
- **Volatility**: Rolling standard deviation of returns
- **Trend Features**: Rolling price ratios for 5 and 60 day horizons

---

## Models Included

| Model | Type |
|-------|------|
| RandomForest | Tree-Based Ensemble |
| GradientBoosting | Tree-Based Ensemble |
| ExtraTrees | Tree-Based Ensemble |
| HistGradientBoosting | Tree-Based Ensemble |
| LogisticRegression | Linear |
| Ridge | Linear |
| KNN | Distance-Based |
| SVC | Support Vector |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |

---

## API Endpoints

The frontend connects to these backend routes:

- `GET /api/stocks` - List available stocks
- `POST /api/predict` - Run prediction pipeline
- `GET /api/models` - Get model comparison results

---

## License

This project is for educational and research purposes.

---

*Last Updated: April 2026*