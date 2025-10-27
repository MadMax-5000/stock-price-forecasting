# Stock Price Forecasting

This project aims to collect and analyze Apple stock price data using Python.  
It fetches historical data from **Yahoo Finance (yfinance)**, saves it to a CSV file, and loads it into a DataFrame for further analysis and modeling.

## 📁 Project Structure
```
stock-prices/
├── data_collection.py     # Script to download and save stock data
├── data_cleaning.py       # Script for cleaning and preparing data
├── data/                  # Folder for raw and processed data
└── .gitignore             # Ignored files (e.g., CSVs)
````

## ⚙️ How It Works
1. The script fetches Apple’s stock data using the `yfinance` library.  
2. The data is saved to a local `.csv` file and loaded into a Pandas DataFrame.  
3. Cleaned and prepared data can then be used for future forecasting models.

## 🧠 Requirements
- Python 3.10 or higher  
- Install dependencies:

```bash
  pip install yfinance pandas
````

## 🧩 Usage
Run the data collection script

```bash
python data_collection.py
```
