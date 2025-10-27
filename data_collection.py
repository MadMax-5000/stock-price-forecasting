# =================================================
# the main goal of this project is to answer this question
# Cpicklean I predict whether Apple’s stock will go up or down tomorrow and explain the factors driving it?

import yfinance as yf
import pandas as pd
# Download historical data for the past 10 years (2025 - 2015)
data = yf.download("AAPL", start = "2015-01-01", end = "2025-10-25", auto_adjust=True)

# flatten columns if needed 

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Keep main columns only 
data = data[["Open", "High", "Low", "Close", "Volume"]]
data = data.reset_index()

#saving data to csv file 
data.to_csv("apple_stock_data.csv", index=False)

# load back into dataframe
df = pd.read_csv("apple_stock_data.csv", parse_dates=["Date"])

print(df.head())
