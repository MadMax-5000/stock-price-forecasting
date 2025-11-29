import pandas as pd
import numpy as np
from data_cleaning import clean_data_no_outliers as df

def add_features(dataframe):
    df = dataframe.copy()   

    # ========================
    #  Features
    # ========================

    # Daily returns
    df["returns"] = df["Close"].pct_change()

    # log returns
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Price ranges
    df["High_Low_Range"] = df["High"] - df["Low"]
    df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"]

    # Gap between open and previous close
    df["Gap"] = df["Open"] - df["Close"].shift(1)
    df["Gap_Pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Volume Change
    df["Volume_Pct_Change"] = df["Volume"].pct_change()

    # SMA: Simple Moving Average
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # EMA: Exponential Moving Average (Gives more weight to recent prices)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD : Moving Average Convergence Divergence
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (Measure of volatility)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # Volatility
    df["Volatility_10"] = df["returns"].rolling(window=10).std()
    df["Volatility_20"] = df["returns"].rolling(window=20).std()

    # We give the model "Yesterday's News" to predict "Tomorrow's Price"
    df['Lag_Return_1'] = df['returns'].shift(1)
    df['Lag_Return_5'] = df['returns'].shift(5)
    df['Lag_Close_1'] = df['Close'].shift(1)

    # Extracting date information
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter

    # Target variable (what we're predicting)
    df['Target'] = df['Close'].shift(-1)

    return df

final_df = add_features(df)

# Drop NaNs
# Feature Eng creates NaNs at the start/end because of rolling and shifting
final_df.dropna(inplace=True)

print("Feature Engineering Complete")
print(f"Data Shape: {final_df.shape}")

# saving to CSV
final_df.to_csv("ready_data.csv")