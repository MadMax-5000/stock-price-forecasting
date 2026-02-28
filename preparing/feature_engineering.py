import pandas as pd
import numpy as np
from preparing.data_cleaning import clean_data_no_outliers as source_df

def add_features(dataframe):
    df = dataframe.copy()   

    # ========================
    #  1. Set Date Index
    # ========================
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    if not isinstance(df.index, pd.DatetimeIndex):
         print("Warning: Index is not Datetime. Date features may fail.")

    # ========================
    #  2. Base Features
    # ========================
    
    # Calculate the Target PRELIMINARILY to use it for "Trend" features
    # (We need to know if it went up yesterday to calculate the trend for today)
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    # Daily returns & Log returns
    df["returns"] = df["Close"].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Price ranges (Normalized)
    df["High_Low_Pct"] = (df["High"] - df["Low"]) / df["Close"]
    df["Gap_Pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # Volume Change
    df["Volume_Pct_Change"] = df["Volume"].pct_change()

    # RSI Calculation
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (Normalized width)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    # Using Position in Band rather than raw value makes it stationary
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Volatility
    df["Volatility_10"] = df["returns"].rolling(window=10).std()
    
    # ========================
    #  3. Rolling Horizon Features (Very Important)
    # ========================
    # These look at the last 2, 5, 60 days to see general trend
    horizons = [2, 5, 60, 250]
    
    for horizon in horizons:
        rolling_average = df['Close'].rolling(horizon).mean()
        
        # Ratio: Is price above or below the long term average?
        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_average
        
        # Trend: Sum of "Targets" (Ups) in the previous X days
        # We shift(1) to ensure we don't peek at today's result
        trend_column = f"Trend_{horizon}"
        df[trend_column] = df["Target"].shift(1).rolling(horizon).sum()

    # ========================
    #  4. Cleanup
    # ========================
    
    # Extract Date Info
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month

    # Remove columns that are raw prices (non-stationary) or intermediate calc
    # We keep 'Close' just for reference in backtesting, but won't use it as a predictor
    cols_to_drop = ['EMA_12', 'EMA_26', 'BB_Upper', 'BB_Lower', 'Tomorrow']
    df.drop([c for c in cols_to_drop if c in df.columns], axis=1, inplace=True)
    
    # Drop NaNs created by rolling windows (this will drop the first 250 rows approx)
    df.dropna(inplace=True)

    return df

def get_processed_data():
    return add_features(source_df)

if __name__ == "__main__":
    final_df = get_processed_data()
    print("Feature Engineering Complete")
    print(f"Data Shape: {final_df.shape}")
    print(final_df.head())
    final_df.to_csv("ready_data.csv")
