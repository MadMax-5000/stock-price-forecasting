# this file is for cleaning data from the CSV file data/apple_stock_data.csv
# =========================================
# Missing or Null Values == Correct Data Types == Duplicates == Sort by Date == Create Additional Features 
# Outliers == Set Date as Index

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("apple_stock_data.csv", parse_dates=["Date"])

# ==============================
# checking missing values or Nan
# ==============================

print(df.isna().sum()) # gives count of missing values per column
print(df.isna().any()) # gives the boolean value of column
print(df.isna().sum().sum()) # gives the total number of missing values

# ==============================
# correcting data types
# ==============================


# print(df.dtypes)
pd.from_dummies
expected_types = {
    "Date" : "datetime64[ns]",
    "Open" : "float64",
    "High" : "float64",
    "Low" : "float64",
    "Close" : "float64",
    "Volume" : "int64"
}


mismatches = {col : (df[col].dtype, expected)
              for col, expected in expected_types.items()
                if str(df[col].dtype) != expected}

if not mismatches:
     print("All column data types match the exact types ")
else: 
     print("Mismated data found at ")
     for col, (actual, expected) in mismatches.items():
        print(f" - {col} : expected {expected}, got {actual}")



# =========================
# detecting outliers
# =========================

# using the IQR (Interquartile Range) method
price_cols = ["Open", "High", "Low", "Close"]
Q1 = df[price_cols].quantile(0.25)
Q3 = df[price_cols].quantile(0.75)
IQR = Q3 - Q1

outliers = (df[price_cols] < (Q1 - 1.5*IQR)) | (df[price_cols] > (Q3 + 1.5*IQR))
outliers_row = df[outliers.any(axis=1)]

# using the rolling IQR
# We use the rolling IQR method to detect anomalies to recent prices, ignoring old low prices by focusing
# on a rolling window (e.g 1 year or 252 trading days)

window = 252 # ~1 trading year
for col in ["Open", "High", "Low", "Close"]:
    rolling_Q1 = df[col].rolling(window).quantile(0.25)
    rolling_Q3 = df[col].rolling(window).quantile(0.75)
    rolling_IQR = rolling_Q3 - rolling_Q1
    outliers_col = (df[col] < rolling_Q1 - 1.5*rolling_IQR) | (df[col] > rolling_Q3 + 1.5*rolling_IQR)
    df[col + '_outlier'] = outliers_col

outlier_rows = df[df[['Open_outlier','High_outlier','Low_outlier','Close_outlier']].any(axis=1)]
# checking missing values
print(df.isnull().sum())

# checking duplicates
print(df.duplicated(subset=["Date"]).sum())

# checking inconsistent rows 
inconsistent = df[(df['Low'] > df['Open']) | (df['High'] < df['Open']) |
                  (df['Low'] > df['Close']) | (df['High'] < df['Close'])]
print(inconsistent)

 
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.scatter(
    df.loc[df['Close_outlier'], 'Date'],
    df.loc[df['Close_outlier'], 'Close'],
    color='red',
    label='Outliers'
)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Apple Stock with Rolling-IQR Outliers Highlighted')
plt.legend()
plt.show()


# plotting the IQR outliers and the apple stock graph over the years
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Close Price')
plt.scatter(outliers_col['Date'], outliers_col['Close'], color='red', label='Outliers')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Prices with Outliers Highlighted')
plt.legend()
plt.show()
