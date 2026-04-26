import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

prices = pd.read_csv("data/raw/prices.csv", index_col=0, parse_dates=True)
sectors = pd.read_csv("data/raw/sectors.csv")

# Fix column names — yfinance sometimes uses 'Ticker' capitalised
prices.columns = [str(c).upper() for c in prices.columns]
sectors["ticker"] = sectors["ticker"].str.upper()

print(f"Loaded: {prices.shape[1]} stocks, {prices.shape[0]} trading days")

log_returns = np.log(prices / prices.shift(1)).dropna()

annual_return     = log_returns.mean() * 252
annual_volatility = log_returns.std() * np.sqrt(252)
sharpe_ratio      = annual_return / annual_volatility

metrics = pd.DataFrame({
    "annual_return":     annual_return,
    "annual_volatility": annual_volatility,
    "sharpe_ratio":      sharpe_ratio
}).reset_index()
metrics.columns = ["ticker", "annual_return", "annual_volatility", "sharpe_ratio"]

print("Metrics tickers:", metrics["ticker"].tolist()[:5])
print("Sectors tickers:", sectors["ticker"].tolist()[:5])

df = metrics.merge(sectors, on="ticker")

print(f"After merge: {len(df)} stocks")

for col in ["annual_return", "annual_volatility"]:
    low  = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df[col] = df[col].clip(low, high)

df.to_csv("data/processed/stock_metrics.csv", index=False)
log_returns.to_csv("data/processed/log_returns.csv")

print(f"\nFinal dataset: {len(df)} stocks")
print(df[["ticker","sector","annual_return","annual_volatility","sharpe_ratio"]].to_string())
print("\nSaved to data/processed/")

