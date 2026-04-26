import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

prices = pd.read_csv("data/raw/prices.csv", index_col=0, parse_dates=True)

# Skip any non-date header rows yfinance sometimes adds
prices = prices[~prices.index.astype(str).str.contains("Price|Ticker|Date", na=False)]
prices = prices.apply(pd.to_numeric, errors="coerce")
prices = prices.dropna(axis=1, how="all")
prices.columns = [str(c).strip().upper() for c in prices.columns]

print(f"Loaded: {prices.shape[1]} stocks, {prices.shape[0]} days")

sectors = pd.read_csv("data/raw/sectors.csv")
sectors["ticker"] = sectors["ticker"].str.strip().str.upper()

log_returns = np.log(prices / prices.shift(1)).dropna()

annual_return     = log_returns.mean() * 252
annual_volatility = log_returns.std() * np.sqrt(252)
sharpe_ratio      = annual_return / annual_volatility

metrics = pd.DataFrame({
    "ticker":            annual_return.index.tolist(),
    "annual_return":     annual_return.values,
    "annual_volatility": annual_volatility.values,
    "sharpe_ratio":      sharpe_ratio.values
})

df = metrics.merge(sectors, on="ticker")

for col in ["annual_return", "annual_volatility"]:
    low  = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df[col] = df[col].clip(low, high)

df.to_csv("data/processed/stock_metrics.csv", index=False)
log_returns.to_csv("data/processed/log_returns.csv")

print(f"Final dataset: {len(df)} stocks")
print(df[["ticker","sector","annual_return","annual_volatility","sharpe_ratio"]].to_string())
print("Done — saved to data/processed/")
