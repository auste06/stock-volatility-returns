
import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

prices = pd.read_csv("data/raw/prices.csv", index_col=0, parse_dates=True)
prices.columns = [str(c).strip().upper() for c in prices.columns]
prices = prices.apply(pd.to_numeric, errors="coerce")

print(f"Loaded: {prices.shape[1]} stocks, {prices.shape[0]} days")

sectors = pd.read_csv("data/raw/sectors.csv")
sectors["ticker"] = sectors["ticker"].str.strip().str.upper()

log_returns = np.log(prices / prices.shift(1)).dropna()

annual_return     = log_returns.mean() * 252
annual_volatility = log_returns.std() * np.sqrt(252)
sharpe_ratio      = annual_return / annual_volatility

metrics = pd.DataFrame({
    "ticker":            list(annual_return.index),
    "annual_return":     annual_return.values,
    "annual_volatility": annual_volatility.values,
    "sharpe_ratio":      sharpe_ratio.values
})

df = metrics.merge(sectors, on="ticker")
print(f"After merge: {len(df)} stocks")

for col in ["annual_return", "annual_volatility"]:
    df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

df.to_csv("data/processed/stock_metrics.csv", index=False)
log_returns.to_csv("data/processed/log_returns.csv")

print(f"Final dataset: {len(df)} stocks")
print(df[["ticker","sector","annual_return","annual_volatility","sharpe_ratio"]].to_string())
print("Saved to data/processed/")
