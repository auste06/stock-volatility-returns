import yfinance as yf
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

sectors = {
    "Technology": ["AAPL", "MSFT", "NVDA", "INTC", "AMD"],
    "Healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABBV"],
    "Finance":    ["JPM", "BAC", "GS", "WFC", "C"],
    "Energy":     ["XOM", "CVX", "COP", "EOG", "SLB"],
    "Consumer":   ["AMZN", "HD", "MCD", "NKE", "SBUX"]
}

all_tickers = [t for s in sectors.values() for t in s]

print("Downloading price data... (takes ~1 min)")
data = yf.download(
    all_tickers,
    start="2019-01-01",
    end="2024-01-01",
    auto_adjust=True,
    progress=True
)

prices = data["Close"]
prices.to_csv("data/raw/prices.csv")
print(f"Saved prices: {prices.shape[0]} days, {prices.shape[1]} stocks")

sector_map = []
for sector, tickers in sectors.items():
    for t in tickers:
        sector_map.append({"ticker": t, "sector": sector})

pd.DataFrame(sector_map).to_csv("data/raw/sectors.csv", index=False)
print("Saved sector mapping")
print("Done! Check data/raw/ folder")