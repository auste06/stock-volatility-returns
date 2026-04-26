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

# Download one ticker at a time — avoids MultiIndex issues completely
frames = []
for ticker in all_tickers:
    try:
        df = yf.download(ticker, start="2019-01-01", end="2024-01-01",
                        auto_adjust=True, progress=False)
        frames.append(df[["Close"]].rename(columns={"Close": ticker}))
        print(f"  {ticker}: {len(df)} rows")
    except Exception as e:
        print(f"  {ticker}: FAILED — {e}")

prices = pd.concat(frames, axis=1)
prices.index.name = "Date"
prices.to_csv("data/raw/prices.csv")
print(f"\nSaved: {prices.shape[1]} stocks, {prices.shape[0]} days")
print("Sample:\n", prices.head(2))

# Save sector map
sector_map = [{"ticker": t, "sector": s}
              for s, tickers in sectors.items() for t in tickers]
pd.DataFrame(sector_map).to_csv("data/raw/sectors.csv", index=False)
print("Saved sectors.csv")
