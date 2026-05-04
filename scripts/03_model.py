import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
import os

np.random.seed(42)
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/processed/stock_metrics.csv")
log_returns = pd.read_csv("data/processed/log_returns.csv",
                          index_col=0, parse_dates=True)

PALETTE = {
    "Technology": "#7F77DD",
    "Healthcare": "#1D9E75",
    "Finance":    "#378ADD",
    "Energy":     "#EF9F27",
    "Consumer":   "#D85A30"
}

m1 = smf.ols("annual_return ~ annual_volatility", data=df).fit()
m2 = smf.ols("annual_return ~ annual_volatility + C(sector)", data=df).fit()
print("Model 1 R2:", round(m1.rsquared, 3))
print("Model 2 R2:", round(m2.rsquared, 3))

# Plot 1
fig, ax = plt.subplots(figsize=(9, 6))
for sector, grp in df.groupby("sector"):
    ax.scatter(grp["annual_volatility"], grp["annual_return"],
               color=PALETTE[sector], label=sector, s=90, alpha=0.85, zorder=3)
    for _, row in grp.iterrows():
        ax.annotate(row["ticker"],
                    (row["annual_volatility"], row["annual_return"]),
                    fontsize=7, xytext=(3,3), textcoords="offset points",
                    color=PALETTE[sector])
x_line = np.linspace(df["annual_volatility"].min(), df["annual_volatility"].max(), 100)
y_line = m1.params["Intercept"] + m1.params["annual_volatility"] * x_line
ax.plot(x_line, y_line, color="#2C2C2A", linewidth=1.5, linestyle="--",
        label=f"OLS fit (b={m1.params['annual_volatility']:.2f})")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax.set_xlabel("Annualised Volatility", fontsize=12)
ax.set_ylabel("Annualised Return", fontsize=12)
ax.set_title("Volatility vs return: 25 S&P 500 stocks (2019-2024)", fontsize=13)
ax.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig("figures/01_scatter_main.png", dpi=150)
plt.close()
print("Saved 01")

# Plot 2
sector_betas = {}
for sector in df["sector"].unique():
    sub = df[df["sector"] == sector]
    if len(sub) >= 3:
        m = smf.ols("annual_return ~ annual_volatility", data=sub).fit()
        sector_betas[sector] = round(m.params["annual_volatility"], 3)
beta_s = pd.Series(sector_betas).sort_values()
fig, ax = plt.subplots(figsize=(8, 4))
colors = [PALETTE[s] for s in beta_s.index]
bars = ax.barh(beta_s.index, beta_s.values, color=colors, alpha=0.85)
ax.axvline(0, color="#888780", linewidth=0.8)
ax.set_xlabel("Volatility coefficient (beta)", fontsize=11)
ax.set_title("Risk-return relationship by sector", fontsize=13)
for bar, val in zip(bars, beta_s.values):
    offset = 0.02 if val >= 0 else -0.12
    ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig("figures/02_sector_betas.png", dpi=150)
plt.close()
print("Saved 02")

# Plot 3
order = df.groupby("sector")["sharpe_ratio"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="sector", y="sharpe_ratio", order=order,
            hue="sector", palette=PALETTE, ax=ax, linewidth=0.8, legend=False)
ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--")
ax.set_xlabel("")
ax.set_ylabel("Sharpe ratio", fontsize=11)
ax.set_title("Risk-adjusted returns by sector (2019-2024)", fontsize=13)
sns.despine()
plt.tight_layout()
plt.savefig("figures/03_sharpe_by_sector.png", dpi=150)
plt.close()
print("Saved 03")

# Plot 4
tickers_plot = [t for t in ["NVDA", "AMZN", "JNJ"] if t in log_returns.columns]
rolling_vol = log_returns[tickers_plot].rolling(30).std() * np.sqrt(252)
fig, ax = plt.subplots(figsize=(10, 4))
colors_rv = ["#7F77DD", "#D85A30", "#1D9E75"]
for ticker, color in zip(tickers_plot, colors_rv):
    ax.plot(rolling_vol.index, rolling_vol[ticker],
            label=ticker, color=color, linewidth=1.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax.set_ylabel("Annualised volatility (30-day rolling)", fontsize=11)
ax.set_title("Volatility over time: high vs low risk stocks", fontsize=13)
ax.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig("figures/04_rolling_volatility.png", dpi=150)
plt.close()
print("Saved 04")

# Plot 5
ranked = df.sort_values("annual_return", ascending=True)
fig, ax = plt.subplots(figsize=(8, 9))
colors = [PALETTE[s] for s in ranked["sector"]]
ax.barh(ranked["ticker"], ranked["annual_return"], color=colors, alpha=0.85)
ax.axvline(0, color="#888780", linewidth=0.8)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
ax.set_xlabel("Annualised Return", fontsize=11)
ax.set_title("All 25 stocks ranked by annualised return (2019-2024)", fontsize=13)
legend_elements = [Patch(facecolor=PALETTE[s], label=s) for s in PALETTE]
ax.legend(handles=legend_elements, frameon=False, fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig("figures/05_ranked_returns.png", dpi=150)
plt.close()
print("Saved 05")

# Plot 6
corr = df[["annual_return","annual_volatility","sharpe_ratio"]].corr()
corr.columns = ["Return","Volatility","Sharpe"]
corr.index = ["Return","Volatility","Sharpe"]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 12})
ax.set_title("Correlation matrix: return, volatility and Sharpe ratio", fontsize=13)
plt.tight_layout()
plt.savefig("figures/06_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved 06")

print("All 6 figures saved to figures/")
