import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import os

np.random.seed(42)
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/processed/stock_metrics.csv")

PALETTE = {
    "Technology": "#7F77DD",
    "Healthcare": "#1D9E75",
    "Finance":    "#378ADD",
    "Energy":     "#EF9F27",
    "Consumer":   "#D85A30"
}

# --- MODEL 1: basic OLS ---
m1 = smf.ols("annual_return ~ annual_volatility", data=df).fit()
print("=== Model 1: Basic OLS ===")
print(m1.summary())

# --- MODEL 2: OLS with sector dummies ---
m2 = smf.ols("annual_return ~ annual_volatility + C(sector)", data=df).fit()
print("\n=== Model 2: With sector controls ===")
print(m2.summary())

# --- SECTOR REGRESSIONS ---
print("\n=== Sector-level betas ===")
sector_betas = {}
for sector in df["sector"].unique():
    sub = df[df["sector"] == sector]
    m = smf.ols("annual_return ~ annual_volatility", data=sub).fit()
    sector_betas[sector] = {
        "beta": round(m.params["annual_volatility"], 3),
        "p":    round(m.pvalues["annual_volatility"], 3),
        "r2":   round(m.rsquared, 3)
    }
    print(f"  {sector}: beta={sector_betas[sector]['beta']}, p={sector_betas[sector]['p']}")

# =====================
# PLOT 1: Main scatter
# =====================
fig, ax = plt.subplots(figsize=(9, 6))
for sector, grp in df.groupby("sector"):
    ax.scatter(grp["annual_volatility"], grp["annual_return"],
               color=PALETTE[sector], label=sector, s=90, alpha=0.85, zorder=3)
    for _, row in grp.iterrows():
        ax.annotate(row["ticker"],
                    (row["annual_volatility"], row["annual_return"]),
                    fontsize=7, ha="left", va="bottom",
                    color=PALETTE[sector], xytext=(3,3), textcoords="offset points")

x_line = np.linspace(df["annual_volatility"].min(), df["annual_volatility"].max(), 100)
y_line = m1.params["Intercept"] + m1.params["annual_volatility"] * x_line
ax.plot(x_line, y_line, color="#2C2C2A", linewidth=1.5, linestyle="--",
        label=f"OLS fit (β={m1.params['annual_volatility']:.2f}, p={m1.pvalues['annual_volatility']:.3f})")

ax.set_xlabel("Annualised Volatility", fontsize=12)
ax.set_ylabel("Annualised Return", fontsize=12)
ax.set_title("Do riskier stocks earn more?", fontsize=13, fontweight="normal")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.legend(frameon=False, fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig("figures/01_scatter_main.png", dpi=150)
plt.close()
print("Saved figures/01_scatter_main.png")

# =====================
# PLOT 2: Sector betas
# =====================
beta_df = pd.DataFrame(sector_betas).T.reset_index()
beta_df.columns = ["sector","beta","p","r2"]
beta_df = beta_df.sort_values("beta", ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
colors = [PALETTE[s] for s in beta_df["sector"]]
bars = ax.barh(beta_df["sector"], beta_df["beta"], color=colors, alpha=0.85)
ax.axvline(0, color="#888780", linewidth=0.8)
ax.set_xlabel("Volatility coefficient (β)", fontsize=11)
ax.set_title("Risk-return relationship by sector", fontsize=13, fontweight="normal")
for bar, (_, row) in zip(bars, beta_df.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"β={row['beta']:.2f}", va="center", fontsize=9)
sns.despine()
plt.tight_layout()
plt.savefig("figures/02_sector_betas.png", dpi=150)
plt.close()
print("Saved figures/02_sector_betas.png")

# =====================
# PLOT 3: Sharpe by sector
# =====================
fig, ax = plt.subplots(figsize=(8, 5))
order = df.groupby("sector")["sharpe_ratio"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="sector", y="sharpe_ratio", order=order,
            palette=PALETTE, ax=ax, linewidth=0.8, fliersize=4)
ax.axhline(0, color="#888780", linewidth=0.8, linestyle="--")
ax.set_xlabel("")
ax.set_ylabel("Sharpe ratio", fontsize=11)
ax.set_title("Risk-adjusted returns by sector", fontsize=13, fontweight="normal")
sns.despine()
plt.tight_layout()
plt.savefig("figures/03_sharpe_by_sector.png", dpi=150)
plt.close()
print("Saved figures/03_sharpe_by_sector.png")

# =====================
# PLOT 4: Rolling volatility
# =====================
log_returns = pd.read_csv("data/processed/log_returns.csv", index_col=0, parse_dates=True)
tickers_to_plot = ["NVDA", "JNJ", "AMZN"]
rolling_vol = log_returns[tickers_to_plot].rolling(30).std() * np.sqrt(252)

fig, ax = plt.subplots(figsize=(10, 4))
colors_rv = ["#7F77DD", "#1D9E75", "#D85A30"]
for ticker, color in zip(tickers_to_plot, colors_rv):
    ax.plot(rolling_vol.index, rolling_vol[ticker], label=ticker, color=color, linewidth=1.2)
ax.set_ylabel("Annualised volatility (30-day rolling)", fontsize=11)
ax.set_title("Volatility over time: high vs low risk stocks", fontsize=13, fontweight="normal")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(frameon=False)
sns.despine()
plt.tight_layout()
plt.savefig("figures/04_rolling_volatility.png", dpi=150)
plt.close()
print("Saved figures/04_rolling_volatility.png")

print("\nAll figures saved to figures/")

