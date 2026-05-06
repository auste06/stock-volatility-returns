# Stock Returns vs Volatility

BEE2041 Empirical Project — Risk-return analysis of S&P 500 stocks — University of Exeter, 2026

## Blog post

https://hackmd.io/@ZH22mUpcRlCw3CIQHc13kQ/H1iQTM80Zx

## Research questions

- Do riskier stocks deliver higher returns?
- Which sectors show the strongest risk-return relationship?
- Does volatility predict returns after controlling for sector?

## Project summary
This project analyses 25 large-cap S&P 500 stocks across five sectors: Technology, Healthcare, Finance, Energy, and Consumer. It uses daily price data from Yahoo Finance to calculate annual returns, volatility, and Sharpe ratios, then estimates sector-level relationships using regression analysis.

## Main finding
The project finds that the relationship between volatility and returns is not uniform across sectors. Technology shows the clearest positive risk-return pattern, while some sectors show weak or even negative relationships once sector composition is taken into account.

## Data sources
The dataset consists of daily closing prices downloaded from Yahoo Finance using the `yfinance` Python package. The sample covers January 2019 to January 2024 and includes 25 large-cap S&P 500 stocks from five sectors:
- Technology
- Healthcare
- Finance
- Energy
- Consumer

## Key outputs
The final blog post includes six pieces of output, including:
- Volatility vs return scatter plot by sector
- Sector-level coefficient comparison
- Sharpe ratio distribution by sector
- Rolling volatility time-series for selected stocks
- Additional regression and sector-comparison outputs shown in the final blog post

## Project structure

data/raw - original downloaded data
data/processed - cleaned analysis-ready data
scripts/01_scrape.py - downloads price data
scripts/02_clean.py - computes returns and volatility
scripts/03_model.py - OLS regressions and 6 figures
notebooks/blog.ipynb - main blog post
figures - all 6 saved plots

## Replication
To replicate the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/stock-volatility-returns.git
   cd stock-volatility-returns
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts in order:
   ```bash
   python scripts/01_download_data.py
   python scripts/02_clean_data.py
   python scripts/03_analysis.py
   python scripts/04_make_figures.py
   ```

4. Open the final notebook:
   - `notebooks/blog.ipynb`

## Software used
- Python
- Jupyter Notebook / VS Code
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- yfinance
- Git and GitHub
- HackMD for the final published blog format