# Stock Returns vs Volatility

BEE2041 Empirical Project — Risk-return analysis of S&P 500 stocks — University of Exeter, 2026

## Blog post

https://hackmd.io/@ZH22mUpcRlCw3CIQHc13kQ/ryANYmp6-g

## Research questions

- Do riskier stocks (higher volatility) deliver higher returns?
- Which sectors show the strongest risk-return relationship?
- Does volatility still predict returns after controlling for sector?

## Project structure

## Setup

## Replication — run in this order

## Data sources

Daily closing prices downloaded from Yahoo Finance via yfinance API.
Period: January 2019 to January 2024. 25 large-cap S&P 500 stocks across
5 sectors: Technology, Healthcare, Finance, Energy, Consumer.

## Key outputs

- Scatter plot: volatility vs annual return coloured by sector
- Bar chart: OLS coefficient by sector
- Box plot: Sharpe ratio distribution by sector
- Time series: 30-day rolling volatility for selected stocks