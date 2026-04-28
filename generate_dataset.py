"""
generate_dataset.py — Stock Market Dataset Generator

Generates a realistic synthetic stock-market dataset with
technical indicators and a binary Trend label (UP / DOWN).
Run once to create dataset.csv, then commit it to the repo.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

NUM_ROWS = 1000

# ── Simulate OHLCV price data ─────────────────────────────────
base_price = 150.0
prices = [base_price]
for _ in range(NUM_ROWS - 1):
    change = np.random.normal(0, 2)            # daily fluctuation
    prices.append(max(prices[-1] + change, 10)) # floor at $10

close = np.array(prices)
open_price = close + np.random.normal(0, 1.5, NUM_ROWS)
high = np.maximum(open_price, close) + np.abs(np.random.normal(0, 1.0, NUM_ROWS))
low = np.minimum(open_price, close) - np.abs(np.random.normal(0, 1.0, NUM_ROWS))
volume = np.random.randint(500_000, 10_000_000, NUM_ROWS).astype(float)

# ── Technical indicators ──────────────────────────────────────
def sma(series, window):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=window, min_periods=1).mean().values

def rsi(series, period=14):
    """Relative Strength Index (Wilder)."""
    delta = pd.Series(series).diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    return (100 - (100 / (1 + rs))).values

def macd(series, fast=12, slow=26):
    """MACD line (fast EMA – slow EMA)."""
    s = pd.Series(series)
    return (s.ewm(span=fast, adjust=False).mean()
            - s.ewm(span=slow, adjust=False).mean()).values

sma_10 = sma(close, 10)
sma_50 = sma(close, 50)
rsi_vals = rsi(close, 14)
macd_vals = macd(close)
prev_close = np.roll(close, 1)
prev_close[0] = close[0]  # first row has no "yesterday"

# ── Target: next-day trend ────────────────────────────────────
next_close = np.roll(close, -1)
next_close[-1] = close[-1]
trend = np.where(next_close > close, "UP", "DOWN")

# ── Build DataFrame ───────────────────────────────────────────
df = pd.DataFrame({
    "Open":       np.round(open_price, 2),
    "High":       np.round(high, 2),
    "Low":        np.round(low, 2),
    "Close":      np.round(close, 2),
    "Volume":     volume.astype(int),
    "SMA_10":     np.round(sma_10, 2),
    "SMA_50":     np.round(sma_50, 2),
    "RSI":        np.round(rsi_vals, 2),
    "MACD":       np.round(macd_vals, 4),
    "Prev_Close": np.round(prev_close, 2),
    "Trend":      trend,
})

# Inject a handful of missing values for realism (< 2 %)
for col in ["Open", "High", "Volume", "RSI", "MACD"]:
    mask = np.random.choice(NUM_ROWS, size=5, replace=False)
    df.loc[mask, col] = np.nan

df.to_csv("dataset.csv", index=False)
print(f"[DONE] dataset.csv created -- {df.shape[0]} rows, {df.shape[1]} columns")
print(f"       Trend distribution:\n{df['Trend'].value_counts().to_string()}")
