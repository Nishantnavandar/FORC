import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# ----------------------
# Streamlit UI
# ----------------------
st.title("📈 Stock Forecasting + Technical & Fundamental Analysis (ARIMA)")

ticker = st.text_input("Enter Stock Ticker (Example: RELIANCE.NS, TCS.NS)", "RELIANCE.NS")

forecast_period = st.selectbox(
    "Select Forecast Period",
    ["6 Months", "1 Year", "2 Years"]
)

period_map = {
    "6 Months": 6,
    "1 Year": 12,
    "2 Years": 24
}
forecast_steps = period_map[forecast_period]

# ----------------------
# Fetch Stock Data
# ----------------------
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="max", interval="1mo")
        if df.empty:
            return None
        df = df.dropna()
        return df
    except:
        return None

df = get_stock_data(ticker)

if df is None:
    st.error("❌ No data found. Check ticker symbol.")
    st.stop()

st.success("✅ Data loaded successfully!")
st.write(df.tail())

# ----------------------
# Technical Indicators
# ----------------------
st.subheader("📊 Technical Analysis Indicators")

df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["EMA_20"] = df["Close"].ewm(span=20).mean()
df["EMA_50"] = df["Close"].ewm(span=50).mean()
df["Returns"] = df["Close"].pct_change()

# RSI
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

st.write(df[["Close", "SMA_20", "SMA_50", "EMA_20", "EMA_50", "RSI"]].tail())

# Plot technical chart
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df["Close"], label="Close Price")
ax.plot(df.index, df["SMA_20"], label="SMA 20")
ax.plot(df.index, df["SMA_50"], label="SMA 50")
ax.plot(df.index, df["EMA_20"], label="EMA 20")
ax.plot(df.index, df["EMA_50"], label="EMA 50")
ax.legend()
st.pyplot(fig)

# ----------------------
# Fundamental Analysis
# ----------------------
st.subheader("📑 Fundamental Analysis")

stock = yf.Ticker(ticker)

st.write("### 🏢 Company Info")
st.write(stock.info)

st.write("### 💰 Balance Sheet")
st.write(stock.balance_sheet)

st.write("### 🔄 Cash Flow")
st.write(stock.cashflow)

st.write("### 📦 Quarterly Earnings")
st.write(stock.quarterly_earnings)

# ----------------------
# ARIMA FORECASTING
# ----------------------
st.subheader("🔮 ARIMA Forecasting")

close_prices = df["Close"]

# Fit ARIMA Model
model = ARIMA(close_prices, order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=forecast_steps)

# ----------------------
# Plot 1 – Change in Price (Actual Only)
# ----------------------
st.write("### 📈 Price History (Actual)")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(close_prices.index, close_prices, label="Actual Price")
ax.legend()
st.pyplot(fig)

# ----------------------
# Plot 2 – Actual vs Forecast (Overlapped)
# ----------------------
st.write("### 🔁 ARIMA Forecast (Overlapped with Actual)")

future_index = pd.date_range(start=close_prices.index[-1], periods=forecast_steps+1, freq="M")[1:]
forecast_series = pd.Series(forecast.values, index=future_index)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(close_prices.index, close_prices, label="Actual Price")
ax.plot(forecast_series.index, forecast_series, label="Forecast", linestyle="--")
ax.legend()
st.pyplot(fig)

# ----------------------
# Plot 3 – Forecast Only
# ----------------------
st.write("### 🚀 Forecast Future Price")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(forecast_series.index, forecast_series, label="Future Forecast", linestyle="--")
ax.legend()
st.pyplot(fig)

st.success("✨ Forecasting Completed!")
