import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.title("📈 Stock Forecasting + Technical & Fundamental Analysis (SAFE MODE)")

ticker = st.text_input("Enter Stock Ticker (Example: RELIANCE.NS, AAPL)", "RELIANCE.NS")

forecast_period = st.selectbox(
    "Select Forecast Period",
    ["6 Months", "1 Year", "2 Years"]
)

period_map = {"6 Months": 6, "1 Year": 12, "2 Years": 24}
forecast_steps = period_map[forecast_period]

# ---------------------------------------
# Fetch SAFE stock data
# ---------------------------------------
def get_data(ticker):
    try:
        data = yf.download(ticker, period="max", interval="1mo")
        if data.empty:
            return None
        return data
    except:
        return None

df = get_data(ticker)

if df is None:
    st.error("❌ No data found.")
    st.stop()

st.success("✅ Data loaded successfully!")
st.write(df.tail())

# ---------------------------------------
# SAFE FUNDAMENTAL DATA (no rate limit)
# ---------------------------------------
st.subheader("📑 Fundamental Summary (Safe Mode – No Rate Limit)")

stock = yf.Ticker(ticker)

safe_fundamental = {
    "Company Name": stock.fast_info.get("longName", "N/A"),
    "Market Cap": stock.fast_info.get("marketCap", "N/A"),
    "Currency": stock.fast_info.get("currency", "N/A"),
    "Previous Close": stock.fast_info.get("previousClose", "N/A"),
    "Year High": stock.fast_info.get("yearHigh", "N/A"),
    "Year Low": stock.fast_info.get("yearLow", "N/A"),
}

st.write(safe_fundamental)

# ---------------------------------------
# Technical Indicators
# ---------------------------------------
st.subheader("📊 Technical Analysis Indicators")

df["SMA_20"] = df["Close"].rolling(20).mean()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["EMA_20"] = df["Close"].ewm(span=20).mean()
df["EMA_50"] = df["Close"].ewm(span=50).mean()

# RSI
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

st.write(df[["Close", "SMA_20", "SMA_50", "EMA_20", "EMA_50", "RSI"]].tail())

# Technical Chart
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df["Close"], label="Close")
ax.plot(df.index, df["SMA_20"], label="SMA 20")
ax.plot(df.index, df["SMA_50"], label="SMA 50")
ax.plot(df.index, df["EMA_20"], label="EMA 20")
ax.plot(df.index, df["EMA_50"], label="EMA 50")
ax.legend()
st.pyplot(fig)

# ---------------------------------------
# ARIMA FORECASTING
# ---------------------------------------
st.subheader("🔮 ARIMA Forecasting")

close_data = df["Close"].dropna()

model = ARIMA(close_data, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=forecast_steps)

# Plot 1 – Actual Price
st.write("### 📈 Actual Price History")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(close_data.index, close_data, label="Actual Price")
ax.legend()
st.pyplot(fig)

# Plot 2 – Overlap Forecast
st.write("### 🔁 Actual vs Forecast")

future_dates = pd.date_range(close_data.index[-1], periods=forecast_steps+1, freq="M")[1:]
forecast_series = pd.Series(forecast, index=future_dates)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(close_data.index, close_data, label="Actual")
ax.plot(forecast_series.index, forecast_series, label="Forecast", linestyle="--")
ax.legend()
st.pyplot(fig)

# Plot 3 – Future Forecast Only
st.write("### 🚀 Future Price Forecast")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(forecast_series.index, forecast_series, label="Forecast", linestyle="--")
ax.legend()
st.pyplot(fig)

st.success("✨ Forecasting Completed without Rate Limit Errors!")

