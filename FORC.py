import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

st.title("📈 Reliance Price ARIMA Forecasting App")

# -------------------------------
# FUNCTION TO PLOT LINE CHARTS
# -------------------------------
def plot_line_chart(data, title, xlabel="Date", ylabel="Price"):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(data, label=title)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# FUNCTION TO PLOT ACTUAL VS ARIMA
# -------------------------------
def plot_overlap(actual, forecast, title):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(actual.index, actual, label="Actual Price")
    ax.plot(actual.index, forecast, label="Predicted (ARIMA)")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# FUNCTION TO PLOT FUTURE FORECAST
# -------------------------------
def plot_future(actual, future_forecast, future_dates, title):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(actual.index, actual, label="Actual Price")
    ax.plot(future_dates, future_forecast, '--', label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# PROJECT SELECTOR
# -------------------------------
project = st.selectbox(
    "Select Project",
    ["Project 1 (2010–2018)", "Project 2 (2021–2025)"]
)

# -------------------------------
# DATA DOWNLOAD BASED ON PROJECT
# -------------------------------
if project == "Project 1 (2010–2018)":
    start, end = "2010-01-01", "2019-01-01"
    future_start = "2019-01-01"
    future_title = "Forecast for 2019 (12 Months)"
else:
    start, end = "2021-01-01", "2025-01-01"
    future_start = "2025-01-01"
    future_title = "Forecast for 2025–2026 (12 Months)"

st.subheader(f"Downloading data from {start} to {end}...")

data = yf.download("RELIANCE.NS", start=start, end=end, interval="1mo")
data = data.dropna()

close_prices = data['Close']

st.success("Data Loaded Successfully!")

# -------------------------------
# PRICE CHANGE PLOT
# -------------------------------
st.subheader("📌 1. Price Change (Line Chart)")
plot_line_chart(close_prices, f"Price Change ({start} to {end})")

# -------------------------------
# ARIMA MODEL FIT
# -------------------------------
st.subheader("📌 2. ARIMA Forecast vs Actual")

with st.spinner("Training ARIMA model..."):
    model = auto_arima(close_prices, seasonal=False, error_action='ignore')

forecast_full = model.predict(n_periods=len(close_prices))
plot_overlap(close_prices, forecast_full, "ARIMA Forecast Over Actual")

# -------------------------------
# FUTURE FORECAST
# -------------------------------
st.subheader("📌 3. Future Forecast")

future_forecast = model.predict(n_periods=12)
future_dates = pd.date_range(start=future_start, periods=12, freq="M")

plot_future(close_prices, future_forecast, future_dates, future_title)

st.success("✔ All Charts Generated Successfully!")
