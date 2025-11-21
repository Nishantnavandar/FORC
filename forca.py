import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

st.title("Universal ARIMA Forecasting App (Auto Yahoo Finance Fetch)")

# -------------------------------
#  USER INPUT
# -------------------------------

ticker = st.text_input("Enter Company Stock Ticker (Example: RELIANCE.NS, TCS.NS, AAPL, TSLA):")

if ticker:

    st.subheader(f"Fetching Data for: {ticker} ...")

    # -------------------------------
    #  DOWNLOAD FULL DATA FROM YAHOO
    # -------------------------------
    data = yf.download(ticker, period="max", interval="1d")  # fetch all available data

    if data.empty:
        st.error("Invalid Ticker or Data Not Available.")
        st.stop()

    st.success("✔ Data Loaded Successfully!")

    # -----------------------------------------
    #  RESAMPLE TO MONTHLY CLOSE PRICE
    # -----------------------------------------
    monthly = data['Close'].resample("M").last()
    monthly = monthly.dropna()

    st.write("### Monthly Data Preview")
    st.dataframe(monthly.tail())

    # -----------------------------------------
    # 1️⃣ PRICE CHANGE LINE CHART
    # -----------------------------------------
    st.subheader("1. Monthly Price Change")

    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(monthly, label="Monthly Closing Price")
    ax1.set_title(f"{ticker} – Monthly Price Trend")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1)

    # -----------------------------------------
    # FIT ARIMA MODEL
    # -----------------------------------------
    st.subheader("Training ARIMA Model...")
    with st.spinner("Auto-fitting ARIMA model..."):
        model = auto_arima(monthly, seasonal=False, error_action='ignore', trace=False)

    st.success("✔ ARIMA Model Trained Successfully!")

    # -----------------------------------------
    # 2️⃣ FORECAST OVER ACTUAL (OVERLAP)
    # -----------------------------------------
    st.subheader("2. ARIMA Forecast vs Actual")

    forecast_fit = model.predict(n_periods=len(monthly))

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(monthly.index, monthly, label="Actual Price")
    ax2.plot(monthly.index, forecast_fit, label="ARIMA Predicted")
    ax2.set_title(f"{ticker} – ARIMA Forecast Over Actual")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    # -----------------------------------------
    # 3️⃣ FUTURE FORECAST (NEXT 12 MONTHS)
    # -----------------------------------------
    st.subheader("📌 3. Forecast for Next 12 Months")

    future_forecast = model.predict(n_periods=12)
    future_dates = pd.date_range(start=monthly.index[-1] + pd.offsets.MonthEnd(), periods=12, freq="M")

    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.plot(monthly.index, monthly, label="Actual Price")
    ax3.plot(future_dates, future_forecast, '--', label="Forecast (Next 12 Months)")
    ax3.set_title(f"{ticker} – 12-Month ARIMA Forecast")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Price")
    ax3.legend()
    st.pyplot(fig3)

    st.success("All charts generated successfully!")

else:
    st.info("Please enter a stock ticker to generate forecast.")
