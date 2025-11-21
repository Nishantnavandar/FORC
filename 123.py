import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima import auto_arima

st.title("📈 Smart ARIMA Stock Forecasting App (Auto-Ticker Search)")

# ---------------------------------------
# FUNCTION TO SEARCH TICKER BY COMPANY NAME
# ---------------------------------------
def search_ticker(query):
    try:
        results = yf.Ticker(query).history(period="1d")
        if not results.empty:
            return query  # valid ticker
    except:
        pass

    # If the above fails → use Yahoo Suggest API
    import requests
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    response = requests.get(url).json()

    if "quotes" in response and len(response["quotes"]) > 0:
        return response["quotes"][0]["symbol"]  # return best match
    return None


# ---------------------------------------
# USER INPUT
# ---------------------------------------
query = st.text_input("Enter Stock Name or Ticker (Example: Reliance, TCS, AAPL, TSLA):")

if query:

    st.write(f"🔍 Searching Yahoo Finance for **{query}** ...")

    ticker = search_ticker(query)

    if not ticker:
        st.error("❌ Could not find this stock on Yahoo Finance. Try another name.")
        st.stop()

    st.success(f"✔ Found Ticker: **{ticker}**")

    # ---------------------------------------
    # DOWNLOAD ALL AVAILABLE DATA
    # ---------------------------------------
    try:
        data = yf.download(ticker, period="max", interval="1d")

        if data.empty:
            st.error("❌ Yahoo Finance returned empty data. Try another stock.")
            st.stop()

        st.success("📥 Data Downloaded Successfully!")

        # Convert to monthly price
        monthly = data["Close"].resample("M").last().dropna()

        st.subheader("📌 Monthly Price Data Preview")
        st.dataframe(monthly.tail())

        # ---------------------------------------
        # 1️⃣ PRICE TREND
        # ---------------------------------------
        st.subheader("📌 1. Monthly Price Trend")

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(monthly, label="Monthly Closing Price")
        ax1.set_title(f"{ticker} - Monthly Price Trend")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1)

        # ---------------------------------------
        # TRAIN ARIMA MODEL
        # ---------------------------------------
        st.subheader("📌 Training ARIMA Model...")
        with st.spinner("Fitting model..."):
            model = auto_arima(monthly, seasonal=False, error_action='ignore')

        st.success("✔ ARIMA Model Trained Successfully!")

        # ---------------------------------------
        # 2️⃣ FORECAST VS ACTUAL
        # ---------------------------------------
        st.subheader("📌 2. ARIMA Forecast vs Actual")

        forecast_fit = model.predict(n_periods=len(monthly))

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(monthly.index, monthly, label="Actual")
        ax2.plot(monthly.index, forecast_fit, label="ARIMA Forecast")
        ax2.set_title(f"{ticker} – ARIMA Fit")
        ax2.legend()
        st.pyplot(fig2)

        # ---------------------------------------
        # 3️⃣ FUTURE 12-MONTH FORECAST
        # ---------------------------------------
        st.subheader("📌 3. Forecast for Next 12 Months")

        future_forecast = model.predict(12)
        future_dates = pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(),
                                     periods=12, freq="M")

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(monthly.index, monthly, label="Historical")
        ax3.plot(future_dates, future_forecast, "--", label="Future Forecast")
        ax3.set_title(f"{ticker} – 12-Month Future Forecast")
        ax3.legend()
        st.pyplot(fig3)

        st.success("🎉 Forecasting Completed Successfully!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Enter a company name or ticker to begin.")
