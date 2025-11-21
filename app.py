import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pmdarima import auto_arima

st.title("📈 Universal ARIMA Stock Forecasting App")

# -------------------------------
# USER INPUT
# -------------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS, AAPL, TSLA):")

if ticker:

    st.write(f"### Fetching data for **{ticker}** ...")

    try:
        # -------------------------------
        # DOWNLOAD FULL DATA FROM YAHOO
        # -------------------------------
        data = yf.download(ticker, period="max", interval="1d")

        if data.empty:
            st.error("❌ No data found. Check the ticker name.")
            st.stop()

        st.success("✔ Data downloaded!")

        # Convert daily → monthly
        monthly = data["Close"].resample("M").last().dropna()

        st.write("### 📌 Monthly Closing Prices")
        st.dataframe(monthly.tail())

        # -------------------------------
        # 1️⃣ PRICE TREND
        # -------------------------------
        st.subheader("📌 1. Monthly Price Trend")

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(monthly, label="Monthly Close")
        ax1.set_title(f"{ticker} Monthly Price Trend")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1)

        # -------------------------------
        # TRAIN ARIMA MODEL
        # -------------------------------
        st.subheader("📌 Training ARIMA Model...")
        with st.spinner("Fitting ARIMA model..."):
            model = auto_arima(monthly, seasonal=False, error_action='ignore')

        st.success("✔ Model training complete!")

        # -------------------------------
        # 2️⃣ FORECAST OVER ACTUAL
        # -------------------------------
        st.subheader("📌 2. ARIMA Forecast vs Actual")

        forecast_fit = model.predict(n_periods=len(monthly))

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(monthly.index, monthly, label="Actual")
        ax2.plot(monthly.index, forecast_fit, label="Forecasted")
        ax2.set_title("ARIMA Actual vs Forecast")
        ax2.legend()
        st.pyplot(fig2)

        # -------------------------------
        # 3️⃣ FUTURE 12-MONTH FORECAST
        # -------------------------------
        st.subheader("📌 3. Next 12 Months Forecast")

        future_forecast = model.predict(12)
        future_dates = pd.date_range(monthly.index[-1] + pd.offsets.MonthEnd(), periods=12, freq="M")

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(monthly.index, monthly, label="History")
        ax3.plot(future_dates, future_forecast, "--", label="Future Forecast")
        ax3.set_title("12-Month ARIMA Forecast")
        ax3.legend()
        st.pyplot(fig3)

        st.success("🎉 Forecast completed successfully!")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Enter any stock ticker to begin forecasting.")
