# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide")
st.title("📊 Magnificent Stocks Time Series Decomposition App")

# Sidebar inputs
st.sidebar.header("Stock Settings")
ticker = st.sidebar.selectbox(
    "Select a stock ticker",
    ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA'],
    index=0
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2016-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

model_type = st.sidebar.selectbox("Decomposition Model", ['additive', 'multiplicative'])

# Fetch data from Yahoo Finance
@st.cache_data
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)[['Adj Close']]
    df = df.rename(columns={'Adj Close': 'Close'})
    df = df.asfreq('B')  # Business day frequency
    df['Close'] = df['Close'].interpolate()
    return df

df = fetch_stock_data(ticker, start_date, end_date)

# Plot actual time series
st.subheader(f"📈 {ticker} Stock Price (from {start_date})")
st.line_chart(df['Close'])

# Decomposition
st.subheader("📉 Time Series Decomposition")

try:
    result = seasonal_decompose(df['Close'], model=model_type, period=252)

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(result.observed, label='Observed', color='blue')
    axs[0].set_title('Observed')

    axs[1].plot(result.trend, label='Trend', color='orange')
    axs[1].set_title('Trend')

    axs[2].plot(result.seasonal, label='Seasonality', color='green')
    axs[2].set_title('Seasonality')

    axs[3].plot(result.resid, label='Residual', color='red')
    axs[3].set_title('Residual (White Noise)')

    plt.tight_layout()
    st.pyplot(fig)

except ValueError as e:
    st.warning(f"Unable to perform decomposition: {e}")
