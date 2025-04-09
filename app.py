import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide")
st.title("📊 Magnificent Stocks Time Series Decomposition App")

# Load Excel from GitHub URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/AliJaffri/Time-Series-Decomposition/main/magnificent_stocks.xlsx"
    df = pd.read_excel(url, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('B')  # Business days
    df = df.interpolate()
    return df

df = load_data()

# Sidebar for user selection
st.sidebar.header("Stock Settings")
ticker = st.sidebar.selectbox("Select a stock", df.columns.tolist())

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2016-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
model_type = st.sidebar.selectbox("Decomposition Model", ['additive', 'multiplicative'])

# Filter and prepare data
series = df[ticker].loc[start_date:end_date]
series.name = 'Close'

# Plot actual time series
st.subheader(f"📈 {ticker} Stock Price (from {start_date})")
st.line_chart(series)

# Decomposition
st.subheader("📉 Time Series Decomposition")

try:
    result = seasonal_decompose(series, model=model_type, period=252)

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(result.observed, label='Observed', color='blue')
    axs[0].set_title('Observed')

    axs[1].plot(result.trend, label='Trend', color='orange')
    axs[1].set_title('Trend')

    axs[2].plot(result.seasonal, label='Seasonality', color='green')
    axs[2].set_title('Seasonality')

    axs[3].plot(result.resid, label='Residual', color='red')
    axs[3].set_title('Residual')

    plt.tight_layout()
    st.pyplot(fig)

except ValueError as e:
    st.warning(f"Unable to decompose time series: {e}")
