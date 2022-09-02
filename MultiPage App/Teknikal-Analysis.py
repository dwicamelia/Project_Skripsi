import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import talib
import util2

def main():
    ticker_data = util2.get_ticker_data(ticker_symbol, data_period, data_interval)

    if len(ticker_data) != 0:
        ticker_data['fast_ema'] = talib.EMA(ticker_data['Close'], int(ema1))
        ticker_data['slow_ema'] = talib.EMA(ticker_data['Close'], int(ema2))
        ticker_data['rsi'] = talib.RSI(ticker_data['Close'], timeperiod=14)
        ticker_data['sar'] = talib.SAR(ticker_data['High'], ticker_data['Low'], acceleration=0.02, maximum=0.2)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig = util2.get_candle_chart(fig, ticker_data)
        fig = util2.add_row_trace(fig, ticker_data.index, ticker_data['fast_ema'], 'fast EMA', 'yellow', 1)
        fig = util2.add_row_trace(fig, ticker_data.index, ticker_data['slow_ema'], 'slow EMA', 'blue', 1)
        fig = util2.add_row_trace(fig, ticker_data.index, ticker_data['sar'], 'Parabolic SAR', 'black', 1, mode='markers')
        fig = util2.add_row_trace(fig, ticker_data.index, ticker_data['rsi'], 'RSI', 'red', 2)

        st.write(fig)
    
if __name__ == '__main__':
    ticker_symbol = st.sidebar.text_input(
    "Please enter the stock symbol", 'BBCA.JK'
    )
    data_period = st.sidebar.text_input('Period', '10d')
    data_interval = st.sidebar.radio('Interval', ['15m','30m','1h','1d'])
    ema1 = st.sidebar.text_input('EMA 1', 20)
    ema2 = st.sidebar.text_input('EMA 2', 50)

    st.header("Aplikasi Analisis Teknikal :rocket:")
    st.write("*Warning: Hasil yang ditampilkan tidak disarankan menjadi keputusan investasi, Pertimbangkan faktor lainnya!* :sunglasses:")

    main()