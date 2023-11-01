################################################################################################################
## Streamlit App 001 - 01 - 01/01
################################################################################################################
## Notes:
## Stock Closing Price and Volume Price
################################################################################################################

import pandas as pd
import streamlit as st
import yfinance as yf

st.write("""
# Stock Price App

##### Web App that shows the Stock **Closing Price** and *Volume* for a specific company.

""")


st.sidebar.write('Insert Ticker Symbol Here:')
ticker_input = st.sidebar.text_input('Ticker Input', 'GOOGL')
st.sidebar.write('Note: Please make sure it is a correct ticker symbol.')
st.sidebar.write('Examples: "AAPL", "GOOG", "HOG", "INTC"')

##User Input Text
tickerSymbol = ticker_input
tickerData = yf.Ticker(tickerSymbol)

tickerDF = tickerData.history(period='1d', start='2010-05-31', end='2023-10-25')

st.write('')
if (len(tickerDF) == 0):
    st.markdown(f"#### Ticker input: '{tickerSymbol}'.")
    st.markdown("#### Please Provide A Valid Ticker Symbol")
    st.write('')
else:
    st.write('')
    st.write('')


st.markdown(f"<h3 style='text-align: center; color: black;'>Closing Price '{tickerSymbol}'</h1>", unsafe_allow_html=True)
st.line_chart(data=tickerDF['Close'], 
            x=None,
            y=None,
            width=0,
            height=0,
            use_container_width=True)

st.write('')
st.write('')

st.markdown(f"<h3 style='text-align: center; color: black;'>Volume Price '{tickerSymbol}'</h1>", unsafe_allow_html=True)
st.line_chart(data=tickerDF['Volume'], 
            x=None,
            y=None,
            width=0,
            height=0,
            use_container_width=True)

st.write('---')
st.write('')
st.write('')

df_comp_w_ticker = pd.DataFrame({
    "Company": ['Wal-Mart', 'Exxon Mobil','Chevron','Phillips 66','Berkshire Hathaway','Apple','General Motors','General Electric','Valero Energy','Ford Motor','CVS Caremark','McKesson','Hewlett-Packard','Verizon','United Health Care','J.P. Morgan Chase','Cardinal Health','International Business Machines','Bank of America','Costco Wholesale','Kroger'],
    "Ticker_symbol": ['WMT','XOM','CVX','PSX','BRKA','AAPL','GM','GE','VLO','F','CVS','MCK','HPQ','VZ','UNH','JPM','CAH','IBM','BAC','COST', 'KR']
})

#st.write("##### Examples of Company and Ticker Symbols", df_comp_w_ticker)
st.write("##### Examples of Company and Ticker Symbols", df_comp_w_ticker)
st.write('')

st.write('---')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')


st.write('### Resources:')
st.write("""
        [Yahoo Finance for price stocks](https://finance.yahoo.com/lookup/)\n
        [Wikipedia for Ticker symbol information](https://en.wikipedia.org/wiki/Ticker_symbol)\n
        [Streamlit](https://streamlit.io/)
         """)

st.write('###### [Special Thanks to Free Code Camp and Chanin Nantasenamat](https://www.freecodecamp.org/).')
st.write('###### Thank you kindly to all who make information and knowledge available for free.')
