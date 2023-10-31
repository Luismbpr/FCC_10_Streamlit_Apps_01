################################################################################################################
## Streamlit App 005 - 01 - 01/01
################################################################################################################
## Notes:
## App 005 - S&P 500 Stock Closing Price
################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import streamlit as st
import base64
import yfinance as yf

st.title('S&P 500 App')

st.markdown("""
This App retrieves information of the S&P 500 (From Wikipedia) and its corresponding **stock closing price** (Year-to-date)
            
""")

st.sidebar.header('User Input Features')

## Web Scraping S&P 500 Data
@st.cache_data
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

df = load_data()
sector = df.groupby(by='GICS Sector')

## Sidebar - Sector Selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
selected_sector = st.sidebar.multiselect('Choose at least one Sector', sorted_sector_unique, sorted_sector_unique)
st.sidebar.markdown("*Note: Error will be displayed if no sector is displayed.*")

## Filtering Data based on
## Sector - Filtering selected sectors
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
st.dataframe(df_selected_sector)

## Download S&P Data to csv format
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() ## Strings <-> Bytes Conversion
    ## Error:
    #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
    ## Solution:
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

## Retrieve YF Stock Price Data
## Way 01 pdr.get_data_yahoo(...) tickers list or string as well
data = yf.download(
    tickers = list(df_selected_sector[:10]['Symbol']),
    period = "ytd",
    interval = "1d",
    group_by = 'ticker',
    auto_adjust = True,
    prepost = True,
    threads = True,
    proxy = None
)

## Plot Closing Price
def price_plot(symbol):
    df = pd.DataFrame(data[symbol]['Close'])
    df['Date'] = df.index
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,2))
    axes.plot(df['Date'].to_numpy(), df['Close'].to_numpy(), alpha=0.8)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    #axes.set_xlabel('Date', fontweight='bold', fontsize=6)
    #axes.set_ylabel('Closing Price (USD)', fontweight='bold', fontsize=6)
    #axes.set_title(symbol, fontweight='bold', fontsize=6)
    #axes.set_xlabel('Date', fontsize=6)
    axes.set_ylabel('Closing Price (USD)', fontsize=6)
    axes.set_title(symbol, fontsize=6)
    #plt.tight_layout()
    return st.pyplot(fig)

## Slider from 1 to 5
num_company = st.sidebar.slider("Number of companies", 1, 5)

## All rows and all columns from 1 to x in this case 5: then plot
if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_selected_sector['Symbol'])[:num_company]:
        price_plot(i)


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
st.write('')



st.write('### Resources:')
st.write("""
         Data Source: [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)\n
         [Download S&P Data to csv format help](https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806)\n
         [Streamlit](https://streamlit.io/)
         """)

st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')

st.write('##### Thank you to all of you who make information and knowledge available for free.')