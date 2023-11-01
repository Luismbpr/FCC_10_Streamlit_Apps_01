################################################################################################################
## Streamlit App 004 - 01 - 01/01
################################################################################################################
## Notes:
## NFL Player Stats
################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import streamlit as st
import base64


st.header("NFL Football Stats (Rushing) Explorer")
st.write('')

image_01 = Image.open(fp='Projects/App_004_001/Data/Images/drawing_nfl_01.jpg', mode='r', formats=None)
st.image(image_01, use_column_width=True)

st.write('')

st.markdown("""
This App performs web scraping of NFL Football Player Stats Data (Focusing on Rushing).
""")

## Sidebar
st.sidebar.header('User Input Features')
## Sidebar - Year
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1990, 2024))))

## Web scraping website
@st.cache_data
def load_data(year):
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/rushing.htm'
    html = pd.read_html(url, header=1)
    df = html[0]
    raw = df.drop(df[df["Age"] == "Age"].index)## Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(columns=['Rk'], axis=1)## Dropping index since there is already one with Pandas
    return playerstats
playerstats = load_data(selected_year)

## Sidebar - Team
sorted_unique_team = sorted(playerstats['Tm'].unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

## Sidebar - Position
unique_pos = ['QU', 'RB', 'FB', 'WR', 'TE', 'C']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

## Filtering Data
df_selected_team = playerstats[(playerstats['Tm'].isin(selected_team)) & (playerstats['Pos'].isin(selected_pos))]

st.header('Display PlayerStats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + 'rows and ' + str(df_selected_team.shape[1]) + 'columns.')
st.dataframe(df_selected_team)

## Download Player Stats Data
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() ## Strings <-> Bytes
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download csv file</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

## Heatmap
## Load new csv file
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    
    corr = df.corr(numeric_only=True)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, fmt='.2f')
    st.pyplot(fig)
    

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
         Data Source: [https://www.pro-football-reference.com/](https://www.pro-football-reference.com/)\n
         [Streamlit](https://streamlit.io/)
         """)

st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')

st.write('##### Thank you kindly to all who make information and knowledge available for free.')
