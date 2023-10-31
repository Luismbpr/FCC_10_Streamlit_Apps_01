################################################################################################################
## Streamlit App 003 - 01 - 01/01
################################################################################################################
## Notes:
## NBA Player Stats
################################################################################################################
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import streamlit as st

image_01 = Image.open(fp='./Projects/App_003_001/App_003_001_Exported/Data/Images/Facebook Cover 01.jpeg', mode='r', formats=None)
st.image(image_01, use_column_width=True)

st.markdown("""
#### This App Performs Webscraping of NBA Player Stats Data.
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2024))))

## Web scraping NBA Player Stats
@st.cache_data

def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df['Age'] == 'Age'].index)## Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)


sorted_unique_team = sorted(playerstats['Tm'].unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

unique_pos_explanation = pd.DataFrame([{'C': 'Center', 'PF': 'Power Forward','SF': 'Small Forward','PG': 'Point Guard','SG': 'Shooting Guard'}])
st.sidebar.table(unique_pos_explanation)

## Filtering Data
df_selected_team = playerstats[(playerstats['Tm'].isin(selected_team)) & (playerstats['Pos'].isin(selected_pos))]

st.markdown('### Display Player Stats of Selected Teams')
st.write('Data Contains ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

## Download NBA Player Stats Data
## https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() ## Strings <--> Bytes Conversions
    href = f'<a href="data:file/csv;base64, {b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

st.write("#### Heatmap")
### Heatmap 01
if st.button('Correlation Heatmap No Categorical Columns'):
    st.header('Correlation Heatmap')
    st.write("Heatmap *not including* Categorical columns")
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    df_selected_htmp = df.drop(columns=['Player', 'Pos', 'Tm'], axis=1)
    corr = df_selected_htmp.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig02, ax02 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
        ax02 = sns.heatmap(corr, mask=mask, vmax=1, square=True, center=0, cmap='coolwarm')
    st.pyplot(fig02)


### Heatmap 02
if st.button('Correlation Heatmap With Categorical Columns'):
    st.header('Correlation Heatmap')
    st.write("Heatmap including Categorical columns (Position, Team)")
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    df_dummies_w_pos = df[['Pos','Tm']]
    df_dummies_w_pos = pd.get_dummies(data=df_dummies_w_pos, drop_first=False).astype('int')
    df_selected_team_htmp = df.drop(columns=['Player', 'Pos', 'Tm'], axis=1)
    df_selected_team_htmp = pd.concat(objs=[df_selected_team_htmp, df_dummies_w_pos], axis=1)
    corr = df_selected_team_htmp.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        fig01, ax01 = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
        ax01 = sns.heatmap(corr, mask=mask, vmax=1, square=True, center=0, cmap='coolwarm')
    st.pyplot(fig01)


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
         Data Source: [basketball-reference.com](https://www.basketball-reference.com/)\n
         [Streamlit](https://streamlit.io/)\n
         [Download NBA Player Stats Data Help](https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806)
         """)

st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')

st.write('##### Thank you kindly to all who make information and knowledge available for free.')