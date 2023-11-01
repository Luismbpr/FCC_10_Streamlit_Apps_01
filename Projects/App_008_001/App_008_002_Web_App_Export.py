### Working Correctly
################################################################################################################
## Streamlit App 008 - 02 - 02/02 Streamlit Web App
################################################################################################################
## App 008 - Penguin Classification
################################################################################################################
## Streamlit Web App
## Creating the Streamlit Web App using the model created on prior python file
################################################################################################################

import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image


df = pd.read_csv('Projects/App_008_001/App_008_001_Exported/Data/penguins_cleaned.csv')

image_01 = Image.open(fp='Projects/App_008_001/App_008_001_Exported/Data/Images/penguins_01.jpeg', mode='r', formats=None)
st.image(image_01, use_column_width=True)

st.write("""
         # Penguin Classification App
         ###### This App predicts the Penguin Species given a set of penguin description parameters.
         """)

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
                    [Example CSV Input File]([Example CSV input file](https://github.com/Luismbpr/FCC_10_Streamlit_Apps_01/blob/main/Projects/App_008_001/Data/penguins_new_input_same_order.csv)
                    """)

st.sidebar.markdown("""
###### Right click on the link above. Remember to save the file with a '.csv' format.
""")

st.sidebar.markdown("""---""")

### Collecting User Input Features into DataFrame
### A) User Inputs a CSV File with the required fields
### B) User Uses Inputs the fields from the App

### Error: st.sidebar.uploaded_file
### Solution
uploaded_file = st.sidebar.file_uploader("Upload your input CSV File", type=["csv"])
if uploaded_file is not None:
    ### Error I used variable name df when it was input_df
    #Error:df = pd.read_csv(uploaded_file)
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island",('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox("Sex", ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        ### Selecting the first one with index 0
        ### Error: index=0
        ### Solution: index=[0]
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

## Combining user input with the entire dataset
## This will be useful in the encoding phase

penguins_raw = pd.read_csv('Projects/App_008_001/App_008_001_Exported/Data/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'], axis=1)
df = pd.concat([input_df, penguins], axis=0)

#####################################################################################
### Encoding - Ordinal Features - No Modification
#encode = ['sex', 'island']
#for col in encode:
#    #pd.get_dummies(df[col], prefix=col)
#    dummy = pd.get_dummies(df[col], prefix=col).astype('int')
#    df = pd.concat([df, dummy], axis=1)
#    ### Note If error if this does not work I will need to drop manually the columns
#    ## Error: ValueError: X has 9 features, but RandomForestClassifier is expecting 7 features as input.
#    #del df[col]
#    df.head()
#df = df.drop(columns=['sex', 'island'])
#df = df[:1] ## Selecting only first row (user input data)



#####################################################################################
### Encoding - Ordinal Features - With My Modification
## Modifying dummy variables
## dummies 
#encode = ['sex', 'island']
#dummy = pd.get_dummies(df[encode], drop_first=True).astype('int')
#dummy
#df = pd.concat(objs=[df, dummy], axis=1)
#df = df.drop(columns=['island', 'sex'])
## Error: df = df.drop(columns=[encode])
## Solution: df = df.drop(columns=encode)
#df = df.drop(columns=encode)
#df.columns
## Selection of the first row (first index with slicing method)
#df = df[:1] ## Selecting only first row (user input data)

#Index(['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
#       'sex_male', 'island_Dream', 'island_Torgersen'],
#      dtype='object')

#df02.head()

##Note: Trained model has this features in this order
##['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 
##'sex_male', 'island_Dream', 'island_Torgersen']


encode = ['sex', 'island']
dummy = pd.get_dummies(df[encode], drop_first=True).astype('int')
df = pd.concat(objs=[df, dummy], axis=1)
df = df.drop(columns=encode)
df = df[:1] ## Selecting only first row (user input data)

#####################################################################################

### Display the user input features
st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV File to be uploaded. Currently using example input parameters (shown below)')
    st.write(df)

### Read in saved classification model (Random Forest Classifier)
### pickle.load(open(path, 'rb'))
load_clf = pickle.load(open('Projects/App_008_001/App_008_001_Exported/Data/Saved_Models/model_02.pkl', 'rb'))

## Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.write('')
st.write('')
st.write('')

st.markdown('### Prediction Results')
### Note: markdown with centered text is causing issues when using resized web app 
#st.markdown("<h3 style='text-align:center;'>Prediction Results </h3>", unsafe_allow_html=True)
st.write('')

#st.subheader('Prediction')
###df['species'].unique()
##array(['Adelie', 'Gentoo', 'Chinstrap'], dtype=object)
#penguin_species = np.array(['Adelie', 'Gentoo', 'Chinstrap'])
#st.subheader('Prediction')
penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
#st.write(penguin_species[prediction])

#st.subheader('Prediction Probabilities')
st.markdown('#### Prediction Probabilities')
st.write(prediction_proba)


df_02_data = {"Prediction Probabilities":str(np.round(prediction_proba, 2)),
              "Index of prediction":[prediction],
              "Prediction Name":penguin_species[prediction],
              "Prediction Probability":str(np.round(prediction_proba.max() * 100, 2)) + ' %'
              }

df_02 = pd.DataFrame(df_02_data)
st.write(df_02)

st.write('')

name_prediction = penguin_species[prediction][0]
st.markdown(f"""
         ##### Given the input parameters, the model predicts that the species is: *{name_prediction}* with {str(np.round(prediction_proba.max() * 100,2))} % confidence
         """)
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
st.write('')



st.write('### Resources:')
st.write("""
         [Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package version 0.1.0. https://allisonhorst.github.io/palmerpenguins/. doi: 10.5281/zenodo.3960218.](https://allisonhorst.github.io/palmerpenguins/authors.html#citation)\n
         [Data Source: Dataset Derived from](https://github.com/allisonhorst/palmerpenguins/)\n
         [Allison Horst - Author, maintainer.](https://allisonhorst.github.io/palmerpenguins/)\n
         [Allison Hill - Author](https://www.apreshill.com/)\n
         [Kristen Gorman - Author](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)\n
         [Streamlit](https://streamlit.io/)
         """)

st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')
st.write("##### Thank you kindly to all who make information and knowledge available for free.")