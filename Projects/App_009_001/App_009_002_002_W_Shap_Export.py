### Working Correctly
################################################################################################################
## Streamlit App 009 - 02 - 02/02 Streamlit Web App
################################################################################################################
## App 009 - California Housing Regression - With Shapley
################################################################################################################
## Streamlit Web App
## Creating the Streamlit Web App using the model created on prior python file
################################################################################################################
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
import pickle
import shap
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide")## New Modif

st.header("""California Housing Data App""")

image_01 = Image.open('Projects/App_009_001/Data/Images/Housing.jpg')
st.image(image_01, width=500)

st.write("""
#### This Web App predicts the Median House Value given a set of Parameters.
""")

st.write("""---""")

calif_raw = pd.read_csv('Projects/App_009_001/App_009_001_Exported/Data/housing.csv')
calif_raw['total_bedrooms'] = calif_raw['total_bedrooms'].replace({np.nan:calif_raw['total_bedrooms'].median()})
X = calif_raw.drop(columns=['median_house_value'], axis=1)

def user_input_features():
    ocean_proximity = st.sidebar.selectbox('ocean_proximity', ('NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'))
    longitude = st.sidebar.slider('longitude', X['longitude'].min(), X['longitude'].max(), X['longitude'].mean())
    latitude = st.sidebar.slider('latitude', X['latitude'].min(), X['latitude'].max(), X['latitude'].mean())
    housing_median_age = st.sidebar.slider('housing_median_age', X['housing_median_age'].min(), X['housing_median_age'].max(), X['housing_median_age'].mean())
    total_rooms = st.sidebar.slider('total_rooms', X['total_rooms'].min(), X['total_rooms'].max(), X['total_rooms'].mean())
    total_bedrooms = st.sidebar.slider('total_bedrooms', X['total_bedrooms'].min(), X['total_bedrooms'].max(), X['total_bedrooms'].mean())
    population = st.sidebar.slider('population', X['population'].min(), X['population'].max(), X['population'].mean())
    households = st.sidebar.slider('households', X['households'].min(), X['households'].max(), X['households'].mean())
    median_income = st.sidebar.slider('median_income', X['median_income'].min(), X['median_income'].max(), X['median_income'].mean())
    data = {'longitude':longitude,
            'latitude':latitude,
            'housing_median_age':housing_median_age,
            'total_rooms':total_rooms,
            'total_bedrooms':total_bedrooms,
            'population':population,
            'households':households,
            'median_income':median_income,
            'ocean_proximity':ocean_proximity}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

################################################################################################################################################################
## Opening raw csv
calif_raw = pd.read_csv('Projects/App_009_001/App_009_001_Exported/Data/housing.csv')

## Replacing missing values with median values
calif_raw['total_bedrooms'] = calif_raw['total_bedrooms'].replace({np.nan:calif_raw['total_bedrooms'].median()})
calif_features = calif_raw.drop(columns=['median_house_value'], axis=1)

df = pd.concat([input_df, calif_features], axis=0)

## Getting the mean and median values of labels to compare it against prediction
calif_labels = calif_raw['median_house_value']
calif_labels_mean = np.round(calif_labels.mean(),2)
calif_labels_median = np.round(calif_labels.median(),2)

####################### Preprocessing phase

### Dummy variables
encode = ['ocean_proximity']
dummies_01 = pd.get_dummies(data=df[encode], prefix=None, prefix_sep = '_', dummy_na = False, columns = None, sparse = False, drop_first = False, dtype = None).astype('int')

#dummies_01.columns
dummies_01.rename(mapper={'ocean_proximity_<1H OCEAN':'<1H OCEAN',
                            'ocean_proximity_INLAND':'INLAND',
                            'ocean_proximity_ISLAND':'ISLAND',
                            'ocean_proximity_NEAR BAY':'NEAR BAY',
                            'ocean_proximity_NEAR OCEAN':'NEAR OCEAN'}, axis=1, inplace=True)
#dummies_01

## Dropping original features
features_float_01 = df.drop(columns=calif_features[encode], axis=1)
#features_float_01

## I have features_float_01 which is All features except dummy variables and without Y class label
## I have dummy variables
## I have the user input dataframe
## Concatenating features_float_01 and dummy variables

### Concatenation
df = pd.concat(objs=[features_float_01, dummies_01], axis=1)
df = df[:1]

##################################################################
##################################################################

### Main Panel

### Print Specified Input Parameters
st.write("""#### Specified Input Parameters""")
## Show DataFrame
st.write(df)
st.write('---')

## Open ML Model - Regression Model
### Reading the Model
load_reg = pickle.load(open('Projects/App_009_001/App_009_001_Exported/Data/Saved_Models/Model_calif_housing_forest.pkl', 'rb'))

## Apply model to make prediction
## A) Can either use this
model_prediction = np.round(load_reg.predict(df), 2)
str_model_prediction = str(model_prediction[0])

## Or using this B)
### show the prediction as a DF -> Did not chose this
#model_prediction = pd.DataFrame([model_prediction], columns=['model prediction'])


#st.write('#### Prediction')
#st.write('#### Median House Valus Prediction')
st.write(f'### Median House Value Prediction $ {str_model_prediction}')

### Changing Font size and colour
### Help info from snehankekre Source https://discuss.streamlit.io/t/change-font-size-and-font-color/12377/3
#new_title = '<p style="color:DarkBlue; font-size:42px;"> Text </p>'
#new_title = f'<p style="color:Black; font-size:32px;"> Median House Value Prediction $ {str_model_prediction} </p>'
#st.markdown(new_title, unsafe_allow_html=True)

#########################
##### y Label mean and median To compare it against Prediction
st.write(f'#### Class Label Mean: $ {calif_labels_mean}')
#res_mean = ('{:,}'.format(calif_labels_mean))
#st.write(f'#### Class Label Mean: $ {res_mean}')

st.write(f'#### Class Label Median: $ {calif_labels_median}')
#res_median = ('{:,}'.format(calif_labels_median))
#st.write(f'#### Class Label Median: $ {res_median}')

#########################
st.write('')
st.write('')
st.write('')
st.write('')

###################################################################################################
### Shapley Value Plots
###################################################################################################
### Source: https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/8
#####I just stumbled on this one, it’s now easier with Streamlit Components and latest SHAP v0.36+ (which define a new getjs method), to plot JS SHAP plots
#####(some plots like summary_plot are actually Matplotlib and can be plotted with st.pyplot)
#####Here is an example.
### By: andfanilo
import streamlit.components.v1 as components

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(load_reg)
shap_values = explainer.shap_values(df)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
#st.markdown("### SHAP in Streamlit")
st.markdown("### Shapley Value Plots")

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], df.iloc[0,:]))

# visualize the training set predictions
#st_shap(shap.force_plot(explainer.expected_value, shap_values, df), 400)
###################################################################################################
###################################################################################################

st.write("""---""")

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(load_reg)
shap_values = explainer.shap_values(df)

## Working, but with deprecation warnings
#st.header('Feature Importance')
#fig, ax = plt.subplots(nrows=1,ncols=1)
#plt.title('Feature Importance Based on Shapley Values')
#shap_figure_01 = shap.summary_plot(shap_values, df)
#st.pyplot(shap_figure_01, bbox_inches='tight')

## Working correctly, no deprecation warnings
st.header('Feature Importance')
fig01, ax01 = plt.subplots(nrows=1,ncols=1)
plt.title('Feature Importance Based on Shapley Values')
shap_figure_01 = shap.summary_plot(shap_values, df)
st.pyplot(fig01, bbox_inches='tight')

st.write(""" """)
st.write(""" """)
st.write(""" """)

fig02, ax02 = plt.subplots(nrows=1,ncols=1)
plt.title('Feature Importance Based on Shapley Values (Bar Plot)')
shap_figure_02 = shap.summary_plot(shap_values, df, plot_type="bar")
st.pyplot(fig02, bbox_inches='tight')


###################################################################################################
###################################################################################################
#st.write('---')
#st.write("""
#Shapley determines the importance of feature used on the model. They are assigned to each feature.
#""")

###################################################################################################


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
         [Data Source Downloaded From Kaggle - dhirajnirne](https://www.kaggle.com/datasets/dhirajnirne/california-housing-data)\n
         [Data Source From Kaggle - darshanprabhu09](https://www.kaggle.com/datasets/darshanprabhu09/california-housing-dataset)\n
         [Streamlit](https://streamlit.io/)
         [Streamlit help on Shapley Value Plot by andfanilo](https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/8)
         """)
st.write('')
st.write('')
st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')
st.write("##### Thank you kindly to all who make information and knowledge available for free.")
