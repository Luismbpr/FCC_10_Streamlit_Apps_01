################################################################################################################
## Streamlit App 007 - 01 - 01/01
################################################################################################################
## App 007 - Iris Classification
################################################################################################################
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
         # Iris Flower Classification App
         ## This App predicts the different types of varieties from the classic Iris Dataset.
         """)

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length':sepal_length,
            'sepal_width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
## Print out the DataFrame
st.write(df)

iris  = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


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
         Data Source: [Scikit-Learn Datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#)\n
         [Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.](https://archive.ics.uci.edu/dataset/53/iris)\n
         [Streamlit](https://streamlit.io/)
         """)
st.write('###### [Special Thanks to Free Code Camp and Chanin Nantasenamat](https://www.freecodecamp.org/).')
st.write('###### Thank you kindly to all who make information and knowledge available for free.')
