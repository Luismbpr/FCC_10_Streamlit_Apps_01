################################################################################################################
## Streamlit App 008 - 01 - 01/02 - Model Creation
################################################################################################################
## App 008 - Penguin Classification
################################################################################################################

## Model Creation
## Creating and saving the model that will be used for Streamlit Web App
## Series of models was created on Notebook and the best performing one was chosen

### Model To Create:
##param_grid: {'n_estimators': range(1, 101)}
##get_params: <bound method BaseEstimator.get_params of GridSearchCV(estimator=RandomForestClassifier(),
##             param_grid={'n_estimators': range(1, 101)}, verbose=3)>
##best_estimator_: RandomForestClassifier(n_estimators=34)
##best_params_: {'n_estimators': 34}

### Model with {'n_estimators': 19}

### Models performance is good. Will train it to entire dataset.
################################################################################################################

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('Projects/App_008_001/App_008_001_Exported/Data/penguins_cleaned.csv')

### y = species
target = 'species'
encode = ['sex', 'island']

df02 = df[['sex', 'island']]
##df02
df02 = pd.get_dummies(df02, drop_first=True).astype('int')
df02.head()

df03 = pd.concat([df, df02], axis=1)
df03 = df03.drop(columns=['island','sex'], axis=1)
#df03

df03['species'].unique()
##array(['Adelie', 'Gentoo', 'Chinstrap'], dtype=object)

### Encoding Target 'species'

## Creating how it will be modified
target_mapper = {'Adelie':0,
                 'Chinstrap':1,
                 'Gentoo':2}

def target_encode(val):
    return target_mapper[val]

## Note: It is apply(target_encode) and not apply(target_encode())
df03['species'] = df03['species'].apply(target_encode)

## Verifying if encoder worked correctly
#df03['species'][224:230]
#df03['species'][114:120]

## X and y Features and Class Labels
X = df03.drop(columns=['species'], axis=1)
y = df03['species']

### Train Test Split
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                    train_size=None,
#                                                    random_state=42,
#                                                    shuffle=True,
#                                                    stratify=None)

#X.head(2)

#X.columns
##['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
##       'sex_male', 'island_Dream', 'island_Torgersen']


### Grid Search - Random Forest

### Model - Final model no of estimators
###
model_forest_02 = RandomForestClassifier(n_estimators=19,
                                            criterion='gini',
                                            max_depth=None,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            min_weight_fraction_leaf=0.0,
                                            max_features='auto',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            bootstrap=True,
                                            oob_score=False,
                                            n_jobs=None,
                                            random_state=None,
                                            verbose=0,
                                            warm_start=False,
                                            class_weight=None,
                                            ccp_alpha=0.0,
                                            max_samples=None)

#n_estimators = range(1,101)

#param_grid_model_forest_01_grid = {'n_estimators':n_estimators}

#model_forest_01_grid = GridSearchCV(estimator=model_forest_01_to_grid,
#                                    param_grid=param_grid_model_forest_01_grid,
#                                    scoring=None,
#                                    n_jobs=None,
#                                    refit=True,
#                                    cv=None,## Default None = 5
#                                    verbose=3,
#                                    pre_dispatch='2*n_jobs',
#                                    return_train_score=False)
#

#model_forest_01_grid.fit(X_train, y_train)
model_forest_02.fit(X, y)

### Model Predictions
model_forest_02_preds = model_forest_02.predict(X)

### Model Metrics
accuracy_score_model_forest_02 = accuracy_score(y_true=y, y_pred=model_forest_02_preds, normalize=True, sample_weight=None)
error_rt_score_model_forest_02 = 1 - accuracy_score_model_forest_02

print(classification_report(y_true=y, y_pred=model_forest_02_preds,labels=None,
                            target_names=None,
                            sample_weight=None,
                            digits=2,
                            output_dict=False,
                            zero_division='warn'))

print(confusion_matrix(y_true=y, y_pred=model_forest_02_preds, labels=None, sample_weight=None, normalize=None))

ConfusionMatrixDisplay.from_estimator(estimator=model_forest_02,
                                      X=X, y=y,
                                      labels=None,
                                      sample_weight=None,
                                      normalize=None,
                                      display_labels=None,
                                      include_values=True,
                                      xticks_rotation='horizontal',
                                      values_format=None,
                                      cmap='viridis',
                                      ax=None,
                                      colorbar=True)

plt.figure(figsize=(6,4))
plt.scatter(accuracy_score_model_forest_02, accuracy_score_model_forest_02)
plt.scatter(error_rt_score_model_forest_02, error_rt_score_model_forest_02)
plt.show()

print("\n")

print(classification_report(y_true=y, y_pred=model_forest_02_preds,
                            labels=None,
                            target_names=None,
                            sample_weight=None,
                            digits=2,
                            output_dict=False,
                            zero_division='warn'))

print("\n")

print(confusion_matrix(y_true=y, y_pred=model_forest_02_preds, labels=None, sample_weight=None, normalize=None))

print("\n")

ConfusionMatrixDisplay.from_estimator(estimator=model_forest_02,
                                      X=X, y=y,
                                      labels=None,
                                      sample_weight=None,
                                      normalize=None,
                                      display_labels=None,
                                      include_values=True,
                                      xticks_rotation='horizontal',
                                      values_format=None,
                                      cmap='viridis',
                                      ax=None,
                                      colorbar=True)

print("\n")

ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=model_forest_02_preds,
                                        labels=None,
                                        sample_weight=None,
                                        normalize=None,
                                        display_labels=None,
                                        include_values=True,
                                        xticks_rotation='horizontal',
                                        values_format=None,
                                        cmap='viridis',
                                        ax=None,
                                        colorbar=True,)

print("\n")

fig = plt.figure(figsize=(6,4))
plt.scatter(accuracy_score_model_forest_02, accuracy_score_model_forest_02, label='Accuracy')
plt.scatter(error_rt_score_model_forest_02, error_rt_score_model_forest_02, label='Error Rate')

plt.title("Accuracy, error rate")
plt.legend()

plt.show()

### 97% of accuracy
### f1 score is good for all class labels
### The model had 2 missclassiciations on the True label 1


#X.columns
##Index(['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
##       'sex_male', 'island_Dream', 'island_Torgersen'],
##      dtype='object')


## Saving Model
#import pickle
pickle.dump(model_forest_02, open('Projects/App_008_001/App_008_001_Exported/Data/Saved_Models/model_01.pkl', 'wb'))

## Opening saved model
with open('Projects/App_008_001/App_008_001_Exported/Data/Saved_Models/model_01.pkl', 'rb') as pickle_file:
    model_02_loaded = pickle.load(pickle_file)

### Model Predictions
model_forest_02_loaded_preds = model_02_loaded.predict(X)

### Model Metrics
accuracy_score_model_forest_02_loaded = accuracy_score(y_true=y, y_pred=model_forest_02_loaded_preds, normalize=True, sample_weight=None)
error_rt_score_model_forest_02_loaded = 1 - accuracy_score_model_forest_02_loaded

plt.figure(figsize=(6,4))
plt.scatter(accuracy_score_model_forest_02_loaded, accuracy_score_model_forest_02_loaded)
plt.scatter(error_rt_score_model_forest_02_loaded, error_rt_score_model_forest_02_loaded)
plt.show()

print("\n")

print(classification_report(y_true=y, y_pred=model_forest_02_loaded_preds,
                            labels=None,
                            target_names=None,
                            sample_weight=None,
                            digits=2,
                            output_dict=False,
                            zero_division='warn'))

print("\n")

print(confusion_matrix(y_true=y, y_pred=model_forest_02_loaded_preds, labels=None, sample_weight=None, normalize=None))

print("\n")

ConfusionMatrixDisplay.from_estimator(estimator=model_02_loaded,
                                      X=X, y=y,
                                      labels=None,
                                      sample_weight=None,
                                      normalize=None,
                                      display_labels=None,
                                      include_values=True,
                                      xticks_rotation='horizontal',
                                      values_format=None,
                                      cmap='viridis',
                                      ax=None,
                                      colorbar=True)

print("\n")

ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=model_forest_02_loaded_preds,
                                        labels=None,
                                        sample_weight=None,
                                        normalize=None,
                                        display_labels=None,
                                        include_values=True,
                                        xticks_rotation='horizontal',
                                        values_format=None,
                                        cmap='viridis',
                                        ax=None,
                                        colorbar=True,)

print("\n")

### Metrics for final model model_01.pkl
###              precision    recall  f1-score   support
###
###           0       1.00      1.00      1.00       146
###           1       1.00      1.00      1.00        68
###           2       1.00      1.00      1.00       119
###
###    accuracy                           1.00       333
###   macro avg       1.00      1.00      1.00       333
###weighted avg       1.00      1.00      1.00       333
###
###
###[[146   0   0]
### [  0  68   0]
### [  0   0 119]]


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
         [Data Source: [Dataset Derived from](https://github.com/allisonhorst/palmerpenguins/)\n
         [Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.](https://archive.ics.uci.edu/dataset/53/iris)\n
         [Streamlit](https://streamlit.io/)
         """)

st.write('###### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*')
st.write("##### Thank you kindly to all who make information and knowledge available for free.")