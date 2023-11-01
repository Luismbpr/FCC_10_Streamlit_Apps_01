### Working Correctly
################################################################################################################
## Streamlit App 010 - 01 - 01/02 Model Creation
################################################################################################################
## App 010 - Solubility Prediction
################################################################################################################
## Model Creation to be used on web app
################################################################################################################
### Model Creation
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import pickle

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv('Projects/App_010_001/App_010_001_Exported/Data/delaney_solubility_with_descriptors.csv')

df.info()
df.head(3)

## X y
X = df.drop(columns=['logS'], axis=1)

#y = df['logS']
y = df.iloc[:,-1]
#y.head(3)

from sklearn.model_selection import train_test_split

X_train, X_valid_to_be, y_train, y_valid_to_be = train_test_split(X, y,
                                                                  test_size=0.2,
                                                                  train_size=None,
                                                                  random_state=42,
                                                                  shuffle=True,
                                                                  stratify=None)

X_valid, X_test, y_valid, y_test = train_test_split(X_valid_to_be,
                                                    y_valid_to_be,
                                                    test_size=0.5,
                                                    train_size=None,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=None)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#### model object
model_lr_01 = LinearRegression(fit_intercept=True,
                               normalize='deprecated',
                               copy_X=True,
                               n_jobs=None,
                               positive=False)

model_lr_01.fit(X_train, y_train)


### Model Predictions
model_lr_01_preds_valid = model_lr_01.predict(X_valid)

mae_rmse_model_lr_01 = mean_absolute_error(y_true=y_valid, y_pred=model_lr_01_preds_valid,
                                           sample_weight=None, 
                                           multioutput='uniform_average')

mse_model_lr_01 = mean_squared_error(y_true=y_valid, y_pred=model_lr_01_preds_valid,
                                     sample_weight=None,
                                     multioutput='uniform_average',
                                     squared=True)

rmse_model_lr_01 = np.sqrt(mean_squared_error(y_true=y_valid, y_pred=model_lr_01_preds_valid,
                                              sample_weight=None,
                                              multioutput='uniform_average',
                                              squared=True))

r2_model_lr_01 = r2_score(y_true=y_valid, y_pred=model_lr_01_preds_valid,
                          sample_weight=None,
                          multioutput='uniform_average')


### Model Performance
print(f"Model: {str(model_lr_01)}")
print(f"Model Coefficients:\n{model_lr_01.coef_}")
print("\n")
print(f"Model Intercept:\n{model_lr_01.intercept_}")
print("\n")
print(f"MAE: {mae_rmse_model_lr_01 :.2f}")
print(f"MSE: {mse_model_lr_01 :.2f}")
print(f"RMSE: {rmse_model_lr_01 :.2f}")
print(f"Coefficient of Determination R2: {r2_model_lr_01 :.2f}")

#### Result
###Model: LinearRegression()
###Model Coefficients:
###[-0.72521491 -0.00663091  0.00502635 -0.50455953]
###
###Model Intercept:
###0.25272417963154803
###
###MAE: 0.81
###MSE: 1.06
###RMSE: 1.03
###Coefficient of Determination R2: 0.75


## Model Equation
##X.columns
##Index(['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion'], dtype='object')


#print('LogS = %.2f %.2f LogP %.4f MW + %.4f RB %.2f AP' % (model_lr_01.intercept_, model_lr_01.coef_[0], model_lr_01.coef_[1], model_lr_01.coef_[2], model_lr_01.coef_[3]))
print('LogS = %.2f %.2f MolLogP %.4f MolWt + %.4f NumRotatableBonds %.2f AromaticProportion' % (model_lr_01.intercept_, model_lr_01.coef_[0], model_lr_01.coef_[1], model_lr_01.coef_[2], model_lr_01.coef_[3]))
###Result: LogS = 0.25 -0.73 MolLogP -0.0066 MolWt + 0.0050 NumRotatableBonds -0.50 AromaticProportion

### Model Visualization
### Plotting y_true vs y_pred
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,5))

plt.scatter(x=y_valid,
            y=model_lr_01_preds_valid,
            s=None,
            c='#7CAE00',
            marker=None,
            cmap=None,
            norm=None,
            vmin=None,
            vmax=None,
            alpha=None,
            linewidths=None,
            edgecolors=None,
            plotnonfinite=False,
            data=None)

### Adding Trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
#z = np.polyfit(y_valid, model_lr_01_preds_valid, 1)
#p = np.poly1d(z)

#plt.plot(z, p, c='red')

plt.ylabel('Predicted LogS')
plt.xlabel('Experimental LogS')
plt.show()

len(y_valid), len(model_lr_01_preds_valid)
##(114, 114)

np.polynomial.polynomial.Polynomial(coef=model_lr_01.coef_, domain=None, window=None, symbol='x')
###[-0.72521491 -0.00663091  0.00502635 -0.50455953]

z = np.polyfit(y_valid, model_lr_01_preds_valid, 1)
p = np.poly1d(z)
print(z)
print(p)
##[ 0.68691444 -0.82291026]
##0.6869 x - 0.8229

### Saving Model
import pickle
pickle.dump(obj=model_lr_01, file=open('Projects/App_010_001/App_010_001_Exported/Data/Saved_Models/App_010_model_lr_01.pkl', 'wb'), fix_imports=True, buffer_callback=None)


#### Resources:
#### [Data Source John S. Delaney. ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure ***J. Chem. #Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005](https://pubs.acs.org/doi/10.1021/ci034243x)\n
#### [Streamlit](https://streamlit.io/)
#### [Adding Trendline](https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs)
#### *Code based on [Free Code Camp](https://www.freecodecamp.org/). Special Thanks to Free Code Camp and instructor Chanin Nantasenamat*
#### Thank you kindly to all who make information and knowledge available for free.