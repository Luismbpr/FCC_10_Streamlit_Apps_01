### Creation of several models and analysis of metrics on Notebook 'App_009_001_001_01 Model Creation Functions_clean.ipynb'
### Creation of best performing model
### Best estimator: RandomForestRegressor(n_estimators=79, random_state=42)

### Working Correctly
################################################################################################################
## Streamlit App 009 - 01 - 01/02 Model Creation
################################################################################################################
## App 008 - California Housing Regression
################################################################################################################
## Streamlit Web App
## Creating the Streamlit Web App using the model created on prior python file
################################################################################################################

import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve, RocCurveDisplay, plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


### Data Input and Processing
#import numpy as np
#import pandas as pd

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

df = pd.read_csv('Projects/App_009_001/App_009_001_Exported/Data/housing.csv')

df.head()
#df.isna().sum()

## Replacing missing values
df['total_bedrooms'] = df['total_bedrooms'].replace({np.nan:df['total_bedrooms'].median()})
#df.isna().sum()

### X and y
X = df.drop(columns=['median_house_value'], axis=1)
y = df['median_house_value']

dummies_ocean_01 = pd.get_dummies(data=X['ocean_proximity'], prefix=None, prefix_sep = '_', dummy_na = False, columns = None, sparse = False, drop_first = False, dtype = None).astype('int')

X.head()

features_float_01 = X.drop(columns=['ocean_proximity'])
features_float_01

### Feature Scaling
#import scipy.stats as stats
features_float_01 = X.select_dtypes('float').apply(lambda x: stats.zscore(a=x, axis=0, ddof=0, nan_policy='propagate'))

features_float_01.aggregate(['mean','std'])
X01 = pd.concat(objs=[features_float_01, dummies_ocean_01], axis = 1, join = 'outer', ignore_index = False, keys = None, levels = None, names = None, verify_integrity = False, sort = False, copy = None)
X01
X01.aggregate(['mean','std'])
X01.head(2)


### Train Test Split
#from sklearn.model_selection import train_test_split, GridSearchCV

### 80%-20%
### Using X02 (Z-score) and regular y
X_train, X_valid_to_be, y_train, y_valid_to_be = train_test_split(X01, y, 
                                                                  test_size=0.8,
                                                                  train_size=None,
                                                                  random_state=42,
                                                                  shuffle=True,
                                                                  stratify=None)

### 50%-50%
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_to_be, y_valid_to_be, 
                                                    test_size=0.5,
                                                    train_size=None,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=None)


### Creating a Function To Perform Regression
#import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve, RocCurveDisplay, plot_precision_recall_curve, plot_roc_curve
#import pickle

trained_model_info_ls = {}
mae_scores_list_ls = []
rmse_scores_list_ls = []
r2_scores_list_ls = []


def model_to_train_regression(model_std_name = 'model_std_name',
                   model_grid_name = 'model_grid_name',
                   param_grid_name = 'param_grid_name',
                   rand_state = 42,
                   using_grid=False,
                   X_to_train = X_train, 
                   y_to_train = y_train, 
                   X_to_predict = X_valid,
                   X_true_pred = X_valid,
                   y_true_pred = y_valid,
                   regression = True,
                   save_model = False,
                   path_and_name_to_save = None):
    """
    This Function receives parameters to fit a ML Algorithm and outputs a fitted model withs its metrics.
    It can also output a pickle file of the fitted model.
    """
    
    trained_model_info = {}
    mae_scores_list = []
    rmse_scores_list = []
    r2_scores_list = []
    
    #import numpy as np
    #from sklearn.ensemble import RandomForestRegressor
    #from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve, RocCurveDisplay, plot_precision_recall_curve, plot_roc_curve
    #from sklearn.model_selection import train_test_split, GridSearchCV

    ## Input model to be fitted on or be used on Grid Search
    model_std_name = RandomForestRegressor(n_estimators=100,
                                           criterion='squared_error',
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
                                           random_state=rand_state,## Def None
                                           verbose=0,
                                           warm_start=False,
                                           ccp_alpha=0.0,
                                           max_samples=None)
    
    if using_grid == True:
        #from sklearn.model_selection import GridSearchCV
        n_estimators = np.arange(1, 101, 1)
        param_grid_name = {'n_estimators':n_estimators}

        model_grid_name = GridSearchCV(estimator=model_std_name,
                                       param_grid=param_grid_name,
                                       scoring=None,
                                       n_jobs=None,
                                       refit=True,
                                       cv=None,
                                       verbose=3,
                                       pre_dispatch='2*n_jobs',
                                       return_train_score=False)
        
        
        model_grid_name.fit(X_to_train, y_to_train)

        model_best_params = model_grid_name.best_params_
        model_best_estimator = model_grid_name.best_estimator_
        
    elif using_grid == False:
        model_std_name.fit(X_to_train, y_to_train)
        
    if regression == True:
        #from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        #import numpy as np
        
        if using_grid == True:
            model_preds = model_grid_name.predict(X_true_pred)
            
        elif using_grid == False:
            model_preds = model_std_name.predict(X_true_pred)

        ##
        mae_model_metric = mean_absolute_error(y_true=y_true_pred, y_pred=model_preds, sample_weight=None, multioutput='uniform_average')
        rmse_model_metric = np.sqrt(mean_squared_error(y_true=y_true_pred, y_pred=model_preds, sample_weight=None, multioutput='uniform_average', squared=True))
        r2_model_metric = r2_score(y_true = y_true_pred, y_pred = model_preds, sample_weight=None, multioutput='uniform_average')

        mae_scores_list.append(mae_model_metric)
        rmse_scores_list.append(rmse_model_metric)
        r2_scores_list.append(r2_model_metric)

        mae_scores_list_ls.append(mae_model_metric)
        rmse_scores_list_ls.append(rmse_model_metric)
        r2_scores_list_ls.append(r2_model_metric)
        
        if using_grid == True:
            print(f"Model Name:")
            print(f"Using Grid Search: {using_grid}")
            print(f"Model Name with grid: {model_grid_name}")
            print(f"Best params: {model_best_params}")
            print(f"Best estimator: {model_best_estimator}")
            print("\n")
            print(f"Model Metrics")
            print(f"y_true mean: {np.mean(y_true_pred)}")
            print(f"y_true mean: {np.median(y_true_pred)}")
            print(f"MAE: {mae_model_metric}")
            print(f"RMSE: {rmse_model_metric}")
            print(f"R Squared: {r2_model_metric}")
            print("\n")
            print(f"RMSE is different than the y_true mean by: {(rmse_model_metric * 100)/(y_true_pred)} %")
                
        elif using_grid == False:
            print(f"Model Name:")
            print(f"Using Grid Search: {using_grid}")
            print(f"Model Name with grid: {model_grid_name}")
            print("\n")
            print(f"Model Metrics")
            print(f"y_true mean: {np.mean(y_true_pred)}")
            print(f"y_true mean: {np.median(y_true_pred)}")
            print(f"MAE: {mae_model_metric}")
            print(f"RMSE: {rmse_model_metric}")
            print(f"R Squared: {r2_model_metric}")
            print("\n")
            print(f"RMSE is different than the y_true mean by: {np.round((rmse_model_metric * 100)/(np.mean(y_true_pred)), 2)} %")

    ### Saving Model
    #import pickle
    if save_model == True:
        #import pickle
        if using_grid == True:
            pickle.dump(model_grid_name, open(path_and_name_to_save, 'wb'))
        
        elif using_grid == False:
            pickle.dump(model_std_name, open(path_and_name_to_save, 'wb'))



#### Perform Regression with Function
### Data Input and Processing
#import numpy as np
#import pandas as pd

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

df = pd.read_csv('Projects/App_009_001/App_009_001_Exported/Data/housing.csv')

df.head()
#df.isna().sum()

## Replacing missing values
df['total_bedrooms'] = df['total_bedrooms'].replace({np.nan:df['total_bedrooms'].median()})
#df.isna().sum()

### X and y
X = df.drop(columns=['median_house_value'], axis=1)
y = df['median_house_value']

dummies_ocean_01 = pd.get_dummies(data=X['ocean_proximity'], prefix=None, prefix_sep = '_', dummy_na = False, columns = None, sparse = False, drop_first = False, dtype = None).astype('int')


X.head()

features_float_01 = X.drop(columns=['ocean_proximity'])
features_float_01

### Feature Scaling
#import scipy.stats as stats
features_float_01 = X.select_dtypes('float').apply(lambda x: stats.zscore(a=x, axis=0, ddof=0, nan_policy='propagate'))

features_float_01.aggregate(['mean','std'])
X01 = pd.concat(objs=[features_float_01, dummies_ocean_01], axis = 1, join = 'outer', ignore_index = False, keys = None, levels = None, names = None, verify_integrity = False, sort = False, copy = None)
X01
X01.aggregate(['mean','std'])
X01.head(2)


### Train Test Split
#from sklearn.model_selection import train_test_split, GridSearchCV

### 80%-20% - Using X02 (Z-score) and regular y
X_train, X_valid_to_be, y_train, y_valid_to_be = train_test_split(X01, y, 
                                                                  test_size=0.8,
                                                                  train_size=None,
                                                                  random_state=42,
                                                                  shuffle=True,
                                                                  stratify=None)

### 50%-50%
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_to_be, y_valid_to_be, 
                                                    test_size=0.5,
                                                    train_size=None,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=None)


#import numpy as np
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve, RocCurveDisplay, plot_precision_recall_curve, plot_roc_curve

trained_model_info_ls = {}
mae_scores_list_ls = []
rmse_scores_list_ls = []
r2_scores_list_ls = []

trained_model_info = {}
mae_scores_list = []
rmse_scores_list = []
r2_scores_list = []


model_to_train_regression(model_std_name = 'model_forest_02_to_grid',
                            model_grid_name = 'model_forest_02_nogrid',
                            param_grid_name = 'param_grid_noname',
                            rand_state = 42,
                            using_grid=False,
                            X_to_train = X_train, 
                            y_to_train = y_train, 
                            X_to_predict = X_valid,
                            X_true_pred = X_valid,
                            y_true_pred = y_valid,
                            regression = True,
                            save_model = True,
                            #save_model_name = 'saved_model01.pkl',
                            path_and_name_to_save = 'Projects/App_009_001/App_009_001_Exported/Data/Saved_Models/App_009_model_Regression.pkl')


#trained_model_info_ls
print(np.round(mae_scores_list_ls,4))
print(np.round(rmse_scores_list_ls,4))
print(np.round(r2_scores_list_ls,4))


