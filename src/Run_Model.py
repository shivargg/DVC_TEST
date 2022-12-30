#!/usr/bin/env python
# coding: utf-8

# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels .tools.tools import add_constant
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor

from tqdm import tqdm
from math import sqrt
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

import sys
import json
import copy
import shap
import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 800)

np.set_printoptions(threshold=sys.maxsize)


# In[16]:


def my_custom_directional_accuracy_func(actual, pred, prev_actual):
    num_increases_matching = 0
    num_decreases_matching = 0
    num_exact_matches = 0
    total_num_records = len(actual)
    actual_direction = actual - prev_actual
    pred_direction = pred - prev_actual
    actual_signs = np.sign(actual_direction)
    predicted_signs = np.sign(pred_direction)
    matches = (actual_signs == predicted_signs).astype(int)
    num_matches = np.sum(matches)
    return ((num_matches / total_num_records) * 100)

def metrics(actual, predicted, prev_actual):
    eval_metrics = {
        'mape': round(np.mean(abs(actual-predicted) / 1 + actual ), 2),
        'median_ape': round(np.median(abs(actual-predicted) / 1 + actual ), 2),
        'mae': round(mean_absolute_error(actual, predicted), 2),
        'mse': round(mean_squared_error(actual, predicted), 2),
        'rsquare': round(r2_score(actual, predicted), 2),
        'directonal_acc': round(my_custom_directional_accuracy_func(actual, predicted, prev_actual), 2)
    }
    eval_metrics['rmse'] = round(sqrt(eval_metrics['mse']), 2)
    return eval_metrics

def generate_model_report(model_name, model, pred_data, y, prev_actual):
    pred_results = model.predict(pred_data).reshape(-1, 1)
    results = metrics(np.ravel(y), np.ravel(pred_results), np.ravel(prev_actual))
    results['model'] = model_name
    return results

def scale_data(X_train, X_test, numeric_cols):
    test_set_passed = X_test is not None
    scaler = MinMaxScaler().fit(X_train[numeric_cols].values)
    X_train_scaled = copy.deepcopy(X_train)
    X_test_scaled = None
    X_train_scaled[numeric_cols] = pd.DataFrame(scaler.transform(X_train[numeric_cols].values), columns=numeric_cols, index=X_train.index).fillna(0)
    if test_set_passed:
        X_test_scaled = copy.deepcopy(X_test)
        X_test_scaled[numeric_cols] = pd.DataFrame(scaler.transform(X_test_scaled[numeric_cols].values), columns=numeric_cols, index=X_test.index).fillna(0)
    return X_train_scaled, X_test_scaled


# In[17]:


# get_ipython().system('pwd')


# In[21]:


filename = '/home/shiv/Github/DVC_TEST/src/../data/API_dataset.pkl'


# In[22]:


model_prep = pickle.load(open(filename, "rb"))


# In[24]:


train = model_prep['train']
test = model_prep['test']
vif_cols_5 = model_prep['vif_cols_5']
numeric_cols = model_prep['numeric_cols']
cat_cols = model_prep['cat_cols']
all_cols_dropped = model_prep['dropped_cols']


# In[25]:


all_future_quarters = np.sort(pd.unique(test['YEAR_QTR_DATE']))


# In[26]:


df = pd.concat([train, test])


# In[32]:


model_type = 'API'


# In[34]:


all_results = {}

for future_quarter in all_future_quarters[-1:]:
    year_qtr = str(pd.to_datetime(future_quarter).year) + '_' + str(pd.to_datetime(future_quarter).quarter)
    all_results[year_qtr] = []
    
    train = df[df['YEAR_QTR_DATE'] < future_quarter]
    test = df[df['YEAR_QTR_DATE'] == future_quarter]
    
    X_train = copy.deepcopy(train)
    y_train = X_train[model_type]
    
    print(year_qtr)
    print("Ranges for train data: ", X_train['YEAR_QTR_DATE'].min(), X_train['YEAR_QTR_DATE'].max())
    X_train.drop(columns = ['AGT_ID', 'YEAR_QTR_DATE', model_type], inplace = True)
    
    X_test = copy.deepcopy(test)
    y_test = X_test[model_type]
    print("Ranges for train data: ", X_test['YEAR_QTR_DATE'].min(), X_test['YEAR_QTR_DATE'].max())
    X_test.drop(columns = ['AGT_ID', 'YEAR_QTR_DATE', model_type], inplace = True)
    
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, numeric_cols)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, criterion = 'absolute_error')
    model.fit(X_train_scaled, y_train)
    previous_quarter_data = X_test[f'{model_type}_Q_MINUS_1']
    results = generate_model_report(model_type, model, X_test_scaled, y_test, previous_quarter_data)
    results['year_qtr'] = year_qtr
    all_results[year_qtr].append(results)


# In[37]:


#with open('/home/shiv/Github/DVC_TEST/experiment_results/results.json', 'a') as f:
#    json.dump(results, f)


# In[ ]:




