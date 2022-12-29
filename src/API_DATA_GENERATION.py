#!/usr/bin/env python
# coding: utf-8

# ### Data Generation Notebook
# 1. Load data from Snowflake.
# 2. Performs relevant checks and operations (removing columns with one value, VIF checks, scaling) to get a dataset ready for modelling for the following agent feature - API.
# 3. Stores all relevent data as a pkl object.

# In[41]:


import os
import yaml
print(os.getcwd())

# In[42]:


# Read params from yaml file.
params = yaml.safe_load(open('./params.yaml'))['prepare']

MODEL_TYPE = params['MODEL_TYPE'] # 'API', 'APP_COUNT', 'FYC' or 'PERSISTENCY'
TEST_SET_LENGTH = int(params['TEST_SET_LENGTH']) # In years.


# In[43]:



# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels .tools.tools import add_constant
from pickle import dumps
from datetime import datetime

import pandas as pd
import numpy as np

import sys
import json
import copy
import datetime, time
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 800)
np.set_printoptions(threshold=sys.maxsize)


# In[45]:


import snowflake_acc

with open('./src/uat_creds.json', 'r') as fp:
    params = json.load(fp)

acc = snowflake_acc.create_snowflake_accessor(params, globals())


# In[46]:


today = datetime.datetime.now()


# In[47]:


file_version_date_time_str = today.strftime("%m_%d_%y__%H_%M")


# ### Function Definitions

# In[48]:


def remove_cols_with_one_unique_value(data):
    cols_to_drop = []
    all_cols = list(data.columns)
    for col in all_cols:
        if np.shape(data[col].unique())[0] == 1:
            cols_to_drop.append(col)
    data.drop(columns = cols_to_drop, inplace = True)
    print("Columns dropped: ", cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns from the dataframe.")
    return cols_to_drop


# In[49]:


def create_training_list(target_value_to_predict):
    return [f'{target_value_to_predict}_Q_MINUS_{i}' for i in range(1, 5)]


def generate_train_test(data_orig, predictor_col):
    data = copy.deepcopy(data_orig)
    latest_quarter = pd.to_datetime(data['YEAR_QTR_DATE'].max())
    # Remove the most current quarter data.
    data = data[data['YEAR_QTR_DATE'] != latest_quarter]
    # Construct the time point where we start splitting our data into train and test.
    split_point = latest_quarter - pd.DateOffset(years= TEST_SET_LENGTH)
    # Build X-Train and y-train.
    train = data[data['YEAR_QTR_DATE'] < split_point]
#     y_train = X_train[predictor_col]
#     X_train.drop(columns = [predictor_col], inplace = True)
    # Build X-Test and y-test.
    test = data[(data['YEAR_QTR_DATE'] >= split_point)]
#     y_test = X_test[predictor_col]
#     X_test.drop(columns = [predictor_col], inplace = True)
    return train, test


def split_df_train(data, predictor_col):
    qtr_to_split = pd.to_datetime(data['YEAR_QTR_DATE'].max()) - pd.DateOffset(months=3)
    # Build X-Train and y-train.
    X_train = data[data['YEAR_QTR_DATE'] < qtr_to_split]
    y_train = X_train[predictor_col]
    X_train.drop(columns = [predictor_col], inplace = True)
    # Build X-Test and y-test.
    X_test = data[data['YEAR_QTR_DATE'] >= qtr_to_split]
    y_test = X_test[predictor_col]
    X_test.drop(columns = [predictor_col], inplace = True)
    return X_train, X_test, y_train, y_test


def min_max_scale_X(data_train, data_test, cols_to_scale, test_set_passed):
    scaler_train = MinMaxScaler().fit(data_train[cols_to_scale].values)
    data_train[cols_to_scale] = pd.DataFrame(scaler_train.transform(data_train[cols_to_scale].values), columns=cols_to_scale, index=data_train.index).fillna(0)
    if test_set_passed:
        data_test[cols_to_scale] = pd.DataFrame(scaler_train.transform(data_test[cols_to_scale].values), columns=cols_to_scale, index=data_test.index).fillna(0)
    return scaler_train


def min_max_scale_y(data_train, data_test, test_set_passed):
    scaler_train = MinMaxScaler().fit(data_train.values.reshape(-1, 1))
    data_train = pd.DataFrame(scaler_train.transform(data_train.values.reshape(-1, 1)), index=data_train.index).fillna(0)
    if test_set_passed:
        data_test = pd.DataFrame(scaler_train.transform(data_test.values.reshape(-1, 1)), index=data_test.index).fillna(0)
    return scaler_train
    

def scale_data(X_train, X_test, y_train, y_test, numeric_cols):
    test_set_passed = X_test is not None and y_test is not None
    if test_set_passed:
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = X_train.copy(deep = True), X_test.copy(deep = True), y_train.copy(deep = True), y_test.copy(deep = True)
        scaler_X_train = min_max_scale_X(X_train_scaled, X_test_scaled, numeric_cols, test_set_passed)
#         scaler_y_train = min_max_scale_y(y_train_scaled, y_test_scaled, test_set_passed)
        return scaler_X_train, None, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
    else:
        X_scaled, y_scaled = X_train.copy(deep = True), y_train.copy(deep = True)
        scaler_X_train = min_max_scale_X(X_scaled, None, numeric_cols, test_set_passed)
#         scaler_y_train = min_max_scale_y(y_scaled, None, test_set_passed)
        return scaler_X_train, None, X_scaled, y_scaled


def calc_max_vif(data):
    vif = pd.DataFrame()
    all_cols = data.columns
    vif_values = [variance_inflation_factor(data.values, i) for i in range(len(all_cols))]
    vif = pd.DataFrame({'column': all_cols, 'vif': vif_values})
    max_vif_col = vif.loc[vif['vif'].idxmax()]
    return max_vif_col


def find_cols_to_remove_vif(X_train, numeric_cols):
    vif_cols_dropped = []
    cols_not_to_drop = ['Quarter__1', 'Quarter__2', 'Quarter__3', 'Quarter__4', 'AGT_F_CLI_COUNT', 'AGT_M_CLI_COUNT', 'NUM_ACTIVE_POLICIES']
    X_train_VIF = X_train.drop(columns = cols_not_to_drop)
    numeric_cols_clean = copy.deepcopy(numeric_cols)
    for el in cols_not_to_drop:
        numeric_cols_clean.remove(el)
    while True:
        highest_vif = calc_max_vif(X_train_VIF[numeric_cols_clean])
        if highest_vif['vif'] > 5.0:
            print(highest_vif)
            col_to_remove = highest_vif['column']
            X_train_VIF.drop(columns = [col_to_remove], inplace = True)
            numeric_cols_clean.remove(col_to_remove)
            vif_cols_dropped.append(col_to_remove)
            print("\n")
        else:
            break
    return vif_cols_dropped


# In[50]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold


def calculate_vif(X, thresh=100, verbose=False):
    cols_to_drop = []
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped = True
    while dropped:
        dropped = False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            cols_to_drop.append(X[cols[variables]].columns[maxloc])
            if verbose:
                print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped = True
    if verbose:
        print('Remaining variables:')
        print(X.columns[variables])
    return cols_to_drop #X[cols[variables]]


def variance_threshold_selector(data, threshold=0.5):
    # https://stackoverflow.com/a/39813304/1956309
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def remove_correlated_features(df: pd.DataFrame, inplace=False):
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    if inplace:
        df.drop(to_drop, axis=1, inplace=True)
        return df, to_drop

    df1 = df.drop(to_drop, axis=1, inplace=False)
    return df1, to_drop


# In[51]:


def get_quarter_to_predict(df, quarters_to_lookback):
    all_quarters = df['YEAR_QTR_DATE'].unique()
    all_quarters.sort()
    quarter_to_predict = all_quarters[-1 * quarters_to_lookback]
    return quarter_to_predict


def extract_year_qtr(df, quarter_to_predict):
    return df.query(f"YEAR_QTR_DATE < '{pd.to_datetime(quarter_to_predict)}'")


def build_dataset_final(X, y, agt_ids, quarters_to_lookback):
    train = copy.deepcopy(X)
    train['y'] = y
    train['AGT_ID'] = agt_ids
    train.reset_index(inplace = True, drop = True)
    quarter_to_predict = get_quarter_to_predict(X, 1)
    print("QUARTER TO PREDICT -> ", quarter_to_predict)
    train_filtered = extract_year_qtr(train, quarter_to_predict)
    year_qtr_max = train_filtered['YEAR_QTR_DATE'].max()
    X_train = train_filtered.drop(columns = ['YEAR_QTR_DATE', 'y', 'AGT_ID'])
    y_train = train_filtered['y']
    final = train.query(f"YEAR_QTR_DATE >= '{quarter_to_predict}'")
    X_final = final.drop(columns = ['YEAR_QTR_DATE', 'y'])
    print(X_final.shape)
    return X_train, y_train, X_final, quarter_to_predict


def get_current_year_quarter():
    current_time = datetime.datetime.now()
    month = current_time.month
    return f'{current_time.year}_{math.floor(month/3) + 1}'


# ### Track Columns Dropped

# In[52]:


all_cols_dropped = []
all_cat_cols = []
all_numeric_cols = []


# ### Load Dataset

# In[53]:


acc.switch_schema('05_MODEL_INPUT_TOP_AGENT')


# In[54]:


df_lagged = acc.retrieve_query_res(f'SELECT * FROM DS_GLOC_DEV_DB."05_MODEL_INPUT_TOP_AGENT"."05_AGENT_MASTER_TABLE_CLEANED"')


# In[55]:


df_score = acc.retrieve_query_res(f'SELECT * FROM DS_GLOC_DEV_DB."05_MODEL_INPUT_TOP_AGENT"."05_AGENT_MASTER_TABLE_CLEANED_SCORING"')


# In[56]:


now = datetime.datetime.fromtimestamp(time.time())
curr_quarter = (now.month-1)//3+1
curr_year = now.year

data_quarter = int(df_lagged['YEAR_QUARTER'].max().split('_')[-1])
data_quarter_date = df_lagged['YEAR_QUARTER'].max()


# In[57]:


data_quarter


# In[58]:


df_lagged['YEAR_QUARTER'].min()


# In[59]:


data_quarter_date


# In[60]:


df_lagged[df_lagged['YEAR_QUARTER'] == data_quarter_date].shape


# In[61]:


# print("Removing last year quarter data from df_lagged.")
# df_lagged = df_lagged[df_lagged['YEAR_QUARTER'] != data_quarter_date]


# In[62]:


# Add the last quarter from the scoring set.
print("Adding the last year quarter from df_scoring.")
df_score = df_score[df_score['YEAR_QUARTER'] == data_quarter_date]
df_score['YEAR_QUARTER'] = f'{curr_year}_{curr_quarter}'
df_score['QUARTER'] = curr_quarter


# In[63]:


df_score.shape


# In[64]:


df_lagged['YEAR_QUARTER'].max()


# In[65]:


df_lagged['YEAR_QUARTER'].min()


# In[66]:


df_score['YEAR_QUARTER'].min()


# In[67]:


df_score['YEAR_QUARTER'].max()


# In[68]:


df_score.shape


# In[69]:


df_lagged.shape


# In[70]:


df = pd.concat([df_lagged, df_score], ignore_index = True, join = 'inner', verify_integrity = True)


# In[71]:


df.shape


# In[72]:


df.info(verbose = True, show_counts = True)


# In[73]:


cat_cols = ['AGT_SEX_F', 'AGT_SEX_M',
            'CLI_OCCP_CLAS_1', 'CLI_OCCP_CLAS_2',
            'AGT_MARIT_STAT_S', 'AGT_MARIT_STAT_C', 'AGT_MARIT_STAT_W', 'AGT_MARIT_STAT_M', 'AGT_MARIT_STAT_D', 'AGT_MARIT_STAT_P',
            'AGT_REGION_TUNAPUNAPIARCO', 'AGT_REGION_TOBAGO', 'AGT_REGION_SIPARIA', 'AGT_REGION_SAN_JUAN_LAVENTILLE', 'AGT_REGION_SAN_FERNANDO',
            'AGT_REGION_SANGRE_GRANDE', 'AGT_REGION_PRINCES_TOWN', 'AGT_REGION_PORT_OF_SPAIN', 'AGT_REGION_POINT_FORTIN', 'AGT_REGION_PENALDEBE',
            'AGT_REGION_OTHER', 'AGT_REGION_MISSING', 'AGT_REGION_MAYARORIO_CLARO', 'AGT_REGION_DIEGO_MARTIN', 'AGT_REGION_COUVATABAQUITETALPARO',
            'AGT_REGION_CHAGUANAS', 'AGT_REGION_ARIMA',
            'AGT_PART_TIME_FLAG'
]

cols_to_drop = ['AGT_SEX_C', 
                'CLI_OCCP_CLAS_MISSING', 
                'AGT_MARIT_STAT_MISSING', 
                'DATE_OF_BIRTH',
                'AGT_FULL_NAME', 'AGT_FULL_NAME_NON_NULL', 'AGT_GIV_NM', 'AGT_SUR_NM', 
                'YEAR', 'MONTH', 'YEAR_QUARTER', 
                'AGT_AGE', 'AGT_BRANCH'
]

cols_to_drop_fluid = [
                'CNTRCT_TRMN_DT_TXT', 'AGT_STAT_CD', 'START_MONTH', 'START_YEAR',
                'FEEDBACK', 'MAX(CFCVG.POL_ID)', 'MAX(CFCVG.CVG_NUM)',
#                 'AGT_M_CLI_COUNT', 'AGT_F_CLI_COUNT', 'AGT_O_CLI_COUNT', 
                'AGT_TOTAL_CLI_INCM_EARNED', 'AGT_CLI_AVG_EMPL_AGE', 'AGT_CLI_OCCP_CLAS_CD_COUNT', 'AGT_CLI_MARIT_STAT_CD_COUNT',
                'CVG_FACE_AMT', 'CVG_ORIG_FACE_AMT', 'CVG_PREV_FACE_AMT', 'CVG_UWG_AMT', 'CVG_UNIT_VALU_AMT', 'CVG_SUM_INS_AMT', 'CVG_AD_FACE_AMT', 'CVG_MPREM_AMT', 'CVG_PFEE_AMT',
                'CVG_BASIC_PREM_AMT', 'CVG_AD_PREM_AMT', 'CVG_WP_PREM_AMT', 'REDC_EP_PREM_AMT', 'OWN_OCCP_PREM_AMT', 'CVG_LTA_PREM_AMT', 'CVG_LTB_PREM_AMT', 'PDISAB_PREM_AMT', 'CVG_COLA_PREM_AMT',
                'CVG_FE_UPREM_AMT', 'CVG_FE_PREM_AMT', 'CVG_ME_PREM_AMT', 'CVG_SALE_TAX_AMT', 'PREV_WP_UPREM_AMT', 'CVG_PREV_UPREM_AMT', 'CVG_NXT_UPREM_AMT', 'CVG_MDRT_AMT', 'CVG_FYR_COMM_AMT',
                'CVG_CLM_YTD_AMT', 'CVG_CLM_LTD_AMT', 'CVG_CLM_CHQ_AMT', 'CVG_WP_YTD_AMT', 'CVG_WP_LTD_AMT', 'CVG_NET_REISS_AMT', 
                'IN_ALLOC_AMT_PCT', 'OUT_ALLOC_AMT_PCT', 'CVG_MAX_COMIT_AMT',
                'CVG_COMM_TRG_AMT', 'PMT_LOAD_TRG_AMT', 'PMT_LOAD_LTD_AMT', 'CVG_PMT_LTD_AMT', 'MNPMT_TRG_LTD_AMT', 'CVG_SURR_TRG_AMT', 'SURR_LOAD_LTD_AMT', 'CVG_SURR_LTD_AMT',
                'CVG_GDLN_APREM_AMT', 'CVG_GDLN_SPREM_AMT', 'CVG_LOAN_CLR_1_AMT', 'CVG_LOAN_CLR_2_AMT', 'CVG_APL_CLR_AMT', 'GIR_OPT_REMN_AMT', 'CVG_FE2_UPREM_AMT', '2018_TOTAL_BONUS_RATE',
                '2018_TOTAL_ADD_BONUS_RATE', '2019_TOTAL_BONUS_RATE', '2019_TOTAL_ADD_BONUS_RATE', '2020_TOTAL_BONUS_RATE', '2020_TOTAL_ADD_BONUS_RATE', '2021_TOTAL_BONUS_RATE', '2021_TOTAL_ADD_BONUS_RATE',
                '2018_FYR_COMM_CMO_AMT', '2018_FYR_COMM_YTD_AMT', '2018_RENW_COMM_CMO_AMT', '2018_RENW_COMM_YTD_AMT', '2018_OVRID_COMM_CMO_AMT', '2018_OVRID_COMM_YTD_AMT', '2018_AGT_PAYO_CMO_AMT',
                '2018_AGT_PAYO_YTD_AMT', '2018_FYR_LCOMM_CMO_AMT', '2018_FYR_LCOMM_YTD_AMT', '2018_RENW_LCOMM_CMO_AMT', '2018_RENW_LCOMM_YTD_AMT', '2018_QLTY_BON_CMO_AMT', '2018_QLTY_BON_YTD_AMT',
                '2018_MKT_BON_CMO_AMT', '2018_MKT_BON_YTD_AMT', '2018_NHS_ALLOW_CMO_AMT', '2018_NHS_ALLOW_YTD_AMT', '2018_AGT_APP_PMO_QTY', '2018_AGT_APP_YTM_QTY', '2018_AGT_PWRIT_PMO_AMT',
                '2018_AGT_PWRIT_YTM_AMT', '2018_LIFE_PWRIT_PMO_AMT', '2018_LIFE_PWRIT_YTM_AMT', '2018_FYR_CPREM_CMO_AMT', '2018_FYR_CPREM_YTD_AMT', '2018_RENW_CPREM_CMO_AMT',
                '2018_RENW_CPREM_YTD_AMT', '2018_TOTAL_FYR_COMM', '2018_TOTAL_RENW_COMM', '2019_FYR_COMM_CMO_AMT', '2019_FYR_COMM_YTD_AMT', '2019_RENW_COMM_CMO_AMT', '2019_RENW_COMM_YTD_AMT',
                '2019_OVRID_COMM_CMO_AMT', '2019_OVRID_COMM_YTD_AMT', '2019_AGT_PAYO_CMO_AMT', '2019_AGT_PAYO_YTD_AMT', '2019_FYR_LCOMM_CMO_AMT', '2019_FYR_LCOMM_YTD_AMT', '2019_RENW_LCOMM_CMO_AMT',
                '2019_RENW_LCOMM_YTD_AMT', '2019_QLTY_BON_CMO_AMT', '2019_QLTY_BON_YTD_AMT', '2019_MKT_BON_CMO_AMT', '2019_MKT_BON_YTD_AMT', '2019_NHS_ALLOW_CMO_AMT', '2019_NHS_ALLOW_YTD_AMT',
                '2019_AGT_APP_PMO_QTY', '2019_AGT_APP_YTM_QTY', '2019_AGT_PWRIT_PMO_AMT', '2019_AGT_PWRIT_YTM_AMT', '2019_LIFE_PWRIT_PMO_AMT', '2019_LIFE_PWRIT_YTM_AMT', '2019_FYR_CPREM_CMO_AMT',
                '2019_FYR_CPREM_YTD_AMT', '2019_RENW_CPREM_CMO_AMT', '2019_RENW_CPREM_YTD_AMT', '2019_TOTAL_FYR_COMM', '2019_TOTAL_RENW_COMM', '2020_FYR_COMM_CMO_AMT', '2020_FYR_COMM_YTD_AMT', '2020_RENW_COMM_CMO_AMT',
                '2020_RENW_COMM_YTD_AMT', '2020_OVRID_COMM_CMO_AMT', '2020_OVRID_COMM_YTD_AMT', '2020_AGT_PAYO_CMO_AMT', '2020_AGT_PAYO_YTD_AMT', '2020_FYR_LCOMM_CMO_AMT', '2020_FYR_LCOMM_YTD_AMT',
                '2020_RENW_LCOMM_CMO_AMT', '2020_RENW_LCOMM_YTD_AMT', '2020_QLTY_BON_CMO_AMT', '2020_QLTY_BON_YTD_AMT', '2020_MKT_BON_CMO_AMT', '2020_MKT_BON_YTD_AMT', '2020_NHS_ALLOW_CMO_AMT',
                '2020_NHS_ALLOW_YTD_AMT', '2020_AGT_APP_PMO_QTY', '2020_AGT_APP_YTM_QTY', '2020_AGT_PWRIT_PMO_AMT', '2020_AGT_PWRIT_YTM_AMT', '2020_LIFE_PWRIT_PMO_AMT', '2020_LIFE_PWRIT_YTM_AMT',
                '2020_FYR_CPREM_CMO_AMT', '2020_FYR_CPREM_YTD_AMT', '2020_RENW_CPREM_CMO_AMT', '2020_RENW_CPREM_YTD_AMT', '2020_TOTAL_FYR_COMM', '2020_TOTAL_RENW_COMM', '2021_FYR_COMM_CMO_AMT',
                '2021_FYR_COMM_YTD_AMT', '2021_RENW_COMM_CMO_AMT', '2021_RENW_COMM_YTD_AMT', '2021_OVRID_COMM_CMO_AMT', '2021_OVRID_COMM_YTD_AMT', '2021_AGT_PAYO_CMO_AMT', '2021_AGT_PAYO_YTD_AMT',
                '2021_FYR_LCOMM_CMO_AMT', '2021_FYR_LCOMM_YTD_AMT', '2021_RENW_LCOMM_CMO_AMT', '2021_RENW_LCOMM_YTD_AMT', '2021_QLTY_BON_CMO_AMT', '2021_QLTY_BON_YTD_AMT', '2021_MKT_BON_CMO_AMT', '2021_MKT_BON_YTD_AMT',
                '2021_NHS_ALLOW_CMO_AMT', '2021_NHS_ALLOW_YTD_AMT', '2021_AGT_APP_PMO_QTY', '2021_AGT_APP_YTM_QTY', '2021_AGT_PWRIT_PMO_AMT', '2021_AGT_PWRIT_YTM_AMT', '2021_LIFE_PWRIT_PMO_AMT', '2021_LIFE_PWRIT_YTM_AMT',
                '2021_FYR_CPREM_CMO_AMT', '2021_FYR_CPREM_YTD_AMT', '2021_RENW_CPREM_CMO_AMT', '2021_RENW_CPREM_YTD_AMT', '2021_TOTAL_FYR_COMM', '2021_TOTAL_RENW_COMM', '3','7', '8',
                '9', 'C', 'D', 'F', 'M', 'N', 'V'
]


# In[74]:


# cols_to_drop_clean = [el for el in cols_to_drop_fluid if 'CVG' not in el]
cols_to_drop_clean = []


# ### Operations to Prepare Data:
# 1. Cast columns to numeric form. ()
# 2. Extract quarter. ()
# 3. Dropping cols (multiple times). ()
# 4. Drop rows with missing data. ()
# 5. Build list of numeric columns. ()
# 6. Drop predictor variables. ()
# 7. Split dataset into train and prediction sets. ()
# 8. Split train dataset into train and test segments. ()
# 9. Apply scaling to split data. ()
# 10. VIF removal. ()
# 11. Drop agent feature from datasets. ()
# 12. Save as pickle. ()

# In[75]:


# For validation/testing - remove the current year_quarter value.
# For production, include the current year_quarter_value.

def prepare_data_walk_forward(df_original, selected_predictor):
    # Copy original dataframe.
    df = copy.deepcopy(df_original)
    
    # Attempt to cast all columns to numeric form.
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='ignore')
    
    # Applying transformations to extract the quarter.
    df['YEAR'] = df['YEAR_QUARTER'].str.split('_').str.get(0)
    df['MONTH'] = df['YEAR_QUARTER'].str.split('_').str.get(1).map({'1': '1', '2': '4', '3': '7', '4': '10'})
    df['YEAR_QTR_DATE'] = pd.to_datetime(df['YEAR'] + '/' + df['MONTH'] + '/01', format = '%Y/%m/%d')
    df['QUARTER'] = df['YEAR_QTR_DATE'].dt.quarter
    quarter_dummies = pd.get_dummies(df['QUARTER'], prefix='Quarter_')
    df = pd.merge(
        left = df,
        right = quarter_dummies,
        left_index=True,
        right_index=True,
    )
    df.drop(columns = ['QUARTER'], inplace = True)
    
    # Drop initial set of columns.
    df.drop(columns = cols_to_drop, inplace = True)
    all_cols_dropped.append(cols_to_drop)
    
    # Drop columns with one value.
    cols_dropped = remove_cols_with_one_unique_value(df)
    all_cols_dropped.append(cols_dropped)
    
    # Drop additional columns.
    additional_cols_to_consider_removing = list(set(list(df.columns)).intersection(set(cols_to_drop_clean)))
    df.drop(columns = additional_cols_to_consider_removing, inplace = True)
    all_cols_dropped.append(additional_cols_to_consider_removing)
    
    # Drop any rows with missing data.
    df.dropna(axis=0, how='any', inplace = True)
    
    # Build the list of numeric columns.
    numeric_cols = set(df.columns).difference(set(cat_cols))
    numeric_cols.remove('AGT_ID')
    numeric_cols.remove('YEAR_QTR_DATE')
    numeric_cols = list(numeric_cols)
    
    # Drop predictor variables from dataset except the selected_predictor.
    all_predictors = ['API', 'PERSISTENCY', 'APP_COUNT', 'FYC']
    for predictor in all_predictors:
        numeric_cols.remove(predictor)
    all_predictors.remove(MODEL_TYPE)
    df.drop(columns = all_predictors, inplace = True)
    all_cols_dropped.append(all_predictors)
    
    # We want to create the initial train and test datasets.
    train, test = generate_train_test(df, selected_predictor)
    
    # Run VIF for training data.
    # vif_cols_5 = calculate_vif(train.drop(columns = ['AGT_ID', 'YEAR_QTR_DATE', selected_predictor]), 5)
    vif_cols_5 = None
    
    dataset = {
        'train': train,
        'test': test,
        'vif_cols_5': vif_cols_5,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols,
        'dropped_cols': all_cols_dropped,
        'prod': df
    }
    
    print("Ranges for train data: ", train['YEAR_QTR_DATE'].min(), train['YEAR_QTR_DATE'].max())
    print("Ranges for test data: ", train['YEAR_QTR_DATE'].min(), train['YEAR_QTR_DATE'].max())
    return dataset


# ### Build Pickle Files / Create Model

# In[76]:


import pickle

print(f"Building dataset for {MODEL_TYPE}.")
model_prep = prepare_data_walk_forward(df, MODEL_TYPE)
with open(f'./data/{MODEL_TYPE}_dataset.pkl', 'wb') as handle:
    pickle.dump(model_prep, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("\n")

