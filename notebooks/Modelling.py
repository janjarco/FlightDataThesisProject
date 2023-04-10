# %% [markdown]
# ### Import libraries

# %%
import os
import pandas as pd
import numpy as np



#set working directory to this from the workspace
os.chdir('/Users/janjarco/Programming/PrivateRepository/FlightDataThesisProject')

#list all the files in current working directory
print(os.listdir("data"))

# %%
import time
start_time = time.time()
df = pd.read_csv("data/processed/clicks_orders_merge_currency.csv", low_memory=True, parse_dates=['clicks_created_at_datetime', 'orders_created_at_datetime'])
end_time = time.time()
elapsed_time = end_time - start_time
print('Time elapsed (in seconds):', elapsed_time)

#%%
# save to csv names of columns with first 5 values by value counts for each column

columns_values = [df[col].value_counts().head(5).index.tolist() for col in df.columns]

# columns_values is a list of lists concatenate them row by row to the dataframe 
df_columns_values = pd.DataFrame(columns_values)
import openpyxl
pd.concat([pd.Series(df.columns), df_columns_values], axis=1).to_excel("data/processed/columns_values.xlsx", index=False)

# %% [markdown]
# read pickle file search_engine_changes
import pickle
search_engine_changes = pickle.load(open("data/processed/search_engine_changes.pkl", "rb"))
    
# %%
# filter df.columns.values  by regex
from src.data.filter_columns_by_regex import filter_columns_by_regex


experiment_vars = filter_columns_by_regex(df, '.*experiments')
experiment_vars

#
# %%
# df create a dummy variable before and after search_engine_changes['mobile.pay.support.Denmark']
df_dkk = (df.query('currency == "DKK"')
        .assign(before_mobile_support_denmark=lambda x: np.where(x['clicks_created_at_datetime'] < search_engine_changes['mobile.pay.support.Denmark'], "before_change", "after_change")))
#%%
# read data/external/google_trends_interest.csv
df_google_trends = pd.read_csv("data/external/google_trends_interest.csv")

# left join df_dkk and df_google_trends on clicks_created_at_datetime = date
df_dkk = df_dkk.assign(clicks_created_at_date=lambda x: x['clicks_created_at_datetime'].dt.date)
df_dkk = df_dkk.merge(df_google_trends[['date', 'google_trends_DK']], how='left', left_on='clicks_created_at_date', right_on='date', )


filter_columns_by_regex(df_dkk, 'trends')


# %%
from pandas.api.types import CategoricalDtype
from tzlocal import get_localzone
from datetime import datetime

time_filter_mobile_pay_support_Denmark = search_engine_changes['mobile.pay.support.Denmark'] - 1*(max(df_dkk['clicks_created_at_datetime']) - search_engine_changes['mobile.pay.support.Denmark'])
# time_filter_mobile_pay_support_Denmark = datetime(2023, 1, 1, 0, 0, 0)
# dkk add column orders_if_order_bin as boolean from orders_if_order
df_dkk = df_dkk.assign(orders_if_order_bin=lambda x: x['orders_if_order'] > 0)
df_dkk = df_dkk.assign(before_mobile_support_denmark_bin=lambda x: x['before_mobile_support_denmark']  == "after_change")


# Extracting weekend information
df_dkk['clicks_created_at_datetime_weekend'] = np.where(df_dkk['clicks_created_at_datetime'].dt.weekday.isin([5, 6]), 1, 0)

# Extracting hour information
df_dkk['clicks_created_at_datetime_hour'] = df_dkk['clicks_created_at_datetime'].dt.hour
#%%
# load df
x_cols = ['clicks_itinerary_travel_timehours', 
        #   'clicks_created_at_datetime',
          'google_trends_DK',
          'clicks_itinerary_totaldistance',
          'clicks_mobile', 
          'clicks_itinerary_sales_price_pax',  
          "clicks_itinerary_segment_count", 
          "clicks_created_at_datetime_weekend",
          "clicks_created_at_datetime_hour",
          "clicks_itinerary_with_baggage"]
y_col = 'orders_if_order_bin'
d_col = "before_mobile_support_denmark_bin"


# %%
# selecting only the columns we need
df_dkk_select = df_dkk[df_dkk['clicks_created_at_datetime'] > time_filter_mobile_pay_support_Denmark][x_cols + [y_col] + [d_col, 'before_mobile_support_denmark']]


# transform all variables that are boolean variables to 0/1
def bool_to_int(s: pd.Series) -> pd.Series:
    """Convert the boolean to binary representation, maintain NaN values."""
    return s.replace({True: 1, False: 0})

df_dkk_select = df_dkk_select.apply(bool_to_int)

from scipy import stats as stats
def perform_t_test(dataframe, group_var, var_test, group1, group2):

    print("test performed for "+group_var)
    print("variable tested: "+var_test)
    # Define the two groups to compare
    group1_data = dataframe.loc[dataframe[group_var] == group1, var_test]
    group2_data = dataframe.loc[dataframe[group_var] == group2, var_test]

    # Perform a two-sample t-test assuming unequal variances
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)

    # Calculate the mean and count for each group
    group1_mean = group1_data.mean()
    group1_count = group1_data.count()
    group2_mean = group2_data.mean()
    group2_count = group2_data.count()

    # Display the results
    print(group1 + " mean:", round(group1_mean,4), "count:", group1_count)
    print(group2 + " mean:", round(group2_mean,4), "count:", group2_count)
    
    # print("t-statistic:", t_stat)
    print("p-value:", p_value)

# perform_t_test(df_dkk_select, "before_mobile_support_denmark", "orders_if_order_bin", "before_change", "after_change")

# save df_dkk to csv
df_dkk_select.to_csv("data/processed/df_dkk_select.csv", index=False)

# %%-----------------------------
# Splitting the data into training and test sets
# -------------------------------
from sklearn.model_selection import train_test_split
# df_train, df_test = train_test_split(df_dkk_select, test_size=0.2, random_state=42)

x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(df_dkk_select[x_cols], df_dkk_select[y_col], df_dkk_select[d_col],
                                                                             test_size=0.5, random_state=42)

# %%-----------------------------
# CausalML approach
# scikit-uplift https://www.uplift-modeling.com/en/latest/api/models/TwoModels.html
# -------------------------------

# import approach
from sklift.models import TwoModels
# import any estimator adheres to scikit-learn conventions
from catboost import CatBoostClassifier

estimator_trmnt = CatBoostClassifier(silent=True, thread_count=2, random_state=42)
estimator_ctrl = CatBoostClassifier(silent=True, thread_count=2, random_state=42)

# define approach
tm_ctrl = TwoModels(
    estimator_trmnt=estimator_trmnt,
    estimator_ctrl=estimator_ctrl,
    method='ddr_control'
)

# fit the models
tm_ctrl = tm_ctrl.fit(
    x_train, y_train, treat_train,
    # estimator_trmnt_fit_params={'cat_features': },
    # estimator_ctrl_fit_params={'cat_features': }
)
uplift_tm_ctrl = tm_ctrl.predict(x_test)  # predict uplift
# uplift_tm_ctrl is an array, draw an histogram
import matplotlib.pyplot as plt
plt.hist(uplift_tm_ctrl, bins=100)

# extract summary statistics for the uplift_tm_ctrl
from sklift.metrics import uplift_at_k
uplift_at_k(y_true=y_test, uplift=uplift_tm_ctrl, treatment=treat_test, strategy='by_group', k=0.3)

# %%-----------------------------
# DoubleML approach
# -------------------------------
# Calculate the time difference between each observation and the target timestamp
# time_diff = abs(x_train['clicks_created_at_datetime'] - search_engine_changes['mobile.pay.support.Denmark'])

# Create a weight vector that assigns higher weights to observations that are closer to the target timestamp
# Calculate weights using the apply method with a lambda function
# weights = time_diff.apply(lambda x: np.exp(-x.total_seconds() / 1000))

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM


# split df into training and test sets
df_doubleml = DoubleMLData(df_dkk_select
                           , x_cols = x_cols, y_col = y_col, d_cols = d_col)

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
ml_g = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
ml_m = RandomForestClassifier(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
# fit the DoubleMLPLR model
dml_plr = DoubleMLPLR(
    df_doubleml, 
    ml_l=ml_g,
    ml_m=ml_m,
    n_folds=5 
    )
dml_plr_fit=dml_plr.fit()
dml_plr_fit.summary
# estimate the treatment effect using the DoubleMLPLR model


# %%
class RandomForestRegressorWithSampleWeight(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)


class RandomForestClassifierWithSampleWeight(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

ml_g = RandomForestRegressorWithSampleWeight()
ml_m = RandomForestClassifierWithSampleWeight()

dml_plr = DoubleMLPLR(
    df_doubleml, 
    ml_l=ml_g,
    ml_m=ml_m,
    n_folds=5
    )
dml_plr_fit = dml_plr.fit()
dml_plr_fit.summary

# %%
