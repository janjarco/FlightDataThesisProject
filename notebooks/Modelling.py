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

#%%
# read all pickles with modelling_dataset in the name into a dictionary
modelling_datasets = {}
for file in os.listdir("data/processed"):
    if "modelling_dataset" in file:
        modelling_datasets[file] = pd.read_pickle(f"data/processed/{file}")
new_keys = ["EUR", "DKK", "SEK", "NOK"]

modelling_datasets={new_keys[i]: modelling_datasets[j] for i, j in enumerate(modelling_datasets)}

#%%
# load df
x_cols = [
    'clicks_itinerary_travel_timehours', 
    'clicks_itinerary_totaldistance',
    'clicks_passengers_count',
    'google_trends',
    'clicks_mobile', 
    'clicks_itinerary_sales_price_pax',  
    "clicks_itinerary_segment_count", 
    "clicks_created_at_datetime_weekend",
    "clicks_created_at_datetime_hour",
    "clicks_itinerary_with_baggage", 
    "clicks_itinerary_direct_flight",
    "interaction_google_trends_distance",
    "interaction_google_trends_passengers",
    "interaction_google_trends_sales_price_pax",
    "interaction_google_trends_sales_price",
    "interaction_google_trends_travel_time",
    "interaction_google_trends_direct_flight",
    "interaction_google_trends_weekend",
    "interaction_google_trends_baggage",
    "ratio_sales_price_travel_time",
    "ratio_distance_passenger",
    "ratio_travel_time_distance",
    "ratio_sales_price_distance",
    "interaction_passengers_weekend",
    "interaction_mobile_direct_flight",
    ]
y_col = 'orders_if_order_bin'
d_col = "mobile_support_denmark"

modelling_df = modelling_datasets["DKK"]

# read pickle file search_engine_changes
import pickle
search_engine_changes = pickle.load(open("data/processed/search_engine_changes.pkl", "rb"))

time_filter_mobile_pay_support_Denmark = search_engine_changes['mobile.pay.support.Denmark'] - 1*(max(modelling_df['clicks_created_at_datetime']) - search_engine_changes['mobile.pay.support.Denmark'])
modelling_df = modelling_df[modelling_df['clicks_created_at_datetime'] > time_filter_mobile_pay_support_Denmark]

modelling_df[d_col]=np.where(modelling_df['clicks_created_at_datetime'] > search_engine_changes['mobile.pay.support.Denmark'], 1, 0).copy()

# %%-----------------------------
# Splitting the data into training and test sets
# -------------------------------
from sklearn.model_selection import train_test_split
# df_train, df_test = train_test_split(modelling_df, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[x_cols], modelling_df[y_col], modelling_df[d_col],
                                                                             test_size=0.5, random_state=42)

# %%-----------------------------
# Feature selection by variable importance
# -------------------------------
# import random forest regressor
from sklearn.ensemble import RandomForestRegressor

feature_select = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=5, min_samples_leaf=2)
feature_select_fit = feature_select.fit(X_train, y_train)

#%%
import numpy as np
import matplotlib.pyplot as plt

# Get the feature importances
importances = feature_select_fit.feature_importances_

# Sort the feature importances in descending order
sorted_idx = np.argsort(importances)[::-1]

# Get the feature names
feature_names = X_train.columns

# Create a horizontal bar plot of feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(X_train.shape[1]), importances[sorted_idx], align='center', color='royalblue')
plt.yticks(range(X_train.shape[1]), feature_names[sorted_idx])
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.title('Feature Importance')

# Show the plot
plt.show()
x_cols_selected = [x_cols[i] for i in range(X_train.shape[1]) if feature_select_fit.feature_importances_[i] > 0.02]

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


# %%
conv_rates_currencies_weekly = (df
                         # add a column with the week calculated from clicks_created_at_datetime to be the monday of the week
                .assign(clicks_created_at_week=lambda x: x['clicks_created_at_datetime'].dt.to_period('W').dt.start_time)
                .groupby(['clicks_created_at_week', 'currency'])['orders_if_order']
                .agg(orders=('sum'), clicks=('count'))
                .assign(conv_rate=lambda x: x['orders'] / x['clicks'])
                .reset_index())
conv_rates_currencies_weekly.head()
# %%
import matplotlib.pyplot as plt

# plot conv_rate_skk_weekly by date
fig, ax = plt.subplots(figsize=(15, 5))
# start a y axis from 0
ax.set_ylim(bottom=0, top=0.12)
# plot the conversion rate for each currency with separate line plot
for currency in conv_rates_currencies_weekly['currency'].unique():
    ax.plot(conv_rates_currencies_weekly[conv_rates_currencies_weekly['currency'] == currency]['clicks_created_at_week'], 
            conv_rates_currencies_weekly[conv_rates_currencies_weekly['currency'] == currency]['conv_rate'], 
            label=currency)
# add legend for each currency
ax.legend()

# add vertical line for each element in  search_engine_changes
for change in search_engine_changes:
    ax.axvline(search_engine_changes[change], color='r', linestyle='--')


# add a label for the search engine change named 'mobile.pay.support.Denmark'
ax.text(search_engine_changes['mobile.pay.support.Denmark'], 0.01, 'mobile.pay.support.Denmark', rotation=90)

# set title and axis labels
ax.set_title('Weekly Conversion Rates for currencies')
ax.set_xlabel('Date')
ax.set_ylabel('Conversion Rate')
# %%
# divide clicks_itinerary_sales_price  by clicks_itinerary_sales_price_pax and make a histogram
plt.hist(
 df['clicks_itinerary_sales_price'] / df['clicks_itinerary_sales_price_pax']
    , bins=100
    , range=(0, 100)
    , density=True
    , alpha=0.5
    , label='clicks_itinerary_sales_price / clicks_itinerary_sales_price_pax'
)

# %% visualize the results

# visualize histogram of y_pred_train and y_pred_test on one plot with alpha = .5
plt.hist(y_pred_train, bins=100, range=(0, .4), density=True, alpha=0.5, label='y_pred_train')
plt.hist(y_pred_test, bins=100, range=(0, .4), density=True, alpha=0.5, label='y_pred_test')
plt.legend()
plt.show()

# put y_pred_test to sigmoid function
y_pred_test_sigmoid = 1 / (1 + np.exp(-y_pred_test))
y_pred_train_sigmoid = 1 / (1 + np.exp(-y_pred_train))
y_train_sigmoid = 1 / (1 + np.exp(-y_train))
y_test_sigmoid = 1 / (1 + np.exp(-y_test))
# visualize histogram of y_pred_train and y_pred_test on one plot with alpha = .5
plt.hist(y_pred_train_sigmoid, bins=100, range=(0, 1), density=True, alpha=0.5, label='y_pred_train')
plt.hist(y_pred_test_sigmoid, bins=100, range=(0, 1), density=True, alpha=0.5, label='y_pred_test')
plt.hist(y_train_sigmoid, bins=100, range=(0, 1), density=True, alpha=0.5, label='y_train')
plt.hist(y_test_sigmoid, bins=100, range=(0, 1), density=True, alpha=0.5, label='y_test')
plt.legend()
plt.show()


#%% Regressor 
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import numpy as np
import os
import time
from tensorboardX import SummaryWriter
from sklearn.metrics import mean_squared_error, r2_score

modelling_df = modelling_df

# Assuming your features and target are stored in X and y
X_train, X_test, y_train, y_test = train_test_split(modelling_df[x_cols], modelling_df[y_col], test_size=0.2, random_state=42)

# Create a validation set from the training set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

ml_g = RandomForestRegressor()

param_grid_regressor = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75],
    'max_depth': [None, 5, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}

# print number of combinations to test from param_grid_regressor
print("Number of combinations to test from param_grid_regressor: ", np.prod([len(v) for v in param_grid_regressor.values()]))
import json
from sklearn.metrics import mean_squared_error
import math

def custom_scoring_function_regressor(estimator, X, y):
    y_pred_train = estimator.predict(X)

    neg_mse_train = -1 * mean_squared_error(y, y_pred_train)  # Use the negative MSE as score to maximize
    y_pred_val = estimator.predict(X_val)
    neg_mse_val = -1 * mean_squared_error(y_val, y_pred_val)  # Use the negative MSE as score to maximize

    return neg_mse_val


def log_metrics_regressor(random_search, logdir_name, n_iter_search= n_iter_search, eval_func = mean_squared_error):
    from tqdm import tqdm

    runs_dir = logdir_name + time.strftime("%Y%m%d_%H%M%S")
    mse_val_list = []
    log_dir_list = []
    for i in tqdm(range(n_iter_search)):
        params = random_search.cv_results_['params'][i]
        temp_model = RandomForestRegressor(**params)
        temp_model.fit(X_train, y_train)

        y_pred_val = temp_model.predict(X_val)
        eval_func_score = eval_func(y_val, y_pred_val)

        metrics_dict = {
            "mse_train" : -random_search.cv_results_['mean_test_score'][i],
            "mse_val" : eval_func_score
            } 
        mse_val_list.append(eval_func_score)

        log_dir = os.path.join("runs", runs_dir, time.strftime("%H%M%S"))
        log_dir_list.append(log_dir)

        writer = SummaryWriter(log_dir=log_dir)
        writer.add_hparams(
            hparam_dict={key: value for key, value in params.items() if key in param_grid_regressor.keys()},
            metric_dict=metrics_dict)
        writer.close()
    return  {"runs": log_dir_list,"params":random_search.cv_results_['params'], "mse_train": -random_search.cv_results_['mean_test_score'], "mse_val": mse_val_list,}
    


# hide warnings
import warnings
warnings.filterwarnings('ignore')
n_iter_search = 100
random_search_regressor = RandomizedSearchCV(estimator=ml_g, 
                                   param_distributions=param_grid_regressor, 
                                   n_iter=n_iter_search, 
                                   scoring=custom_scoring_function_regressor, 
                                   cv=5, n_jobs=-1, verbose=0, random_state=42)

random_search_regressor.fit(X_train, y_train)

rf_regressor_dict = log_metrics_regressor(random_search_regressor, "random_search_regressor_", n_iter_search = n_iter_search)

best_params_regressor = random_search_regressor.cv_results_['params'][np.argmin(rf_regressor_dict['mse_val'])]

# extract parameters where random_search.cv_results_['mean_test_score'] is the lowes
print("Best parameters found: ", best_params_regressor)

best_ml_g = RandomForestRegressor(**best_params_regressor)
best_ml_g.fit(X_train, y_train)

# calculate mse on train and test data
y_pred_train = best_ml_g.predict(X_train)
y_pred_val = best_ml_g.predict(X_val)
y_pred_test = best_ml_g.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_valid = mean_squared_error(y_val, y_pred_val)
mse_test = mean_squared_error(y_test, y_pred_test)

print("MSE on train data: ", mse_train)
print("MSE on validation data: ", mse_valid)
print("MSE on test data: ", mse_test)

# %%-----------------------------
# Classification


# Assuming your features and target are stored in X and y
X_train, X_test, d_train, d_test = train_test_split(modelling_df[x_cols], modelling_df[d_col], test_size=0.2, random_state=42)

# Create a validation set from the training set
X_train, X_val, d_train, d_val = train_test_split(X_train, d_train, test_size=0.1, random_state=42)


ml_m = RandomForestClassifier()

param_grid_classifier = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': [2, 3, 'sqrt', 4, 6],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# print number of combinations to test from param_grid_classifier
print("Number of combinations to test from param_grid_classifier: ", np.prod([len(v) for v in param_grid_classifier.values()]))
import json
from sklearn.metrics import accuracy_score

def custom_scoring_function_classifier(estimator, X, y):
    d_pred_train = estimator.predict(X)

    acc_train = accuracy_score(y, d_pred_train)  
    d_pred_val = estimator.predict(X_val)
    acc_val = accuracy_score(d_val, d_pred_val) 

    return acc_val

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import time

def log_metrics_classifier(random_search, logdir_name, n_iter_search= n_iter_search, eval_func=accuracy_score):
    from tqdm import tqdm

    runs_dir = logdir_name + time.strftime("%Y%m%d_%H%M%S")
    accuracy_val_list = []
    log_dir_list = []
    for i in tqdm(range(n_iter_search)):
        params = random_search.cv_results_['params'][i]
        temp_model = RandomForestClassifier(**params)
        temp_model.fit(X_train, d_train)

        d_pred_val = temp_model.predict(X_val)
        eval_func_score = eval_func(d_val, d_pred_val)

        metrics_dict = {
            "accuracy_train": random_search.cv_results_['mean_test_score'][i],
            "accuracy_val": eval_func_score
        }
        accuracy_val_list.append(eval_func_score)

        log_dir = os.path.join("runs", runs_dir, time.strftime("%H%M%S"))
        log_dir_list.append(log_dir)

        writer = SummaryWriter(log_dir=log_dir)
        writer.add_hparams(
            hparam_dict={key: value for key, value in params.items() if key in param_grid_classifier.keys()},
            metric_dict=metrics_dict)
        writer.close()
    return {"runs": log_dir_list, "params": random_search.cv_results_['params'], "accuracy_train": random_search.cv_results_['mean_test_score'], "accuracy_val": accuracy_val_list}

# hide warnings
import warnings
warnings.filterwarnings('ignore')
n_iter_search = 100
random_search_classifier = RandomizedSearchCV(estimator=ml_m, 
                                   param_distributions=param_grid_classifier, 
                                   n_iter=n_iter_search, 
                                   scoring=custom_scoring_function_classifier, 
                                   cv=5, n_jobs=-1, verbose=0, random_state=42)

random_search_classifier.fit(X_train, d_train)

rf_classifier_dict = log_metrics_classifier(random_search_classifier, "random_search_classifier_", n_iter_search = n_iter_search)

best_params_classifier = random_search_classifier.cv_results_['params'][np.argmax(rf_classifier_dict['accuracy_val'])]

# extract parameters where random_search.cv_results_['mean_test_score'] is the highest
print("Best parameters found: ", best_params_classifier)

best_ml_m = RandomForestClassifier(**best_params_classifier)
best_ml_m.fit(X_train, d_train)

# visualize variable importance in the model

from sklearn.metrics import accuracy_score

# calculate accuracy on train and test data
d_pred_train = best_ml_m.predict(X_train)
d_pred_val = best_ml_m.predict(X_val)
d_pred_test = best_ml_m.predict(X_test)

acc_train = accuracy_score(y_train, d_pred_train)
acc_valid = accuracy_score(y_val, d_pred_val)
acc_test = accuracy_score(y_test, d_pred_test)

print("Accuracy on train data: ", acc_train)
print("Accuracy on validation data: ", acc_valid)
print("Accuracy on test data: ", acc_test)

# %%-----------------------------
# DoubleML approach
# -------------------------------
# Calculate the time difference between each observation and the target timestamp
# time_diff = abs(x_train['clicks_created_at_datetime'] - search_engine_changes['mobile.pay.support.Denmark'])

# Create a weight vector that assigns higher weights to observations that are closer to the target timestamp
# Calculate weights using the apply method with a lambda function
# weights = time_diff.apply(lambda x: np.exp(-x.total_seconds() / 1000))

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM


# modelling_df check what is percentage of NAs
# modelling_df.isna().sum()/len(modelling_df)

# split df into training and test sets
df_doubleml = DoubleMLData(modelling_df, x_cols = x_cols_selected, y_col = y_col, d_cols = d_col)

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# ml_g = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=2,max_features=20)
# ml_m = RandomForestClassifier(**best_params_classifier)
best_params_classifier
ml_g = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=5, min_samples_leaf=2)
ml_m = RandomForestClassifier(n_estimators=100, max_features=2, max_depth=5, min_samples_leaf=2)

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
dml_plr_fit.summary
# dml_plr_fit.plot()
dml_plr_fit.plot_dml1()
# %%

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# split df into training and test sets
df_doubleml = DoubleMLData(modelling_df, x_cols=x_cols, y_col=y_col, d_cols=d_col)

# Initialize learners with default parameters
ml_g = RandomForestRegressor()
ml_m = RandomForestClassifier()

# fit the DoubleMLPLR model
dml_plr = DoubleMLPLR(
    df_doubleml, 
    ml_l=ml_g,
    ml_m=ml_m,
    n_folds=5
)


param_grid_regressor = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': ['sqrt', 4, ],
    'max_depth': [ 3, 5],
    'min_samples_split': [2, 5, ],
    'min_samples_leaf': [ 2, 4, 6]
}

param_grid_classifier = {
    'n_estimators': [100, 150, 200],
    'max_features': ['sqrt', 4, 6],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [2, 4, 6]
}

# Define parameter grids for each learner
param_grids = {
    'ml_l': param_grid_regressor,
    'ml_m': param_grid_classifier,
}

# Tune hyperparameters using randomized search
dml_plr.tune(
    param_grids,
    search_mode='randomized_search',
    scoring_methods = {
    "ml_l": "neg_mean_squared_error",
    "ml_m": "accuracy",
    },
    n_iter_randomized_search=40,
    n_folds_tune=5,
    n_jobs_cv=-1,  # Use all available CPUs
    return_tune_res=True
)

# Fit the model with the tuned hyperparameters
dml_plr_fit = dml_plr.fit()

# save to pickle 
import pickle
# add current date to file name
with open(f'dml_plr_fit_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
    pickle.dump([
        dml_plr, dml_plr_fit
        ], f)
    
# Estimate the treatment effect using the DoubleMLPLR model
dml_plr_fit.summary

dml_plr_fit.get_params(learner='ml_l')
dml_plr_fit.get_params(learner='ml_m')

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
dml_plr.evaluate_learners(metric=mean_squared_error)

dml_plr.evaluate_learners(learners = ["ml_m"], metric=accuracy_score)

# %%
