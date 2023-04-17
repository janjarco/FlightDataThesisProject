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
    # 'clicks_created_at_datetime', 
    'clicks_created_at_datetime_hour', 
    'clicks_created_at_datetime_weekend', 
    'clicks_itinerary_direct_flight', 
    'clicks_itinerary_sales_price_pax', 
    'clicks_itinerary_segment_count', 
    'clicks_itinerary_totaldistance', 
    'clicks_itinerary_travel_timehours', 
    'clicks_itinerary_with_baggage', 
    'clicks_mobile', 
    'clicks_passengers_count', 
    'google_trends', 
    'ratio_sales_price_travel_time',
    'ratio_distance_passenger',
    'ratio_travel_time_distance',
    'ratio_sales_price_distance',
    'carriers_marketing_ratings_count',
    'carriers_marketing_ratings_max',
    'carriers_marketing_ratings_min',
    'carriers_marketing_ratings_mean',
    'ratio_sales_price_carrier_rating_max',
    'ratio_sales_price_carrier_rating_min',
    'ratio_sales_price_carrier_rating_avg',
    'ratio_sales_price_carrier_rating_count',
    'clicks_itinerary_sales_price_if_cheapest',
    'clicks_itinerary_sales_price_if_best',
    'clicks_itinerary_sales_price_if_fastest',
    # 'clicks_itinerary_sales_price_category',
    'clicks_itinerary_sales_price_diff_cheapest',
    'clicks_itinerary_sales_price_diff_best',
    'clicks_itinerary_sales_price_diff_fastest',
    ]
y_col = 'orders_if_order'

modelling_df_full = modelling_datasets["DKK"]
# drop duplicated columns
modelling_df_full = modelling_df_full.loc[:,~modelling_df_full.columns.duplicated()]

# %% Splitting the data into training and test sets with filter by search engine changes
# -------------------------------
# read pickle file search_engine_changes
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

search_engine_changes = pickle.load(open("data/processed/search_engine_changes.pkl", "rb"))

def treatment_variable_generation(modelling_df, dates_dictionary, dates_dictionary_key, days_before, days_after):
    modelling_df_filter = modelling_df.copy()
    del modelling_df
    month_before = dates_dictionary[dates_dictionary_key] + timedelta(days=-days_before)
    month_after = dates_dictionary[dates_dictionary_key] + timedelta(days=days_after)

    def check_date_range(date):
        if month_before < date <= dates_dictionary[dates_dictionary_key]:
            return 0
        elif dates_dictionary[dates_dictionary_key] < date <= month_after:
            return 1
        else:
            return None
    treatment_var_name = "treatment_"+dates_dictionary_key
    modelling_df_filter[treatment_var_name] = modelling_df_filter['clicks_created_at_datetime'].apply(check_date_range)
    # print(modelling_df_filter.shape)
    # filter modelling if treatment variable is not null
    modelling_df_filter = modelling_df_filter[modelling_df_filter[treatment_var_name].notnull()]
    # print(modelling_df_filter.shape)
    # count na values for each column
    modelling_df_filter = modelling_df_filter.dropna()
    # print(modelling_df_filter.isna().sum())
    return modelling_df_filter, treatment_var_name

modelling_df, treatment_var_name = treatment_variable_generation(modelling_datasets['DKK'], search_engine_changes, "hack.bagprice.override.Altea.FLX", 30, 30)

# %%
# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split

# split the data into training and test sets
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[x_cols], modelling_df[y_col], modelling_df[treatment_var_name], test_size=0.2, random_state=42)
# %%
# COvariates axploration
# for all the variables in x_colsm list plot density plot with grouping by d_col
import matplotlib.pyplot as plt
import seaborn as sns
for col in x_cols:
    fig, ax = plt.subplots()
    sns.kdeplot(modelling_df.loc[modelling_df[d_col] == 0, col], label = 'control', ax=ax)
    sns.kdeplot(modelling_df.loc[modelling_df[d_col] == 1, col], label = 'treatment', ax=ax)
    plt.show()

# %%
# for all the variables in x_colsm list plot density plot with grouping by d_col
import matplotlib.pyplot as plt
import seaborn as sns
for col in x_cols:
    fig, ax = plt.subplots()
    sns.kdeplot(modelling_df.loc[modelling_df[y_col] == 0, col], label = 'control', ax=ax)
    sns.kdeplot(modelling_df.loc[modelling_df[y_col] == 1, col], label = 'treatment', ax=ax)
    plt.show()


# %%-----------------------------
# Feature importance functions

# Feature importance plot
def plot_feature_importances(model, df_train):
    import numpy as np
    import matplotlib.pyplot as plt

    # Get the feature importances
    importances = model.feature_importances_

    # Sort the feature importances in descending order
    sorted_idx = np.argsort(importances)[::-1]

    # Get the feature names
    feature_names = df_train.columns

    # Create a horizontal bar plot of feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(df_train.shape[1]), importances[sorted_idx], align='center', color='royalblue')
    plt.yticks(range(df_train.shape[1]), feature_names[sorted_idx])
    plt.ylabel('Feature')
    plt.xlabel('Importance')
    plt.title('Feature Importance')

    # Show the plot
    plt.show()

# Shapley values
def shapley_values(model, df_train):
    import shap

    # Create an explainer object using the random forest regressor and training data
    explainer = shap.explainers.Tree(model, df_train)

    # Compute Shapley values for the entire dataset
    shap_object = explainer(df_train)
    
    return shap_object

# Shapley values plot 
def plot_shapley_values(shap_object, show_bar=False, show_waterfall=False, show_beeswarm=False):
    import shap
    import matplotlib.pyplot as plt


    if show_bar:
        shap.plots.bar(shap_object*100, max_display=len(shap_object.feature_names))
        plt.show()
    # Create a summary plot of the Shapley values
    if show_waterfall:
        shap.plots.waterfall(shap_object[0], max_display=len(shap_object.feature_names))
        plt.show()

    # summarize the effects of all the features
    if show_beeswarm:
        shap.plots.beeswarm(shap_object, max_display=len(shap_object.feature_names), )
        plt.show()

# Shaapley values dataframe
def shap_values_df(shap_object, df_train):
    import numpy as np
    import pandas as pd
    # Compute SHAP values for your dataset
    shap_values = shap_object.values

    # Compute the mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame with feature names and SHAP values
    shap_df = pd.DataFrame({'feature': df_train.columns, 'mean_abs_shap': mean_abs_shap_values}).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_df['cumulative_importance'] = shap_df['mean_abs_shap'].cumsum() / shap_df['mean_abs_shap'].sum()
    # Sort the DataFrame based on the SHAP values
    # shap_df = shap_df.sort_values(by='feature', ascending=True)
    return shap_df


# %%-----------------------------
# Correlation plot between features
# -------------------------------
def plot_correlation_matrix(df):
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    import seaborn as sns

    # Create a correlation matrix
    corr_matrix = df[sorted(df.columns.values)].corr()

    # Create a mask to display only the lower triangle of the matrix
    corr_matrix_tril = np.tril(np.ones_like(corr_matrix)) * corr_matrix

    # Create a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Create a heatmap without annotations
    fig = go.Figure(go.Heatmap(
        z=corr_matrix_tril.values,
        x=list(corr_matrix_tril.columns),
        y=list(corr_matrix_tril.index),
        colorscale=px.colors.sequential.RdBu,
        # display values in the heatmap
        text=corr_matrix_tril.values.round(2),
        hovertemplate='(%{y}, %{x}): %{z:.2f}<extra></extra>'
    ))

    # Customize the layout
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis=dict(title='Features', side='bottom', ticks='outside'),
        yaxis=dict(title='Features', ticks='outside', autorange='reversed'),
        margin=dict(t=50, l=50, b=100),
        width=1000,
        height=800
    )

    fig.show()

plot_correlation_matrix(X_train)

# %%-----------------------------
# to exclude features with high correlation b etween each other
feature_to_exclude = [
    'carriers_marketing_ratings_max',
    'carriers_marketing_ratings_min',

    'ratio_sales_price_carrier_rating_avg', 
    'ratio_sales_price_carrier_rating_min'
    ]

# Feature selection by shapley values
from sklearn.ensemble import RandomForestRegressor

# drop columns from X_train that are in feature_to_exclude
X_train_updated = X_train.drop(feature_to_exclude, axis=1)

feature_select_update = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=5, min_samples_leaf=2)
feature_select_update_fit = feature_select_update.fit(X_train_updated, y_train)

plot_feature_importances(feature_select_update_fit, X_train_updated)

shap_values_update = shapley_values(feature_select_update_fit, X_train_updated)
shap_df_update = shap_values_df(shap_values_update, X_train_updated)


plot_shapley_values(shap_values_update, show_bar=False, show_waterfall=False, show_beeswarm=True)
 
# Set the cutoff threshold
threshold = 1.0

# Find the index of the first feature that meets the threshold
cutoff_index = shap_df_update[shap_df_update['cumulative_importance'] >= threshold].index[0]

# Select features that meet the threshold
selected_features = shap_df_update.iloc[:cutoff_index+1]['feature'].tolist()
# %%
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[selected_features], modelling_df[y_col], modelling_df[d_col],
                                                                             test_size=0.2, random_state=42)

# %% conv_rates_currencies_weekly
conv_rates_currencies_weekly = (modelling_datasets['DKK']
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
# %% divide clicks_itinerary_sales_price  by clicks_itinerary_sales_price_pax and make a histogram
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

# %% Classification



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

ml_g = RandomForestRegressor(n_estimators=100,min_samples_split=2,min_samples_leaf=4,max_features='sqrt',max_depth=3)
ml_m = RandomForestClassifier(n_estimators=100,min_samples_split= 2,min_samples_leaf= 6,max_features= 6,max_depth= 3)

# import xgboost classifier and regressor
from xgboost import XGBClassifier, XGBRegressor
# declare sample xgboost regressor with sample parameters
xgb_regressor = XGBRegressor( n_estimators=100, max_depth=5, min_child_weight=2, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_lambda=1, reg_alpha=0, objective='reg:squarederror', random_state=42 )
# declare sample xgboost classifier with sample parameters
xgb_classifier = XGBClassifier( n_estimators=100, max_depth=5, min_child_weight=2, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, gamma=0.1, reg_lambda=1, reg_alpha=0, objective='binary:logistic', random_state=42 )


# fit the DoubleMLPLR model
dml_plr = DoubleMLPLR(
    df_doubleml, 
    ml_l=xgb_regressor,
    ml_m=xgb_classifier,
    n_folds=5 
    )
dml_plr_fit=dml_plr.fit()
dml_plr_fit.summary
# estimate the treatment effect using the DoubleMLPLR model
# dml_plr_fit.plot()

# %% Double ML Random Forrest hyperparameter tuning

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# split df into training and test sets
df_doubleml = DoubleMLData(modelling_df, x_cols=x_cols_selected, y_col=y_col, d_cols=d_col)

# Initialize learners with default parameters
ml_g = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=2,
    max_features=20
)
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
    'max_features': [3, 'sqrt', ],
    'max_depth': [ 3, 4, 5],
    'min_samples_split': [2, 5, ],
    'min_samples_leaf': [ 2, 4, 6]
}

param_grid_classifier = {
    'n_estimators': [100, 150, 200],
    'max_features': [3, 'sqrt', 6],
    'max_depth': [3, 4, 5],
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
    return_tune_res=True,
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
dml_plr_fit.get_params(learner='ml_m')['mobile_support_denmark'][0][0]

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
dml_plr.evaluate_learners(metric=mean_squared_error)

dml_plr.evaluate_learners(learners = ["ml_m"], metric=accuracy_score)

# 
# %% Uplift modelling from scikit-uplift
# -------------------------------------
def uplift_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test, 
                    classifier, classifier_params, regressor, regressor_params, 
                    selected_features, y_col, d_col):
    # declare sample xgboost regressor with sample parameters
    # add time measurement 
    from datetime import datetime
    start_time = datetime.now()

    models_results = {
        'approach': [],
        'uplift': []
    }

    # Double ML
    # fit the DoubleMLPLR model

    from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # split df into training and test sets
    df_doubleml = DoubleMLData(pd.concat([X_train, y_train, treat_train], axis=1), x_cols=selected_features, y_col=y_col, d_cols=d_col)


    dml_plr = DoubleMLPLR(df_doubleml, 
                          ml_l=regressor(**regressor_params), 
                          ml_m=classifier(**classifier_params), 
                          n_folds=5 )
    dml_plr=dml_plr.fit()
    dm_score = dml_plr.summary['coef'].values[0]

    models_results['approach'].append('DoubleML')
    models_results['uplift'].append(dm_score)

    # Solo Model
    from sklift.metrics import uplift_at_k
    from sklift.viz import plot_uplift_preds
    from sklift.models import SoloModel

    sm = SoloModel(classifier(**classifier_params))
    sm = sm.fit(X_train, y_train, treat_train)

    uplift_sm = sm.predict(X_test)

    sm_score = uplift_at_k(y_true=y_test, uplift=uplift_sm, treatment=treat_test, strategy='by_group', k=0.3)

    models_results['approach'].append('SoloModel')
    models_results['uplift'].append(sm_score)

    # get conditional probabilities (predictions) of performing the target action 
    # during interaction for each object
    sm_trmnt_preds = sm.trmnt_preds_
    # And conditional probabilities (predictions) of performing the target action 
    # without interaction for each object
    sm_ctrl_preds = sm.ctrl_preds_

    # draw the probability (predictions) distributions and their difference (uplift)
    # plot_uplift_preds(trmnt_preds=sm_trmnt_preds, ctrl_preds=sm_ctrl_preds)

    # sm_fi = pd.DataFrame({
    #     'feature_name': sm.estimator.feature_names_,
    #     'feature_score': sm.estimator.feature_importances_
    # }).sort_values('feature_score', ascending=False).reset_index(drop=True)

    # Class Transformation

    from sklift.models import ClassTransformation

    ct = ClassTransformation(classifier(**classifier_params))
    ct = ct.fit(X_train, y_train, treat_train)

    uplift_ct = ct.predict(X_test)

    ct_score = uplift_at_k(y_true=y_test, uplift=uplift_ct, treatment=treat_test, strategy='by_group', k=0.3)

    # plot_uplift_preds(trmnt_preds=ct.trmnt_preds_, ctrl_preds=ct.ctrl_preds_)

    models_results['approach'].append('ClassTransformation')
    models_results['uplift'].append(ct_score)

    # Two models
    import importlib
    import sklift.models
    importlib.reload(sklift.models)


    import sklift.models.models
    importlib.reload(sklift.models.models)
    from sklift.models import TwoModels
    # Two models treatment

    tm_trmnt = TwoModels(
        estimator_trmnt=classifier(**classifier_params), 
        estimator_ctrl=classifier(**classifier_params), 
        method='ddr_treatment'
    )
    tm_trmnt = tm_trmnt.fit(X_train, y_train, treat_train)

    uplift_tm_trmnt = tm_trmnt.predict(X_test)

    tm_trmnt_score = uplift_at_k(y_true=y_test, uplift=uplift_tm_trmnt, treatment=treat_test, strategy='by_group', k=0.3)

    models_results['approach'].append('TwoModels_ddr_treatment')
    models_results['uplift'].append(tm_trmnt_score)
    
    # plot_uplift_preds(trmnt_preds=tm_trmnt.trmnt_preds_, ctrl_preds=tm_trmnt.ctrl_preds_)

    # Two models control
    tm_ctrl = TwoModels(
        estimator_trmnt=classifier(**classifier_params), 
        estimator_ctrl=classifier(**classifier_params), 
        method='ddr_control'
    )
    tm_ctrl = tm_ctrl.fit(X_train, y_train, treat_train)

    uplift_tm_ctrl = tm_ctrl.predict(X_test)

    tm_ctrl_score = uplift_at_k(y_true=y_test, uplift=uplift_tm_ctrl, treatment=treat_test, strategy='by_group', k=0.3)

    models_results['approach'].append('TwoModels_ddr_control')
    models_results['uplift'].append(tm_ctrl_score)

    # plot_uplift_preds(trmnt_preds=tm_ctrl.trmnt_preds_, ctrl_preds=tm_ctrl.ctrl_preds_)
    # print timedifference
    print('Time taken: {}'.format(datetime.now() - start_time), )

    return pd.DataFrame(models_results), [dml_plr, sm, ct, tm_trmnt, tm_ctrl]

# %% XGBoost in all the models
xgb_modelling_results={
    'treatment':[],
    'results_df':[], 
    'models_obj':[]
}

for treatment in search_engine_changes.keys():
    print(treatment)
    modelling_df, treatment_var_name = treatment_variable_generation(modelling_df_full, search_engine_changes, treatment, 30, 30)
    # train test split
    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[selected_features], modelling_df[y_col], modelling_df[treatment_var_name], test_size=0.2, random_state=42)

    # import xgb classifier and regressor
    from xgboost import XGBClassifier, XGBRegressor

    xgb_classifier_params = { "n_estimators":100, "max_depth":5, "min_child_weight":2, "learning_rate":0.01, "subsample":0.8, "colsample_bytree":0.8, "gamma":0.1, "reg_lambda":1, "reg_alpha":0, "objective":'binary:logistic', "random_state":42, }
    xgb_regressor_params = { "n_estimators":100, "max_depth":5, "min_child_weight":2, "learning_rate":0.01, "subsample":0.8, "colsample_bytree":0.8, "gamma":0.1, "reg_lambda":1, "reg_alpha":0, "objective":'reg:squarederror', "random_state":42, }

    xgb_models_results, xgb_models_objects = uplift_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=XGBClassifier, classifier_params=xgb_classifier_params,
                                            regressor=XGBRegressor, regressor_params=xgb_regressor_params,
                                            selected_features=selected_features, y_col=y_col, d_col=treatment_var_name)
    xgb_modelling_results['treatment'].append(treatment_var_name)
    xgb_modelling_results['results_df'].append(xgb_models_results)
    xgb_modelling_results['models_obj'].append(xgb_models_objects)

xgb_modelling_results=pd.concat([pd.DataFrame(xgb_modelling_results['results_df'][i]["uplift"].round(3)).rename(columns = {"uplift":xgb_modelling_results['treatment'][i]}) for i in range(len(xgb_modelling_results['results_df']))], axis = 1)
xgb_modelling_results.index = ["DoubleML", "SoloModel", "ClassTransformation", "TwoModels_ddr_treatment", "TwoModels_ddr_control"]
xgb_modelling_results.T.style.background_gradient(cmap='RdYlGn', vmin=-0.01, vmax=0.01, low=0, high=1, axis=None).format("{:.3f}")


# %% LightGBM in all the models
# import LGBM classifier and regressor 
from lightgbm import LGBMClassifier, LGBMRegressor

lgb_classifier_params = {
        "n_estimators":100,
        "max_depth":5,
        "learning_rate":0.1,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "reg_lambda":1,
        "reg_alpha":0,
        "random_state":42,
    }
lgb_regressor_params = {
        "n_estimators":100,
        "max_depth":5,
        "learning_rate":0.1,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "reg_lambda":1,
        "reg_alpha":0,
        "random_state":42,
    }

lgb_models_results = uplift_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                        classifier=LGBMClassifier, classifier_params=lgb_classifier_params,
                                        regressor=LGBMRegressor, regressor_params=lgb_regressor_params,
                                        selected_features=selected_features, y_col=y_col, d_col=d_col)

print(lgb_models_results)

# %% RandomForrest in all the models
# import RandomForest classifier and regressor
modelling_df_dict = {}

for treatment in search_engine_changes.keys():
    print(treatment)
    modelling_df, treatment_var_name = treatment_variable_generation(modelling_df_full, search_engine_changes, treatment, 30, 30)
    # train test split
    X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[selected_features], modelling_df[y_col], modelling_df[treatment_var_name], test_size=0.2, random_state=42)
    df_dict = { 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'treat_train':treat_train, 'treat_test':treat_test }

    modelling_df_dict[treatment] = df_dict

# %%
rf_modelling_results={ 'treatment':[], 'results_df':[], 'models_obj':[] }

for treatment in search_engine_changes.keys():
    

    # import rf classifier and regressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    rf_regressor_params = { "n_estimators":100, "max_depth":5, "min_samples_leaf":2, "max_features":20 }
    rf_classifier_params = { "n_estimators":100, "max_depth":5, "min_samples_leaf":2, "max_features":20 }
    
    rf_models_results, rf_models_objects = uplift_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=RandomForestClassifier, classifier_params=rf_classifier_params,
                                            regressor=RandomForestRegressor, regressor_params=rf_regressor_params,
                                            selected_features=selected_features, y_col=y_col, d_col=treatment_var_name)
    rf_modelling_results['treatment'].append(treatment_var_name)
    rf_modelling_results['results_df'].append(rf_models_results)
    rf_modelling_results['models_obj'].append(rf_models_objects)

rf_modelling_results=pd.concat([pd.DataFrame(rf_modelling_results['results_df'][i]["uplift"].round(3)).rename(columns = {"uplift":rf_modelling_results['treatment'][i]}) for i in range(len(rf_modelling_results['results_df']))], axis = 1)
rf_modelling_results.index = ["DoubleML", "SoloModel", "ClassTransformation", "TwoModels_ddr_treatment", "TwoModels_ddr_control"]
rf_modelling_results.T.style.background_gradient(cmap='RdYlGn', vmin=-0.01, vmax=0.01, low=0, high=1, axis=None).format("{:.3f}")

#%% 


uplift_sm = sm.predict(X_test)
uplift_ct = ct.predict(X_test)

# statistical test if its bigger than 0
from scipy import stats

# generate numpy array with 0 of number of rows in X_test
zero_array = np.zeros(X_test.shape[0])
stats.ttest_ind(uplift_ct, zero_array, equal_var=False)
stats.ttest_ind(uplift_sm, zero_array, equal_var=False)

# %% Saving workspace 
# to saved_workspaces/workspace_{now}.pkl with time after user confirmation
import dill
import datetime
 
# Prompt user for confirmation
confirmation = input("Are you sure you want to save the workspace? (yes/no) ")

# Save workspace if user confirms
if confirmation == "yes":
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"saved_workspaces/workspace_{now}.pkl"
    with open(filename, "wb") as f:
        dill.dump_session(f)
    print("Workspace saved.")
else:
    print("Workspace not saved.")
 
# %% Loading workspace from saved_workspaces/workspace_name.pkl
import dill

# Load workspace from file
filename = "/saved_workspaces/workspace_2023-04-16_14-28-22.pkl"
with open(os.getcwd() + filename, "rb") as f:
    dill.load_session(f)

# Workspace variables are now available in this script
print("Workspace loaded.")

# %%
