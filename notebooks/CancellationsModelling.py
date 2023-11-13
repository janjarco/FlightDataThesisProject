# %% [markdown]
# ### Import libraries

# %%
import os
import pandas as pd
import numpy as np



#set working directory to this from the workspace
os.chdir('/Users/janjarco/Programming/PrivateRepository/FlightDataThesisProject')

from src.data.filter_columns_by_regex import filter_columns_by_regex

#list all the files in current working directory
print(os.listdir("data"))

# #%%
# # read all pickles with modelling_dataset in the name into a dictionary
# modelling_datasets = {}
# for file in os.listdir("data/processed"):
#     if "modelling_dataset" in file:
#         modelling_datasets[file] = pd.read_pickle(f"data/processed/{file}")
# new_keys = ["EUR", "DKK", "SEK", "NOK"]

# modelling_datasets={new_keys[i]: modelling_datasets[j] for i, j in enumerate(modelling_datasets)}

#%%
# load df
import pickle
modelling_df_full = pd.read_pickle("data/processed/modelling_dataset_orders_DK.pkl")
modelling_df_full['clicks_created_at_datetime_weekday'] = pd.to_datetime(modelling_df_full['clicks_created_at_datetime']).dt.dayofweek + 1
# scale google trends from 0 to 100

for col in filter_columns_by_regex(modelling_df_full, "google.*_DK"):
    modelling_df_full[col] = modelling_df_full[col].copy() / modelling_df_full[col].max() * 100

# reduce with PCA all varablies in filter_columns_by_regex(modelling_df_full, "google.*_DK")
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
modelling_df_full['google_trends_reduced_DK'] = pca.fit_transform(modelling_df_full[filter_columns_by_regex(modelling_df_full, "smooth.*_DK")]).reshape(-1,1)


modelling_df_full['interaction_google_trends_reduced_weekday'] = modelling_df_full['clicks_created_at_datetime_weekday'] * modelling_df_full['google_trends_reduced_DK']
# modelling_df_full['interaction_google_trends_weekly_weekday'] = modelling_df_full['clicks_created_at_datetime_weekday'] * modelling_df_full['google_trends_weekly_DK']

modelling_df_full['clicks_created_at_datetime_date'] = pd.to_datetime(modelling_df_full['clicks_created_at_datetime']).dt.date
# group by date and count number of observations for each day in from clicks_created_at_datetime
orders_count = pd.DataFrame(modelling_df_full
 .groupby('clicks_created_at_datetime_date')['clicks_created_at_datetime_date']
 .count().reset_index(name='orders_count'))

# orders_count group by week and summarize the sum of orders_count for weach week
# create a column with a first day of the week (monday) on the basis of date clicks_created_at_datetime_date
orders_count['clicks_created_at_datetime_date'] = pd.to_datetime(orders_count['clicks_created_at_datetime_date'])
# orders_count['clicks_created_at_datetime_week'] = orders_count['clicks_created_at_datetime_date'] - pd.to_timedelta(orders_count['clicks_created_at_datetime_date'].dt.dayofweek, unit='d')

google_trends = pd.read_csv("data/external/google_trends_interest.csv")
google_trends['date'] = pd.to_datetime(google_trends['date'])
# google_trends['week'] = google_trends['date'] - pd.to_timedelta(google_trends['date'].dt.dayofweek, unit='d')
# google_trends = google_trends.groupby('week').mean().reset_index()

orders_count_google_trends = orders_count.merge(google_trends, how='inner', left_on='clicks_created_at_datetime_date', right_on='date')

orders_count_google_trends = orders_count_google_trends[['clicks_created_at_datetime_date', 'orders_count', 'google_trends_smooth_DK']]

# transform orders_count and google_trends_smooth_DK with min max scaling
orders_count_google_trends['orders_count'] = orders_count_google_trends['orders_count'] / orders_count_google_trends['orders_count'].max() * 100
orders_count_google_trends['google_trends_smooth_DK'] = orders_count_google_trends['google_trends_smooth_DK'] / orders_count_google_trends['google_trends_smooth_DK'].max() * 100
orders_count_google_trends['ratio_clicks_google_trends'] = orders_count_google_trends['orders_count'] / orders_count_google_trends['google_trends_smooth_DK']
orders_count_google_trends['diff_clicks_google_trends'] = orders_count_google_trends['orders_count'] - orders_count_google_trends['google_trends_smooth_DK']

# make this to string orders_count_google_trends['clicks_created_at_datetime_date'] 
# orders_count_google_trends['clicks_created_at_datetime_date'] = orders_count_google_trends['clicks_created_at_datetime_date'].astype(str)
modelling_df_full['clicks_created_at_datetime_date'] = pd.to_datetime(modelling_df_full['clicks_created_at_datetime_date'])

# left join orders_count_google_trends to modelling_df_full on clicks_created_at_datetime_date
modelling_df_full = modelling_df_full.merge(orders_count_google_trends.drop("google_trends_smooth_DK", axis=1), how='left', left_on='clicks_created_at_datetime_date', right_on='clicks_created_at_datetime_date', suffixes=('', ''))

# for columns 'orders_search_data.search_parameters.children', 'orders_search_data.search_parameters.infants', 'orders_search_data.search_parameters.adults', replace missing values with 0
modelling_df_full['orders_search_data.search_parameters.children'] = modelling_df_full['orders_search_data.search_parameters.children'].fillna(0)
modelling_df_full['orders_search_data.search_parameters.infants'] = modelling_df_full['orders_search_data.search_parameters.infants'].fillna(0)
modelling_df_full['orders_search_data.search_parameters.adults'] = modelling_df_full['orders_search_data.search_parameters.adults'].fillna(0)

# modelling_df_full = (modelling_df_full.merge(orders_count_google_trends, how='left', left_on='clicks_created_at_datetime_date', right_on='clicks_created_at_datetime_date'))

x_cols = [
    'clicks_created_at_datetime', 
    'clicks_created_at_datetime_hour', 
    'clicks_created_at_datetime_weekend', 
    'clicks_created_at_datetime_weekday',
    'clicks_itinerary_direct_flight', 
    'clicks_itinerary_sales_price_pax', 
    'clicks_itinerary_segment_count', 
    'clicks_itinerary_totaldistance', 
    'clicks_itinerary_travel_timehours', 
    'clicks_itinerary_with_baggage', 
    'clicks_mobile', 
    'clicks_passengers_count', 
    'ratio_sales_price_travel_time',
    'ratio_distance_passenger',
    'ratio_travel_time_distance',
    'ratio_sales_price_distance',
    'carriers_marketing_ratings_max',
    'carriers_marketing_ratings_min',   
    'carriers_marketing_ratings_mean',
    'carriers_marketing_ratings_count',
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
    'orders_count', 
    'google_trends_weekly_DK',
    'google_trends_weekly_DK_lag_7',
    'ratio_clicks_google_trends',
    'diff_clicks_google_trends',
    'google_trends_smooth_DK',
    'google_trends_smooth_DK_lag_1',
    'google_trends_smooth_DK_lag_2',
    'google_trends_smooth_DK_lag_3',
    'google_trends_smooth_DK_lag_4',
    'google_trends_smooth_DK_lag_5',
    'google_trends_smooth_DK_lag_6',
    'google_trends_smooth_DK_lag_7',

    'orders_search_data.search_parameters.children', 
    'orders_search_data.search_parameters.infants', 
    'orders_search_data.search_parameters.adults',
    'orders_order_type',
    'orders_search_data.gate_name',
    'orders_addon_totalsum',
    # 'orders_search_data.search_parameters.last_search',
    # 'orders_search_data.search_parameters.only_with_bags',
    # 'orders_search_data.search_parameters.only_direct_flights',
    # 'orders_search_data.search_parameters.max_one_stop',
    # 'orders_search_data.search_parameters.only_private_fares',
    ]

# x_cols = x_cols + filter_columns_by_regex(modelling_df_full, "google.*_DK")
y_col = 'orders_cancelled'
# modelling_df_full[['clicks_created_at_datetime', 'clicks_created_at_datetime_weekday']].tail(50)

# drop duplicated columns
modelling_df_full = modelling_df_full[x_cols + [y_col]]
modelling_df_full = modelling_df_full.loc[:,~modelling_df_full.columns.duplicated()]
modelling_df_full['clicks_created_at_datetime'] = pd.to_datetime(modelling_df_full['clicks_created_at_datetime'])

# remove missing values from modelling_df_full
no_rows_before_drop = modelling_df_full.shape[0]
modelling_df_full = modelling_df_full.dropna()
print(modelling_df_full.shape[0] /no_rows_before_drop)
# modelling_df, treatment_var_name = treatment_variable_generation(modelling_datasets['DKK'], search_engine_changes, "hack.bagprice.override.Altea.FLX", 30, 30)

# %%
# Splitting the data into training and test sets
from sklearn.model_selection import train_test_split

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(modelling_df_full[x_cols], modelling_df_full[y_col], test_size=0.2, random_state=42)
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
def plot_feature_importances(model, df):
    import numpy as np
    import matplotlib.pyplot as plt

    # Get the feature importances
    importances = model.feature_importances_

    # Sort the feature importances in descending order
    sorted_idx = np.argsort(importances)[::-1]

    # Get the feature names
    feature_names = df.columns.values

    # Create a horizontal bar plot of feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances[sorted_idx], align='center', color='royalblue')
    plt.yticks(range(len(importances)), feature_names[sorted_idx])
    plt.ylabel('Feature')
    plt.xlabel('Importance')
    plt.title('Feature Importance')

    # Show the plot
    plt.show()

# Shapley values
def shapley_values(model, X_train):
    import shap

    # Create an explainer object using the random forest regressor and training data
    explainer = shap.explainers.Tree(model, X_train)

    # Compute Shapley values for the entire dataset
    shap_object = explainer(X_train)
    
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
def shap_values_df(shap_object, X_train):
    import numpy as np
    import pandas as pd
    # Compute SHAP values for your dataset
    shap_values = shap_object.values

    # Compute the mean absolute SHAP values for each feature
    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame with feature names and SHAP values
    shap_df = pd.DataFrame({'feature': X_train.columns, 'mean_abs_shap': mean_abs_shap_values}).sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
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

# plot histogram of google_trends in X_train
# X_train['interaction_google_trends_weekday'].hist(bins = 100)
# X_train['clicks_created_at_datetime_weekday'].hist(bins = 100)
# %%-----------------------------
# to exclude features with high correlation b etween each other
feature_to_exclude = [
    # 'carriers_marketing_ratings_max',
    # 'carriers_marketing_ratings_min',
    'clicks_created_at_datetime', 
    # 'ratio_sales_price_carrier_rating_avg', 
    # 'ratio_sales_price_carrier_rating_min',
    'carriers_marketing_ratings_max',
    'carriers_marketing_ratings_min',
    'ratio_sales_price_carrier_rating_min',
    'ratio_sales_price_carrier_rating_avg',
    # 'clicks_created_at_datetime'
    # 'orders_order_type',
    # 'google_trends_reduced_DK',
    ] #+ filter_columns_by_regex(X_train, 'weekly|smooth')
selected_features = list(set([i for i in x_cols if i not in feature_to_exclude+["orders_cancelled"]]))
# %% Feature selection by shapley values
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

# drop columns from X_train that are in feature_to_exclude
X_train_updated = X_train.drop(feature_to_exclude, axis=1)

from sklearn.model_selection import train_test_split

# import plot_feature_importances from xgboost 
from xgboost import plot_importance
feature_select_update = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.5, colsample_bytree=0.5, random_state=42)
feature_select_update_fit = feature_select_update.fit(X_train_updated, y_train)
feature_select_update_fit.feature_types
plot_importance(feature_select_update_fit, X_train_updated)

# shap_values_update = shapley_values(feature_select_update_fit, X_train_updated)
# shap_df_update = shap_values_df(shap_values_update, X_train_updated)


# plot_shapley_values(shap_values_update, show_bar=False, show_waterfall=False, show_beeswarm=True)
 
# Set the cutoff threshold
# threshold = 1.0

# Find the index of the first feature that meets the threshold
# cutoff_index = shap_df_update[shap_df_update['cumulative_importance'] >= threshold].index[0]
#%%
# Select features that meet the threshold
# selected_features = shap_df_update.iloc[:cutoff_index+1]['feature'].tolist()
# selected_features = [i for i in x_cols if i not in feature_to_exclude+["orders_cancelled"]]
# %%
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[selected_features], modelling_df[y_col], modelling_df["mobile.pay.support.Denmark"],
                                                                             test_size=0.2, random_state=42)

# %% conv_rates_currencies_weekly
cancellations_currencies_weekly = (modelling_df_full
                         # add a column with the week calculated from clicks_created_at_datetime to be the monday of the week
                .assign(clicks_created_at_week=lambda x: x['clicks_created_at_datetime'].dt.to_period('W').dt.start_time)
                # add google_trends variable and orders_if_order groupby clicks_created_at_week
                .groupby(['clicks_created_at_week'])[['google_trends_weekly_DK', 'orders_cancelled']] 
                .agg(cancellation_rate=('orders_cancelled', 'mean'), google_trends=('google_trends_weekly_DK', 'mean'))
                .reset_index())
cancellations_currencies_weekly.head()

# %% conv_rates_currencies_daily
cancellations_currencies_daily = (modelling_df_full
                         # add a column with the week calculated from clicks_created_at_datetime to be the monday of the week
                .assign(clicks_created_at_date=lambda x: x['clicks_created_at_datetime'].dt.to_period('d').dt.start_time)
                .groupby(['clicks_created_at_date'])[['google_trends_weekly_DK', 'orders_cancelled']] 
                .agg(cancellation_rate=('orders_cancelled', 'mean'), google_trends=('google_trends_weekly_DK', 'mean'))
                .reset_index())
cancellations_currencies_daily.head()
# %%

# import from pickle a file data/raw/changes_dict.pickle
import pickle
with open('data/raw/changes_dict.pickle', 'rb') as handle:
    changes_dict = pickle.load(handle)
    
# filter changes_dict.keys() by starting from "payment"
# payment_changes_dict = [[k, v['date'], v['description']] for k, v in changes_dict.items() if k.startswith('pay')]
# layout_changes_dict = [[k, v['date'], v['description']] for k, v in changes_dict.items() if k.startswith('layout')]
# engine_changes_dict = [[k, v['date'], v['description']] for k, v in changes_dict.items() if k.startswith('engine')]
all_changes_dict = [[k, v['date'], v['description']] for k, v in changes_dict.items()]

# payment_changes_df = pd.DataFrame(payment_changes_dict, columns=['change', 'date', 'description']).sort_values('date')
# layout_changes_df = pd.DataFrame(layout_changes_dict, columns=['change', 'date', 'description']).sort_values('date')
# engine_changes_df = pd.DataFrame(engine_changes_dict, columns=['change', 'date', 'description']).sort_values('date')
# changes_df = pd.concat([payment_changes_df, layout_changes_df, engine_changes_df]).sort_values('date')

all_changes_df = pd.DataFrame(all_changes_dict, columns=['change', 'date', 'description']).sort_values('date')
# filter  if all_changes_df['change'] does not starts from "engine"
all_changes_df = all_changes_df[~all_changes_df['change'].str.startswith('engine')]

all_changes_df = all_changes_df[all_changes_df['change'].str.startswith('payment')|all_changes_df['change'].str.startswith('_pl')]

# add column how many days to the next change
all_changes_df['days_to_next'] = (all_changes_df['date'].shift(-1) - all_changes_df['date']).dt.days
all_changes_df['days_to_previous'] = (all_changes_df['date'] - all_changes_df['date'].shift(1)).dt.days
# new column time_window is the min of days_to_next and days_to_previous
all_changes_df['time_window'] = all_changes_df[['days_to_next', 'days_to_previous']].min(axis=1)
all_changes_df['time_window'] = 10
all_changes_df = all_changes_df.drop(['days_to_next', 'days_to_previous'], axis=1)


# create treatments_dict from all_changes_df where key is the change
treatments_dict = {}

for index, row in all_changes_df.iterrows():
    change_key = row['change']
    treatments_dict[change_key] = {
        'date': row['date'],
        'description': row['description'],
        'time_window': row['time_window'],
    }

all_changes_df

# print(all_changes_df.to_latex(index=False,
#                   formatters={"name": str.upper},
#                   float_format="{:.1f}".format,
# ))  

# %% Visualize the conversion rate by date with labels
# --------------------
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# set font to comply with default in Latex
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# plot conv_rate_skk_weekly by date
fig, ax = plt.subplots(figsize=(15, 8))
fig.set_facecolor('white')
ax.set_facecolor('white')

# start a y axis from 0 with 1% interval
ax.set_ylim(bottom=0, top=0.20)
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

# plot the conversion rate and google trends for each currency with separate line plot
line1 = ax.plot(
    cancellations_currencies_weekly['clicks_created_at_week'], 
    cancellations_currencies_weekly['cancellation_rate'],
    # cancellations_currencies_daily['clicks_created_at_date'],
    # cancellations_currencies_daily['cancellation_rate'],
                label='Payment cancellation Rate',
                color='black')
# create a twin axis for google trends
ax2 = ax.twinx()
line2 = ax2.plot(
    cancellations_currencies_weekly['clicks_created_at_week'], 
    cancellations_currencies_weekly['google_trends'],
    # cancellations_currencies_daily['clicks_created_at_date'],
    # cancellations_currencies_daily['cancellation_rate'],
                 color='green', 
                 label='Google Trends')
# combine line plots for conversion rate and google trends
lines = line1 + line2
# get labels for each line plot
labels = [l.get_label() for l in lines]
# add legend for both line plots
leg = ax.legend(lines, labels, loc='upper left', facecolor='lightgrey', edgecolor='black')
for text in leg.get_texts():
    plt.setp(text, color='black')

# set x-axis tick interval to show each month
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

# add vertical line for each element in search_engine_changes
for change in treatments_dict:
    ax.axvline(treatments_dict[change]['date'], color='lightgrey', linestyle='--')

# add a label for each treatments_dict.keys()
for change in treatments_dict:
    ax.text(treatments_dict[change]['date'], 0.002, change, rotation=90, color='black')

# set axis lines color to black
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')

# set tick labels color to black and change y-axis tick labels to percentages
ax.tick_params(axis='both', colors='black')
ax2.tick_params(axis='both', colors='black')

# set title and axis labels
ax.set_title('Weekly Conversion Rates for Danish market', color='black')
ax.set_xlabel('Date', color='black')
ax.set_ylabel('Conversion Rate', color='black')

# set y-axis label for google trends
ax2.set_ylabel('Google Trends Index', color='black')
# set limits for google trends y-axis
ax2.set_ylim(bottom=0)

plt.show()



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


#%% Regressor ------------------------------- 
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
# -------------------------------


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

# %% DoubleML apprfine tuning
# -------------------------------
# Calculate the time difference between each observation and the target timestamp
# time_diff = abs(x_train['clicks_created_at_datetime'] - search_engine_changes['mobile.pay.support.Denmark'])

# Create a weight vector that assigns higher weights to observations that are closer to the target timestamp
# Calculate weights using the apply method with a lambda function
# weights = time_diff.apply(lambda x: np.exp(-x.total_seconds() / 1000))

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM


# modelling_df check what is percentage of NAs
# modelling_df.isna().sum()/len(modelling_df)

treatment = "layout_remove_luckyorange"

X_train = modelling_df_dict[treatment]['X_train']
y_train = modelling_df_dict[treatment]['y_train']
treat_train = modelling_df_dict[treatment]['treat_train']
treatment_var_name = 'treatment_' + treatment
# split df into training and test sets
df_doubleml = DoubleMLData(pd.concat([X_train, y_train, treat_train], axis=1), x_cols=X_train.columns.values.tolist(), y_col=y_col, d_cols=treatment_var_name)

from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# hide warnings
import warnings
warnings.filterwarnings('ignore')


from sklift.metrics import ate_auc_score
# import rfoost classifier and regressor
from xgboost import XGBClassifier, XGBRegressor
# declare sample rfoost regressor with sample parameters
xgb_regressor = XGBRegressor(**rf_regressor_params)
# declare sample rfoost classifier with sample parameters
xgb_classifier = XGBClassifier(**xgb_classifier_params)


# fit the DoubleMLPLR model
# dml_plr = DoubleMLPLR(
#     df_doubleml, 
#     ml_l=xgb_classifier,
#     ml_m=xgb_classifier,
#     n_folds=5 
#     )
# dml_plr_fit=dml_plr.fit()
# dml_plr_fit.summary


# fit the DoubleMLIRM model
dml_irm = DoubleMLIRM(
    df_doubleml, 
    dml_procedure = 'dml2',    
    ml_g=xgb_classifier,
    ml_m=xgb_classifier,
    n_folds=5 
    )
dml_irm_fit=dml_irm.fit()
dml_irm_fit.summary
# estimate the treatment effect using the DoubleMLPLR model

dml_predictions = pd.Series([i[0][0] for i in dml_plr_fit.predictions['ml_l']])

y_train
# plot histogram of predictions from dmlpredictions


# Creating the histogram
plt.hist(dml_predictions, bins=100)
plt.hist(y_train, bins = 100)
plt.title('Histogram of Data')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Displaying the histogram
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Applying the sigmoid function to the data
sigmoid_data = dml_predictions.apply(sigmoid)

# Creating the histogram
plt.hist(sigmoid_data, bins='auto')
plt.title('Histogram of Sigmoid Transformed Data')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Displaying the histogram
plt.show()

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
# %% Average treatment effects modelling 
# -------------------------------------
# %% Average treatment effects modelling 
# -------------------------------------
model_names = ["IPW", "CausalForest", "DoubleMlPLR1", "DoubleMLPLR2"]

import causallib
import pandas as pd
import numpy as np
from datetime import datetime
# from dowhy import CausalModel
from econml.dml import CausalForestDML

def ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test, 
                 classifier, classifier_params, regressor, regressor_params, 
                 selected_features, y_col, d_col):
    
    # add time measurement 
    start_time = datetime.now()

    models_results = {
        'approach': [],
        'ate': []
    }
    from causallib.estimation import IPW
    
    # Inverse Probability Weighting (IPW)
    ipw = IPW(
        learner=classifier(**classifier_params)
    )
    ipw.fit(X_train, treat_train)
    ipw_outcomes = ipw.estimate_population_outcome(X_train, treat_train, y_train)
    ipw_score = ipw.estimate_effect(ipw_outcomes[1], ipw_outcomes[0], effect_types=["diff"])['diff']

    models_results['approach'].append('IPW')
    models_results['ate'].append(ipw_score)

    # # Outcome Regression (OR)
    # or_model = causallib.estimation.LinearOutcomeRegression(
    #     model_regression=regressor(**regressor_params), 
    #     fit_cate_intercept=True
    # )
    # or_model.fit(X_train, treat_train, y_train)
    # or_score = or_model.estimate_ate(X_test, treat_test, y_test)

    # models_results['approach'].append('OR')
    # models_results['ate'].append(or_score)

    # from causallib.estimation import DoublyRobust
    # # Doubly Robust (DR)
    # dr = DoublyRobust(
    #     learner_regression=regressor(**regressor_params),
    #     learner_propensity=classifier(**classifier_params),
    #     fit_cate_intercept=True
    # )


    from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # split df into training and test sets
    df_doubleml = DoubleMLData(pd.concat([X_train, y_train, treat_train], axis=1), x_cols=selected_features, y_col=y_col, d_cols=d_col)

    # #  DoubleML 1
    dml_plr_1 = DoubleMLPLR(df_doubleml, 
                          ml_l=regressor(**regressor_params), 
                          ml_m=classifier(**classifier_params),
                        #   ml_m=regressor(**regressor_params),  
                          dml_procedure='dml1',
                          n_folds=5
                          )
    dml_plr_1 = dml_plr_1.fit(store_predictions = True, store_models = True)
    dml_plr_1_score = dml_plr_1.summary['coef'].values[0]

    models_results['approach'].append('DoubleMPLR1')
    models_results['ate'].append(dml_plr_1_score)
    # DOuble ML 2
    dml_plr_2 = DoubleMLPLR(df_doubleml, 
                          ml_l=regressor(**regressor_params), 
                        #   ml_m=classifier(**classifier_params),
                          ml_m=classifier(**classifier_params), 
                          dml_procedure='dml2',
                          n_folds=5
                          )
    dml_plr_2=dml_plr_2.fit(store_predictions = True, store_models = True)
    dml_plr_2_score = dml_plr_2.summary['coef'].values[0]

    models_results['approach'].append('DoubleMPLR2')
    models_results['ate'].append(dml_plr_2_score)

    from econml.dml import CausalForestDML
    from sklearn.linear_model import LassoCV

    # set parameters for causal forest 
    causal_forest = CausalForestDML(criterion='het', 
                                    n_estimators=10000,       
                                    min_samples_leaf=10, 
                                    max_samples=0.5,
                                    max_features=15,
                                    max_depth=15,
                                    discrete_treatment=True,
                                    honest=True,
                                    inference=True,
                                    cv=5,
                                    model_t=classifier(**classifier_params),
                                    model_y=classifier(**classifier_params),
                                    )
                        
    # fit train data to causal forest model 
    causal_forest = causal_forest.fit(y_train, treat_train, X=X_train, W=None)
    # estimate the CATE with the test set 
    causal_forest_score = causal_forest.ate(X_test)

    models_results['approach'].append('CausalForest')
    models_results['ate'].append(causal_forest_score)

    #  DoubleML 1
    # dml_irm_1 = DoubleMLIRM(df_doubleml, 
    #                       ml_g=classifier(**classifier_params), 
    #                       ml_m=classifier(**classifier_params), 
    #                       dml_procedure='dml1',
    #                       n_folds=5
    #                       )
    # dml_irm_1 = dml_irm_1.fit(store_predictions = True, store_models = True)
    # dml_irm_1_score = dml_irm_1.summary['coef'].values[0]

    # models_results['approach'].append('DoubleMLIRM1')
    # models_results['ate'].append(dml_irm_1_score)
    # # DOuble ML 2
    # dml_irm_2 = DoubleMLIRM(df_doubleml, 
    #                       ml_g=classifier(**classifier_params), 
    #                       ml_m=classifier(**classifier_params), 
    #                       dml_procedure='dml2',
    #                       n_folds=5
    #                       )
    # dml_irm_2=dml_irm_2.fit(store_predictions = True, store_models = True)
    # dml_irm_2_score = dml_irm_2.summary['coef'].values[0]

    # models_results['approach'].append('DoubleML2')
    # models_results['ate'].append(dml_irm_2_score)
    # # print("Double ML done")

    # print timedifference
    print('Time taken: {}'.format(datetime.now() - start_time), )

    return pd.DataFrame(models_results), {
        'IPW': ipw, 
        "CausalForest": causal_forest,
        # 'OR': or_model, 
        # 'DoublyRobust': dr_model, 
        'DoubleMlPLR1': dml_plr_1, 
        'DoubleMLPLR2': dml_plr_2, 
        # 'DoubleMLIRM1': dml_irm_1, 
        # 'DoubleMLIRM2': dml_irm_2
    }

# %% Splitting the data into training and test sets with filter by search engine changes
# -------------------------------
# read pickle file search_engine_changes
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def treatment_variable_generation(modelling_df, dates_dictionary, dates_dictionary_key, time_window):
    modelling_df_filter = modelling_df.copy()
    del modelling_df
    time_window = int(time_window)
    month_before = dates_dictionary[dates_dictionary_key]['date'] + timedelta(days=-time_window)
    month_after = dates_dictionary[dates_dictionary_key]['date'] + timedelta(days=time_window)

    def check_date_range(date):
        if month_before < date <= dates_dictionary[dates_dictionary_key]['date']:
            return 0
        elif dates_dictionary[dates_dictionary_key]['date'] < date <= month_after:
            return 1
        else:
            return None
    treatment_var_name = "treatment_"+dates_dictionary_key
    modelling_df_filter['clicks_created_at_datetime'] = pd.to_datetime(modelling_df_filter['clicks_created_at_datetime'])
    modelling_df_filter[treatment_var_name] = modelling_df_filter['clicks_created_at_datetime'].apply(check_date_range)
    # print(modelling_df_filter.shape)
    # filter modelling if treatment variable is not null
    modelling_df_filter = modelling_df_filter[modelling_df_filter[treatment_var_name].notnull()]
    # print(modelling_df_filter.shape)
    # count na values for each column
    modelling_df_filter = modelling_df_filter.dropna()
    # print(modelling_df_filter.isna().sum())
    return modelling_df_filter, treatment_var_name


test_set_is_train_set = True

modelling_df_dict = {}
treatments = treatments_dict.keys()
for treatment in treatments:
    print(treatment)
    modelling_df, treatment_var_name = treatment_variable_generation(modelling_df_full, treatments_dict, treatment, time_window = 10
                                                                    #  treatments_dict[treatment]['time_window']
                                                                     )
    
    # train test split
    if test_set_is_train_set:
        X_train, y_train, treat_train = modelling_df[selected_features], modelling_df[y_col], modelling_df[treatment_var_name]
        X_test = X_train; y_test = y_train; treat_test = treat_train
    else:
        X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[selected_features], modelling_df[y_col], modelling_df[treatment_var_name], test_size=0.8, random_state=42)

    from sklearn.preprocessing import QuantileTransformer
    import pandas as pd

    # Define the columns you want to scale
    columns_to_scale = ['clicks_itinerary_sales_price_pax',  'clicks_itinerary_segment_count',  'clicks_itinerary_totaldistance', 'clicks_itinerary_travel_timehours', 'clicks_passengers_count','ratio_sales_price_travel_time','ratio_distance_passenger', 'ratio_travel_time_distance', 'ratio_sales_price_distance', 'carriers_marketing_ratings_count', 'carriers_marketing_ratings_mean', 'ratio_sales_price_carrier_rating_max', 'ratio_sales_price_carrier_rating_count',  'clicks_itinerary_sales_price_diff_cheapest', 'clicks_itinerary_sales_price_diff_best', 'clicks_itinerary_sales_price_diff_fastest', 
                        'google_trends_weekly_DK', 'google_trends_weekly_DK_lag_7', 'ratio_clicks_google_trends', 'diff_clicks_google_trends', 'google_trends_smooth_DK', 'google_trends_smooth_DK_lag_1', 'google_trends_smooth_DK_lag_2', 'google_trends_smooth_DK_lag_3', 'google_trends_smooth_DK_lag_4', 'google_trends_smooth_DK_lag_5', 'google_trends_smooth_DK_lag_6', 'google_trends_smooth_DK_lag_7',
    'orders_count', 'diff_clicks_google_trends', 'orders_addon_totalsum', 'orders_search_data.search_parameters.children', 'orders_search_data.search_parameters.infants', 'orders_search_data.search_parameters.adults', 
    # 'google_trends_smooth_DK_lag_1', 'google_trends_smooth_DK_lag_2', 'google_trends_smooth_DK_lag_3', 'google_trends_smooth_DK_lag_4', 'google_trends_smooth_DK_lag_5', 'google_trends_smooth_DK_lag_6', 'google_trends_smooth_DK_lag_7',  'google_trends_smooth_DK',
    ]
    # Define the transformer
    transformer = QuantileTransformer(n_quantiles=400, output_distribution='uniform', random_state=0)

    # Fit on training set only.
    transformer.fit(X_train[columns_to_scale])

    # Apply transform to both the training set and the test set.
    X_train[columns_to_scale] = transformer.transform(X_train[columns_to_scale])
    X_test[columns_to_scale] = transformer.transform(X_test[columns_to_scale])


    
    X_train['clicks_created_at_datetime_hour'] = np.where(X_train['clicks_created_at_datetime_hour'] <= 6, "0 - 6", 
                                                          np.where(X_train['clicks_created_at_datetime_hour'] <= 12,"7 - 12", 
                                                                   np.where(X_train['clicks_created_at_datetime_hour'] <= 18, "13 - 18",  
                                                                            "19 - 24")) 
                                                          )
    if test_set_is_train_set:
        X_test = X_train

    categorical_features = ['clicks_created_at_datetime_weekday', 'clicks_created_at_datetime_hour', 'orders_order_type',"orders_search_data.gate_name"]

    # X_train[categorical_features] = X_train[categorical_features].astype('category')
    # X_test[categorical_features] = X_test[categorical_features].astype('category')

    # make one hot encoding for categorical features
    X_train = pd.get_dummies(X_train.copy(), columns=categorical_features).copy()
    X_test = pd.get_dummies(X_test.copy(), columns=categorical_features).copy()
    print("Test sample size: ", y_test.shape[0])
    print("Negative_class", y_test.value_counts()[1]/y_test.shape)
    modelling_df_dict[treatment] =  { 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'treat_train':treat_train, 'treat_test':treat_test }

# for treatment in search_engine_changes.keys():
#     print(treatment, ": ", modelling_df_dict[treatment]['X_train'].shape)
# %% XGBoost in all the models
xgb_modelling_outputs={}
treatments = list(treatments_dict.keys())
for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    # X, y, treat = modelling_df_dict[treatment]['X'], modelling_df_dict[treatment]['y'], modelling_df_dict[treatment]['treat']
    X_train, X_test, y_train , y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']

    # import xgb classifier and regressor
    from xgboost import XGBClassifier, XGBRegressor

    xgb_classifier_params = { "n_estimators":100, "max_depth":10, "min_child_weight":2, "learning_rate":0.001, #"subsample":0.8, 
                              "objective":'binary:logistic', "random_state":42,
                              "eval_metric":'auc', "disable_default_eval_metric":1,
                            #   "enable_categorical":True, "feature_types":['c' if i in categorical_features else 'q' for i in X_train.columns.values]
                              }
    xgb_regressor_params = { "n_estimators":100, "max_depth":10, "min_child_weight":2, "learning_rate":0.001, #"subsample":0.8, 
                             "objective":'reg:squarederror', "random_state":42, 
                            #  "enable_categorical":True, "feature_types":['c' if i in categorical_features else 'q' for i in X_train.columns.values]
                            }

    models_results, models_objects = ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=XGBClassifier, classifier_params=xgb_classifier_params,
                                            regressor=XGBRegressor, regressor_params=xgb_regressor_params,
                                            selected_features=X_train.columns.values.tolist(), y_col=y_col, d_col=treatment_var_name)
    modelling_outputs_dict = {
        "results_df":models_results,
        "models_obj":models_objects
    }
    xgb_modelling_outputs[treatment] = modelling_outputs_dict
print("Learning finished")

# save xgb_modelling_outputs to pickle
import pickle
import time
# add current date to file name
# with open(f'reports/xgboost/modelling_outputs_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
#     pickle.dump(xgb_modelling_outputs, f)

#%%

xgb_ates = pd.concat([xgb_modelling_outputs[treatment]['results_df']['ate'] for treatment in treatments], axis = 1)
model_names = [i for i in model_names if i not in ['OR','DoublyRobust',]]
xgb_ates.index = model_names
xgb_ates.columns = treatments

[pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_"+treatment).agg("mean").T for treatment in treatments]

cancellation_rates= []
for treatment in treatments:
    cancellation_rate = pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_" + treatment).agg("mean").T
    cancellation_rate.index = [treatment]
    cancellation_rate.columns = ["before_change", "after_change"]
    # calculate difference rate between before and after change
    cancellation_rate["cancellation_rates_diff"] = np.abs(cancellation_rate["after_change"] - cancellation_rate["before_change"])
    cancellation_rates.append(cancellation_rate)

cancellation_rates = pd.concat(cancellation_rates, axis = 0)

time_windows_df = pd.DataFrame({
    "time_window" : [treatments_dict[treatment]['time_window'] for treatment in treatments],
    "test_sample_size" : [modelling_df_dict[treatment]['y_test'].shape[0] for treatment in treatments],
    "cancellation_rate_diff" : [cancellation_rates.loc[treatment]["cancellation_rates_diff"] for treatment in treatments],

    }, index = treatments)

xgb_ates = pd.concat([xgb_ates.T,time_windows_df], axis = 1,  )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

red_white_green = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.0, "green"),
        (0.5, "white"),
        (1.0, "red"),
    ],
)

cols_to_show = [i for i in xgb_ates.columns if i not in [ 
    # "time_window",
    'SoloModel',
    'ClassTransform', 
    'TwoModelTrmnt',
    'TwoModelCtrl',
    # 'DoubleMLIRM1',	
    # 'DoubleMLIRM2'
    ]]

# xgb_ates['ModelsAverage'] = xgb_ates[models_subset].mean(axis=1)
# models_subset = models_subset + ['ModelsAverage']
xgb_ates.sort_index()[cols_to_show].style.background_gradient(cmap=red_white_green, vmin=-0.1, vmax=0.1, low=0, high=0, axis=None, subset=model_names).format("{:.4f}")


# %%
print(xgb_ates)
xgb_ates.to_clipboard()
# %% Digging deeper into Double ML
xgb_ates_doubleML1 = pd.concat([xgb_modelling_outputs[treatment]['models_obj']['DoubleMLPLR2'].summary for treatment in treatments], axis=0  )
xgb_ates_doubleML1.index = np.char.replace(xgb_ates_doubleML1.index.values.tolist(), 'treatment_', '')
xgb_doubleml_summaries = pd.concat([xgb_ates_doubleML1, time_windows_df], axis = 1,  )
xgb_doubleml_summaries[['coef', 'P>|t|', "cancellation_rate_diff"]].sort_values('P>|t|', ascending=False).style.background_gradient(
    cmap='RdYlGn', axis=0, subset=['coef',"cancellation_rate_diff"]
)
#%%
xgb_doubleml_summaries[['coef', 'P>|t|', "time_window"]].to_clipboard()

# copy xgb_doubleml_summaries[['coef', 'P>|t|', "time_window"]] to clipboard

# %% Histograms of the data

# Extracting the data
data = pd.Series([i[0][0] for i in xgb_modelling_outputs['payment_mobile.pay.support.Denmark']['models_obj']['DoubleMLIRM1'].predictions['ml_g0']])

# Creating the histogram
plt.hist(data, bins=100)
plt.title('Histogram of Data')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Displaying the histogram
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Applying the sigmoid function to the data
sigmoid_data = data.apply(sigmoid)

# Creating the histogram
plt.hist(sigmoid_data, bins='auto')
plt.title('Histogram of Sigmoid Transformed Data')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Displaying the histogram
plt.show()

# %% Feature importance
from xgboost import plot_importance
from matplotlib import pyplot
from matplotlib.pyplot import figure
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 10, 10
# plot feature importance
for treatment in treatments:
    plot_importance(xgb_modelling_outputs[treatment]['models_obj']['TwoModelTrmnt']['estimator_trmnt']['treatment_'+treatment][0], max_num_features=20, importance_type='gain')
    pyplot.show()


#%% p adjusting for Double ML

xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML1'].summary
xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML2'].score


xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML'].p_adjust(method='bonferroni')
xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML'].p_adjust(method='holm')
xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML'].dml_procedure

xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML2'].summary
plot_feature_importances(xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML2'].models['ml_l']['treatment_layout_remove_luckyorange'][0][3], 
                         modelling_df_dict['layout_remove_luckyorange']['X_train'])

#%% Plotting the Qini curve

from sklift.viz import plot_qini_curve

for treatment in sorted(treatments):
    y_true = modelling_df_dict[treatment]['y_test']
    ate_predicted = xgb_modelling_outputs[treatment]['models_obj']['TwoModelTrmnt'].predict(modelling_df_dict[treatment]['X_test'])
    trmnt_test = modelling_df_dict[treatment]['treat_test']

    qini_disp = plot_qini_curve(
        y_true,ate_predicted, trmnt_test,
        perfect=True, name=treatment
    )

    qini_disp.figure_.suptitle("Qini curve")

#%% Plotting the Uplift curve

from sklift.viz import plot_ate_curve

for treatment in sorted(treatments):
    y_true = modelling_df_dict[treatment]['y_test']
    ate_predicted = xgb_modelling_outputs[treatment]['models_obj']['TwoModelTrmnt'].predict(modelling_df_dict[treatment]['X_test'])
    trmnt_test = modelling_df_dict[treatment]['treat_test']

    qini_disp = plot_ate_curve(
        y_true,ate_predicted, trmnt_test,
        perfect=True, name= treatment
    )

    qini_disp.figure_.suptitle(treatment + " Uplift curve")

# %% LightGBM in all the models
# import LGBM classifier and regressor 

lgb_modelling_outputs={}

for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    # X, y, treat = modelling_df_dict[treatment]['X'], modelling_df_dict[treatment]['y'], modelling_df_dict[treatment]['treat']
    X_train, X_test, y_train, y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']

    # import LGBM classifier and regressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    
    lgb_classifier_params = {
            "n_estimators":10,
            "max_depth":10,
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
    
    models_results, models_objects = ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=LGBMClassifier, classifier_params=lgb_classifier_params,
                                            regressor=LGBMRegressor, regressor_params=lgb_regressor_params,
                                            selected_features=X_train.columns.values.tolist(), y_col=y_col, d_col=treatment_var_name)
    modelling_outputs_dict = {
        "results_df":models_results,
        "models_obj":models_objects
    }
    lgb_modelling_outputs[treatment] = modelling_outputs_dict
print("Learning finished")


# %% RandomForrest in all the models
# import RandomForest classifier and regressor

rf_modelling_outputs={}


for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    X_train, X_test, y_train, y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']
    
    # import rf classifier and regressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    rf_regressor_params = { "n_estimators":100, "max_depth":10, "min_samples_leaf":2, "max_features":20 }
    rf_classifier_params = { "n_estimators":100, "max_depth":10, "min_samples_leaf":2, "max_features":20 }
    
    models_results, models_objects = ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=RandomForestClassifier, classifier_params=rf_regressor_params,
                                            regressor=RandomForestRegressor, regressor_params=rf_classifier_params,
                                            selected_features=X_train.columns.values.tolist(), y_col=y_col, d_col=treatment_var_name)
    modelling_outputs_dict = {
        "results_df":models_results,
        "models_obj":models_objects
    }
    rf_modelling_outputs[treatment] = modelling_outputs_dict
print("Learning finished")

# %%
rf_ates = pd.concat([rf_modelling_outputs[treatment]['results_df']['ate'] for treatment in treatments], axis = 1)
rf_ates.index = model_names
rf_ates.columns = treatments
time_windows_df = pd.DataFrame({"time_window" : [treatments_dict[treatment]['time_window'] for treatment in treatments]}, index = treatments)

[pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_"+treatment).agg("mean").T for treatment in treatments]

cancellation_rates= []
for treatment in treatments:
    cancellation_rate = pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_" + treatment).agg("mean").T
    cancellation_rate.index = [treatment]
    cancellation_rate.columns = ["before_change", "after_change"]
    # calculate difference rate between before and after change
    cancellation_rate["cancellation_rates_diff"] = np.abs(cancellation_rate["after_change"] - cancellation_rate["before_change"])
    cancellation_rates.append(cancellation_rate)

cancellation_rates = pd.concat(cancellation_rates, axis = 0)

time_windows_df = pd.DataFrame({
    "time_window" : [treatments_dict[treatment]['time_window'] for treatment in treatments],
    "test_sample_size" : [modelling_df_dict[treatment]['y_test'].shape[0] for treatment in treatments],
    "cancellation_rate_diff" : [cancellation_rates.loc[treatment]["cancellation_rates_diff"] for treatment in treatments],

    }, index = treatments)

rf_ates = pd.concat([rf_ates.T,time_windows_df], axis = 1,  )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

red_white_green = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.0, "green"),
        (0.5, "white"),
        (1.0, "red"),
    ],
)

cols_to_show = [i for i in rf_ates.columns if i not in [ 
    'SoloModel',
    'ClassTransform', 
    'TwoModelTrmnt',
    'TwoModelCtrl',
    # 'DoubleMLIRM1',	
    # 'DoubleMLIRM2'
    ]]

# xgb_ates['ModelsAverage'] = xgb_ates[models_subset].mean(axis=1)
# models_subset = models_subset + ['ModelsAverage']
rf_ates.sort_index()[cols_to_show].style.background_gradient(cmap=red_white_green, vmin=-0.1, vmax=0.1, low=0, high=0, axis=None, subset=['DoubleMlPLR1', 'DoubleMLPLR2', 'DoubleMLIRM1', 'DoubleMLIRM2',]).format("{:.4f}")

# %% Digging deeper into Double ML

rf_doubleml_summaries = pd.concat([rf_modelling_outputs[treatment]['models_obj']['DoubleML'].summary for treatment in treatments], axis=0  )
rf_doubleml_summaries['P>|t|'] = rf_doubleml_summaries['P>|t|'].apply(lambda x: "{:.6f}".format(x))
rf_doubleml_summaries

# %% Logistic Regression and Linear Regression in all the models
lr_modelling_outputs={}
treatments = list(treatments_dict.keys())
for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    # X, y, treat = modelling_df_dict[treatment]['X'], modelling_df_dict[treatment]['y'], modelling_df_dict[treatment]['treat']
    X_train, X_test, y_train , y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']

    # import LogisticRegression and LinearRegression
    from sklearn.linear_model import LogisticRegression, LinearRegression

    lr_classifier_params = {"random_state":42, "solver":'lbfgs'}
    lr_regressor_params = {}

    models_results, models_objects = ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=LogisticRegression, classifier_params=lr_classifier_params,
                                            regressor=LinearRegression, regressor_params=lr_regressor_params,
                                            selected_features=X_train.columns.values.tolist(), y_col=y_col, d_col=treatment_var_name)
    modelling_outputs_dict = {
        "results_df":models_results,
        "models_obj":models_objects
    }
    lr_modelling_outputs[treatment] = modelling_outputs_dict
print("Learning finished")


# %%

lr_ates = pd.concat([lr_modelling_outputs[treatment]['results_df']['ate'] for treatment in treatments], axis = 1)
model_names = [i for i in model_names if i not in ['OR','DoublyRobust',]]
lr_ates.index = model_names
lr_ates.columns = treatments

[pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_"+treatment).agg("mean").T for treatment in treatments]

cancellation_rates= []
for treatment in treatments:
    cancellation_rate = pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_" + treatment).agg("mean").T
    cancellation_rate.index = [treatment]
    cancellation_rate.columns = ["before_change", "after_change"]
    # calculate difference rate between before and after change
    cancellation_rate["cancellation_rates_diff"] = np.abs(cancellation_rate["after_change"] - cancellation_rate["before_change"])
    cancellation_rates.append(cancellation_rate)

cancellation_rates = pd.concat(cancellation_rates, axis = 0)

time_windows_df = pd.DataFrame({
    "time_window" : [treatments_dict[treatment]['time_window'] for treatment in treatments],
    "test_sample_size" : [modelling_df_dict[treatment]['y_test'].shape[0] for treatment in treatments],
    "cancellation_rate_diff" : [cancellation_rates.loc[treatment]["cancellation_rates_diff"] for treatment in treatments],

    }, index = treatments)

lr_ates = pd.concat([lr_ates.T,time_windows_df], axis = 1,  )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

red_white_green = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.0, "green"),
        (0.5, "white"),
        (1.0, "red"),
    ],
)

cols_to_show = [i for i in xgb_ates.columns if i not in [ 
    # "time_window",
    'SoloModel',
    'ClassTransform', 
    'TwoModelTrmnt',
    'TwoModelCtrl',
    # 'DoubleMLIRM1',	
    # 'DoubleMLIRM2'
    ]]

# xgb_ates['ModelsAverage'] = xgb_ates[models_subset].mean(axis=1)
# models_subset = models_subset + ['ModelsAverage']
lr_ates.sort_index()[cols_to_show].style.background_gradient(cmap=red_white_green, vmin=-0.1, vmax=0.1, low=0, high=0, axis=None, subset=model_names).format("{:.4f}")



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
filename = "/saved_workspaces/workspace_2023-04-18_21-41-55.pkl"
with open(os.getcwd() + filename, "rb") as f:
    dill.load_session(f)

# Workspace variables are now available in this script
print("Workspace loaded.")

# %%
ates_average_time_df = (all_changes_df
 .set_index('change')
 .merge(xgb_ates['ModelsAverage'],
         left_index=True, right_index=True, how='left')
 .sort_values('date', ascending=False)
 )
# %%
ates_average_time_df['start_interval'] = ates_average_time_df['date'] - pd.to_timedelta(ates_average_time_df['time_window'], unit='D')
ates_average_time_df['end_interval'] = ates_average_time_df['date'] + pd.to_timedelta(ates_average_time_df['time_window'], unit='D')
import matplotlib.pyplot as plt

# Assume "df" is your DataFrame with columns "date", "end_interval", "coef", "google_trends", and "clicks_created_at_week"
fig, ax = plt.subplots(figsize=(10, 5))

# Create the bar plot using the "date", "end_interval", and "coef" columns
ax.bar(ates_average_time_df['date'], ates_average_time_df['ModelsAverage'], width=ates_average_time_df['end_interval'] - ates_average_time_df['date'], align='edge')

# Create a line plot for the "google_trends" data using a twin axis
# line1 = ax.plot(conv_rates_currencies_weekly['clicks_created_at_week'], conv_rates_currencies_weekly['conv_rate'], label='Conversion Rate')
# ax2 = ax.twinx()
# line2 = ax2.plot(conv_rates_currencies_weekly['clicks_created_at_week'], conv_rates_currencies_weekly['google_trends'], color='green', label='Google Trends')

line1 = ax.plot(conv_rates_currencies_daily['clicks_created_at_date'], orders_count_google_trends['orders_count']/100)
# ax2 = ax.twinx()
line2 = ax.plot(conv_rates_currencies_daily['clicks_created_at_date'], orders_count_google_trends['ratio_clicks_google_trends'], color='green')


# Set the y-axis limits for the twin axis to 0 and 100
ax2.set_ylim([0, 100])

# Combine the line plots for "Conversion Rate" and "Google Trends"
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left')

# Set the x-axis label to "Date"
ax.set_xlabel('Date')

# Set the y-axis label to "Coefficient"
ax.set_ylabel('Coefficient')

# Set the title of the plot
ax.set_title('Coefficient vs. Date')

# Display the plot
plt.show()

# %%# draw a plot of the data and the least squares line
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# Plot the responses for different events and regions
sns.lineplot(x="date", y="coef",
                hue="change",
                data=ates_average_time_df)
# %%
# Plot the responses for different events and regions
import seaborn as sns


