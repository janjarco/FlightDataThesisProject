# %% [markdown]
# ### Import libraries

# %%
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
# change default font to latex font
plt.rcParams['font.family'] = 'serif'




#set working directory to this from the workspace
os.chdir('/Users/janjarco/Programming/PrivateRepository/FlightDataThesisProject')

from src.data.filter_columns_by_regex import filter_columns_by_regex

#list all the files in current working directory
print(os.listdir("data"))
# %%
# read all pickles with modelling_dataset in the name into a dictionary
# modelling_datasets = {}
# for file in os.listdir("data/processed"):
#     if "modelling_dataset_clicks_" in file:
#         modelling_datasets[file] = pd.read_pickle(f"data/processed/{file}")
# new_keys = ["EUR", "DKK", "SEK", "NOK"]

# modelling_datasets={new_keys[i]: modelling_datasets[j] for i, j in enumerate(modelling_datasets)}


# load df

modelling_df_full = modelling_df_full = pd.read_pickle("data/processed/modelling_dataset_clicks_DK.pkl")

modelling_df_full['clicks_created_at_datetime'] = pd.to_datetime(modelling_df_full['clicks_created_at_datetime'])
modelling_df_full['clicks_created_at_datetime_weekday'] = modelling_df_full['clicks_created_at_datetime'].dt.dayofweek + 1
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
clicks_count = pd.DataFrame(modelling_df_full
 .groupby('clicks_created_at_datetime_date')['clicks_created_at_datetime_date']
 .count().reset_index(name='clicks_count'))

# clicks_count group by week and summarize the sum of clicks_count for weach week
# create a column with a first day of the week (monday) on the basis of date clicks_created_at_datetime_date
clicks_count['clicks_created_at_datetime_date'] = pd.to_datetime(clicks_count['clicks_created_at_datetime_date'])
# clicks_count['clicks_created_at_datetime_week'] = clicks_count['clicks_created_at_datetime_date'] - pd.to_timedelta(clicks_count['clicks_created_at_datetime_date'].dt.dayofweek, unit='d')

google_trends = pd.read_csv("data/external/google_trends_interest.csv")
google_trends['date'] = pd.to_datetime(google_trends['date'])
# google_trends['week'] = google_trends['date'] - pd.to_timedelta(google_trends['date'].dt.dayofweek, unit='d')
# google_trends = google_trends.groupby('week').mean().reset_index()

clicks_count_google_trends = clicks_count.merge(google_trends, how='inner', left_on='clicks_created_at_datetime_date', right_on='date')

clicks_count_google_trends = clicks_count_google_trends[['clicks_created_at_datetime_date', 'clicks_count', 'google_trends_smooth_DK']]

# transform clicks_count and google_trends_smooth_DK with min max scaling
clicks_count_google_trends['clicks_count'] = clicks_count_google_trends['clicks_count'] / clicks_count_google_trends['clicks_count'].max() * 100
clicks_count_google_trends['google_trends_smooth_DK'] = clicks_count_google_trends['google_trends_smooth_DK'] / clicks_count_google_trends['google_trends_smooth_DK'].max() * 100
clicks_count_google_trends['ratio_clicks_google_trends'] = clicks_count_google_trends['clicks_count'] / clicks_count_google_trends['google_trends_smooth_DK']
clicks_count_google_trends['diff_clicks_google_trends'] = clicks_count_google_trends['clicks_count'] - clicks_count_google_trends['google_trends_smooth_DK']

# add a clicks_count lag from the fay before
clicks_count_google_trends['clicks_count_lag1'] = clicks_count_google_trends['clicks_count'].shift(1)
clicks_count_google_trends = clicks_count_google_trends.bfill()
# make this to string clicks_count_google_trends['clicks_created_at_datetime_date'] 
# clicks_count_google_trends['clicks_created_at_datetime_date'] = clicks_count_google_trends['clicks_created_at_datetime_date'].astype(str)
modelling_df_full['clicks_created_at_datetime_date'] = pd.to_datetime(modelling_df_full['clicks_created_at_datetime_date'])
clicks_count_google_trends['clicks_created_at_datetime_date'] = pd.to_datetime(clicks_count_google_trends['clicks_created_at_datetime_date'])
# left join clicks_count_google_trends to modelling_df_full on clicks_created_at_datetime_date
modelling_df_full = modelling_df_full.merge(clicks_count_google_trends.drop("google_trends_smooth_DK", axis=1), how='left', left_on='clicks_created_at_datetime_date', right_on='clicks_created_at_datetime_date', suffixes=('', ''))

# modelling_df_full summarize number of missing values per column and order ascending
modelling_df_full.isnull().sum().sort_values(ascending=False)
# modelling_df_full = (modelling_df_full.merge(clicks_count_google_trends, how='left', left_on='clicks_created_at_datetime_date', right_on='clicks_created_at_datetime_date'))

x_cols = [
    'clicks_created_at_datetime', 
    'clicks_created_at_datetime_hour', 
    # 'clicks_created_at_datetime_weekend', 
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
    'clicks_itinerary_sales_price_diff_cheapest',
    'clicks_itinerary_sales_price_diff_best',
    'clicks_itinerary_sales_price_diff_fastest',
    # 'clicks_itinerary_sales_price_category',
    # 'clicks_created_at_datetime_weekday',
    # new features
    'clicks_count',
    # 'clicks_count_lag1',
    # 'google_trends_reduced_DK',
    # 'google_trends_weekly_DK',
    # 'google_trends_weekly_DK_lag_7',
    # 'ratio_clicks_google_trends',
    # 'diff_clicks_google_trends',
    # 'google_trends_smooth_DK',
    # 'google_trends_smooth_DK_lag_1',
    # 'google_trends_smooth_DK_lag_2',
    # 'google_trends_smooth_DK_lag_3',
    # 'google_trends_smooth_DK_lag_4',
    # 'google_trends_smooth_DK_lag_5',
    # 'google_trends_smooth_DK_lag_6',
    # 'google_trends_smooth_DK_lag_7',
    # 'google_trends_weekly',
    # 'google_trends_smooth', #old feature with new naming
    # 'interaction_google_trends_smooth_weekday', # interaction I dunno if it makes sense
    # 'interaction_google_trends_weekly_weekday',
    ]

# x_cols = x_cols + filter_columns_by_regex(modelling_df_full, "google.*_DK")
y_col = 'orders_if_order'
modelling_df_full[['clicks_created_at_datetime', 'clicks_created_at_datetime_weekday']].tail(50)

# drop duplicated columns
modelling_df_full = modelling_df_full.loc[:,~modelling_df_full.columns.duplicated()]

# remove missing values from modelling_df_full
no_rows_before_drop = modelling_df_full.shape[0]
modelling_df_full = modelling_df_full.dropna()
print(modelling_df_full.shape[0] /no_rows_before_drop)
# modelling_df, treatment_var_name = treatment_variable_generation(modelling_datasets['DKK'], search_engine_changes, "hack.bagprice.override.Altea.FLX", 30, 30)

# %% normalizing the data
 # --------------------
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Assume X_train and X_test are your training and testing dataframes

# Define the columns you want to scale
columns_to_scale = [
    'clicks_itinerary_sales_price_pax', 
    'clicks_itinerary_segment_count',  
    'clicks_itinerary_totaldistance', 
    'clicks_itinerary_travel_timehours', 
    'clicks_passengers_count',
    'ratio_sales_price_travel_time',
    'ratio_distance_passenger', 
    'ratio_travel_time_distance', 
    'ratio_sales_price_distance', 
    'carriers_marketing_ratings_count', 
    'carriers_marketing_ratings_mean', 
    'ratio_sales_price_carrier_rating_max', 
    'ratio_sales_price_carrier_rating_count',  
    'clicks_itinerary_sales_price_diff_cheapest', 
    'clicks_itinerary_sales_price_diff_best', 
    'clicks_itinerary_sales_price_diff_fastest', 
    # 'google_trends_smooth_DK',
# 'google_trends_smooth_DK_lag_1', 'google_trends_smooth_DK_lag_2', 'google_trends_smooth_DK_lag_3', 'google_trends_smooth_DK_lag_4', 'google_trends_smooth_DK_lag_5', 'google_trends_smooth_DK_lag_6', 'google_trends_smooth_DK_lag_7',
]
# Define the transformer to normal distribution
# transformer = StandardScaler()
transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=0)

# Fit on training set only.
transformer.fit(modelling_df_full[columns_to_scale])

modelling_df_full[columns_to_scale] = transformer.transform(modelling_df_full[columns_to_scale])

time_related_features = [
    'clicks_count',
    'google_trends_weekly_DK', 
    'google_trends_weekly_DK_lag_7', 
    'ratio_clicks_google_trends', 
    'diff_clicks_google_trends',
    ]

for feature in time_related_features:
    modelling_df_full[feature] = pd.qcut(modelling_df_full[feature], q=10, labels=False, duplicates='drop')


import pandas as pd

# Define the time intervals
time_intervals = {
    (0, 6): '0 - 6',
    (7, 12): '7 - 12',
    (13, 18): '13 - 18',
    (19, 24): '19 - 24'
}

# Create a function to map the numerical values to time intervals
def map_to_time_interval(hour):
    for interval, label in time_intervals.items():
        if interval[0] <= float(hour) <= interval[1]:
            return label
        
modelling_df_full['clicks_created_at_datetime_hour'] = modelling_df_full['clicks_created_at_datetime_hour'].apply(map_to_time_interval)

# Apply the mapping function to the 'clicks_created_at_datetime_hour' colum
# %%-----------------------------
# to exclude features with high correlation b etween each other
feature_to_exclude = [
    'carriers_marketing_ratings_max',
    'carriers_marketing_ratings_min',
    'clicks_created_at_datetime', 
    'ratio_sales_price_carrier_rating_avg', 
    'ratio_sales_price_carrier_rating_min',
    # 'google_trends_reduced_DK',
    ] #+ filter_columns_by_regex(X_train, 'weekly|smooth')

# Select features that meet the threshold
# selected_features = shap_df_update.iloc[:cutoff_index+1]['feature'].tolist()
selected_features = [i for i in x_cols if i not in feature_to_exclude+["orders_if_order"]]

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
    import plotly.figure_factory as ff
    import plotly.io as pio
    import numpy as np
    import seaborn as sns

    # Create a correlation matrix
    corr_matrix = df[sorted(df.columns.values)].corr()

    # Create a mask to display only the lower triangle of the matrix
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_tril = corr_matrix.where(mask)

    # Create a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Create a custom color scale with 0 as white
    colorscale = ff.create_annotated_heatmap(
        z=corr_matrix_tril.values, colorscale= [[0, 'blue'], [0.5, 'white'], [1.0, 'red']], zmin=-1, zmax=1
    ).data[0].colorscale
    # Create a heatmap without annotations
    fig = go.Figure(go.Heatmap(
        z=corr_matrix_tril.values,
        x=list(corr_matrix_tril.columns),
        y=list(corr_matrix_tril.index),
        colorscale=colorscale,
        # display values in the heatmap
        text=corr_matrix_tril.values.round(2),
        hovertemplate='(%{y}, %{x}): %{z:.2f}<extra></extra>'
    ))

    # Customize the layout
    fig.update_layout(
        title=dict(text='Correlation Heatmap', x=0.5, xanchor='center', font=dict(size=24, color='black', family='Times New Roman, Times, serif')),
        xaxis=dict(title='Features', side='bottom', ticks='outside', title_font=dict(size=18, color='black', family='Times New Roman, Times, serif'), tickfont=dict(size=14, color='black', family='Times New Roman, Times, serif')),
        yaxis=dict(title='Features', ticks='outside', autorange='reversed', title_font=dict(size=18, color='black', family='Times New Roman, Times, serif'), tickfont=dict(size=14, color='black', family='Times New Roman, Times, serif')),
        margin=dict(t=50, l=50, b=100),
        width=1200,
        height=1200
    )

    fig.show()

    # Export the plot to a PNG file
    # pio.write_image(fig, 'Correlation matrix.png')

# [i for i in X_train.columns.values not in filter_columns_by_regex(X_train, "weekday|hour")]
plot_correlation_matrix(X_train[sorted(X_train.columns.values)])


# plot histogram of google_trends in X_train
# X_train['interaction_google_trends_weekday'].hist(bins = 100)
# X_train['clicks_created_at_datetime_weekday'].hist(bins = 10

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
# %%
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[selected_features], modelling_df[y_col], modelling_df["mobile.pay.support.Denmark"],
                                                                             test_size=0.2, random_state=42)

# %% conv_rates_currencies_weekly
conv_rates_currencies_weekly = (modelling_df_full
                         # add a column with the week calculated from clicks_created_at_datetime to be the monday of the week
                .assign(clicks_created_at_week=lambda x: x['clicks_created_at_datetime'].dt.to_period('W').dt.start_time)
                # add google_trends variable and orders_if_order groupby clicks_created_at_week
                .groupby(['clicks_created_at_week'])[['google_trends_weekly_DK', 'orders_if_order']] 
                .agg(orders=('orders_if_order', 'sum'), clicks=('orders_if_order', 'count'), google_trends=('google_trends_weekly_DK', 'mean'))
                .assign(conv_rate=lambda x: x['orders'] / x['clicks'])
                .reset_index())
conv_rates_currencies_weekly.head()

# %% conv_rates_currencies_daily
conv_rates_currencies_daily = (modelling_df_full
                         # add a column with the week calculated from clicks_created_at_datetime to be the monday of the week
                .assign(clicks_created_at_date=lambda x: x['clicks_created_at_datetime'].dt.to_period('d').dt.start_time)
                .groupby(['clicks_created_at_date'])['orders_if_order']
                .agg(orders=('sum'), clicks=('count'))
                .assign(conv_rate=lambda x: x['orders'] / x['clicks'])
                .reset_index())
conv_rates_currencies_daily.head()
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

# add column how many days to the next change
all_changes_df['days_to_next'] = (all_changes_df['date'].shift(-1) - all_changes_df['date']).dt.days
all_changes_df['days_to_previous'] = (all_changes_df['date'] - all_changes_df['date'].shift(1)).dt.days
# new column time_window is the min of days_to_next and days_to_previous
all_changes_df['time_window'] = all_changes_df[['days_to_next', 'days_to_previous']].min(axis=1)
# limit time_window to max 14 days
# all_changes_df['time_window'] = all_changes_df['time_window'].apply(lambda x: min(x, 5))
all_changes_df = all_changes_df.drop(['days_to_next', 'days_to_previous'], axis=1)

# all_changes_df = all_changes_df[all_changes_df['change'].str.startswith('placebo')]
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

# remove upper line in the frame of a chart
ax.spines['top'].set_visible(False)

# start a y axis from 0 with 1% interval
ax.set_ylim(bottom=0, top=1)
# ax.yaxis.set_major_locator(mtick.MultipleLocator(0.01))
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=2))

# plot the conversion rate and google trends for each currency with separate line plot
line1 = ax.plot(conv_rates_currencies_weekly['clicks_created_at_week'], 
                conv_rates_currencies_weekly['conv_rate'] / max(conv_rates_currencies_weekly['conv_rate']),
                label='Conversion Rate Index',
                color='black')

# combine line plots for conversion rate and google trends
lines = line1
# get labels for each line plot
labels = [l.get_label() for l in lines]
# add legend for both line plots
leg = ax.legend(lines, labels, loc='upper left', facecolor='white', edgecolor='black')
for text in leg.get_texts():
    plt.setp(text, color='black')

# set x-axis tick interval to show each month
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))

def placebo_actual_color(treatment):
    if treatment.startswith('placebo'):
        return "grey"
    else:
        return "black"

# add vertical line for each element in search_engine_changes
for change in treatments_dict:
    ax.axvline(treatments_dict[change]['date'], color=placebo_actual_color(change), linestyle='--')

# add a label for each treatments_dict.keys()
for change in treatments_dict:
    ax.text(treatments_dict[change]['date'], 0.02, change + " (" +str(int(treatments_dict[change]['time_window'])) + " days)", rotation=90, color=placebo_actual_color(change))

# set axis lines color to black
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')

# set tick labels color to black and change y-axis tick labels to percentages
ax.tick_params(axis='both', colors='black')

# Increase font size for the title, x-axis label, and y-axis label
title_font = {'fontsize': 18, 'fontweight': 'bold'}
x_label_font = {'fontsize': 16, }
y_label_font = {'fontsize': 16, }

# set title and axis labels with increased font size
ax.set_title('Weekly Conversion Rates for Danish market', fontdict=title_font, color='black')
ax.set_xlabel('Date', fontdict=x_label_font, color='black')
ax.set_ylabel('Conversion Rate Index', fontdict=y_label_font, color='black')

# set resolution forthe plot to make it readable in Latex
plt.savefig('plots/conv_rates_currencies_weekly.png', dpi=300, bbox_inches='tight')

plt.show()

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
# Calculate the time differences between each observation and the target timestamp
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


from sklift.metrics import uplift_auc_score
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



# %% Splitting the data into training and test sets with filter by search engine changes
# -------------------------------
# read pickle file search_engine_changes
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

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


# %% Splitting the data into training and test sets with filter by treatment
# --------------------

modelling_df_dict = {}
treatments = treatments_dict.keys()
for treatment in treatments:
    print(treatment)
    modelling_df, treatment_var_name = treatment_variable_generation(modelling_df_full, treatments_dict, treatment, time_window = treatments_dict[treatment]['time_window'])
    # print(modelling_df.columns.values)
    # train test split
    # X, y, treat = modelling_df[selected_features], modelling_df[y_col], modelling_df[treatment_var_name]
    if test_set_is_train_set:
        X_train, y_train, treat_train = modelling_df[selected_features], modelling_df[y_col], modelling_df[treatment_var_name]
        X_test = X_train; y_test = y_train; treat_test = treat_train
    else:
        X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(modelling_df[selected_features], modelling_df[y_col], modelling_df[treatment_var_name], test_size=0.2, random_state=42)
    
    X_train = pd.get_dummies(X_train, columns=['clicks_created_at_datetime_hour'])
    X_test = pd.get_dummies(X_test, columns=['clicks_created_at_datetime_hour'])
                                                          
    modelling_df_dict[treatment] =  { 'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test, 'treat_train':treat_train, 'treat_test':treat_test }

# for treatment in search_engine_changes.keys():
#     print(treatment, ": ", modelling_df_dict[treatment]['X_train'].shape)

# %% CHecking the means differencess between treatment and control groups

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats
from matplotlib.colors import LinearSegmentedColormap

p_values = {}

for treatment in treatments:
    # Extract the data for this treatment group
    X_train = modelling_df_dict[treatment]['X_train']#[columns_to_scale]
    treat_train = modelling_df_dict[treatment]['treat_train']

    # Split X_train into two groups based on the treatment variable
    X_ctrl = X_train[treat_train == 0]
    X_treat = X_train[treat_train == 1]

    # Compute the means for each variable
    means_ctrl = X_ctrl.mean(axis=0)
    means_treat = X_treat.mean(axis=0)

    # Compute the standard deviations
    std_ctrl = X_ctrl.std(axis=0)
    std_treat = X_treat.std(axis=0)

    # Compute the t-statistic and p-value
    t_stat, p_value = ttest_ind_from_stats(means_ctrl, std_ctrl, len(X_ctrl),
                                            means_treat, std_treat, len(X_treat),
                                            equal_var=False)

    # Store the p-value in the dictionary
    p_values[treatment] = p_value

# Convert the dictionary into a DataFrame
p_values = pd.DataFrame(p_values, index=X_train.columns)

red_green_significance = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.0, "green"),
        (0.5, "yellow"),
        (1, "red"),
    ],
)
# rotate column names by 90 degrees
# filter treatments in treatments list that
actual_treatments = [treatment for treatment in treatments if  "placebo" not in treatment]

p_values[actual_treatments].style.background_gradient(cmap=red_green_significance, vmin=-0.0, vmax=0.1, low=0, high=0, axis=None).format("{:.3f}")

#%%
def significance_level(p_value):
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    elif p_value < 0.1:
        sig = "."
    else:
        sig = ""
    return f"{p_value:.3f} {sig}"

# Assuming df is your DataFrame with p-values
df_significance = p_values[actual_treatments].applymap(significance_level)
print(df_significance.to_latex(index=True, escape=False))
# in p_values[actual_treatments] return a table with string of values and levels of significance in starts to scientific paper

# %%

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats

y_means = {}

for treatment in treatments:
    # Extract the data for this treatment group
    y_train = modelling_df_dict[treatment]['y_train']
    treat_train = modelling_df_dict[treatment]['treat_train']

    # Split X_train into two groups based on the treatment variable
    y_ctrl = y_train[treat_train == 0]
    y_treat = y_train[treat_train == 1]

    # Compute the means for each variable
    means_ctrl = y_ctrl.mean(axis=0)
    means_treat = y_treat.mean(axis=0)

    # Compute the standard deviations
    std_ctrl = y_ctrl.std(axis=0)
    std_treat = y_treat.std(axis=0)

    # Compute the t-statistic and p-value
    t_stat, p_value = ttest_ind_from_stats(means_ctrl, std_ctrl, len(y_ctrl),
                                            means_treat, std_treat, len(y_treat),
                                            equal_var=False)

    # Store the p-value in the dictionary
    y_means[treatment] = [means_ctrl, means_treat, means_treat-means_ctrl, p_value]

# Convert the dictionary into a DataFrame
y_means = pd.DataFrame(y_means, index=["means_ctrl", "means_treat", "means_diff","means_p_value"]).T
y_means

# %% Average treatment effects modelling function
# -------------------------------------
model_names = [
    "LinearRegression",
    # "LogisticRegression",
    # "PLR",
    # "IPW", 
    # "CausalForest", 
    # "DoubleMLPLR1",
    "DoubleMLPLR2",
    # "DoubleMLIRM1",
    # "DoubleMLIRM2",
               ]

import causallib
import pandas as pd
import numpy as np
from datetime import datetime
# from dowhy import CausalModel
from econml.dml import CausalForestDML

def ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test, 
                 classifier, classifier_params, regressor, regressor_params, 
                 selected_features, y_col, d_col, model_names = model_names):
    
    # add time measurement 
    start_time = datetime.now()

    models_results = {
        'approach': [],
        'ate': []
    }
    models_objects = {}
    if "LinearRegression" in model_names:
        # import OLS 
        import statsmodels.api as sm

        input = sm.add_constant(
            pd.concat([X_train, treat_train], axis=1)
            )

        # Fit the model
        lin_reg = sm.OLS(y_train.to_numpy(), input).fit()

        lin_reg_ate = lin_reg.params[d_col]
        
        # Store the results
        models_results['approach'].append('LinearRegression')
        models_results['ate'].append(lin_reg_ate)

        # Store the model
        models_objects['LinearRegression'] = lin_reg


    if "LogisticRegression" in model_names:
        # import logistic regression
        from sklearn.linear_model import LogisticRegression

        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(pd.concat([X_train, treat_train], axis=1), y_train)

        # return coefficients and return coefficeints and return last value
        log_reg_ate = log_reg.coef_[0][-1]

        models_results['approach'].append('LogisticRegression')
        models_results['ate'].append(log_reg_ate)

        models_objects['LogisticRegression'] = log_reg

    if "PLR" in model_names:
        # Partially linear regression
        # import
        # Initialize the model
        import statsmodels.api as sm
        
        g_model = regressor(**rf_regressor_params)

        def g(X_train, y_train):
            # Fit the model
            g_model.fit(X_train,y_train)
        
            # Return the predicted values
            return g_model.predict(X_train)

        input = pd.DataFrame({
                "d":treat_train,
                "g":g(X_train, y_train)
                })
        input = sm.add_constant(input)

        # Fit the model
        plr = sm.OLS(y_train, input).fit()

        plr_ate = plr.params['d']
        
        models_results['approach'].append('PLR')
        models_results['ate'].append(plr_ate)

        models_objects['PLR'] = plr

    if "IPW" in model_names:
        from causallib.estimation import IPW
        # Inverse Probability Weighting (IPW)
        ipw = IPW(
            learner=classifier(**classifier_params)
        )
        ipw.fit(X_train, treat_train)

        ipw_outcomes = ipw.estimate_population_outcome(X_train, treat_train, y_train)
        ipw_ate = ipw.estimate_effect(ipw_outcomes[1], ipw_outcomes[0], effect_types=["diff"])['diff']

        models_results['approach'].append('IPW')
        models_results['ate'].append(ipw_ate)

        models_objects['IPW'] = ipw

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

    # split df into training and test sets
    df_doubleml = DoubleMLData(pd.concat([X_train, y_train, treat_train], axis=1), x_cols=selected_features, y_col=y_col, d_cols=d_col)

    if "DoubleMLPLR1" in model_names:
        # #  DoubleML 1
        dml_plr_1 = DoubleMLPLR(df_doubleml, 
                            ml_l=regressor(**regressor_params), 
                            #   ml_m=regressor(**regressor_params),  
                            ml_m=classifier(**classifier_params),
                            dml_procedure='dml1',
                            n_folds=5
                            )
        dml_plr_1 = dml_plr_1.fit(store_predictions = True, store_models = True)
        dml_plr_1_ate = dml_plr_1.summary['coef'].values[0]

        models_results['approach'].append('DoubleMPLR1')
        models_results['ate'].append(dml_plr_1_ate)

        models_objects['DoubleMPLR1'] = dml_plr_1

    if "DoubleMLPLR2" in model_names:    
        # DOuble ML 2
        dml_plr_2 = DoubleMLPLR(df_doubleml, 
                            ml_l=regressor(**regressor_params), 
                            #   ml_m=regressor(**regressor_params),
                            ml_m=classifier(**classifier_params),
                            dml_procedure='dml2',
                            n_folds=5
                            )
        dml_plr_2=dml_plr_2.fit(store_predictions = True, store_models = True)
        dml_plr_2_ate = dml_plr_2.summary['coef'].values[0]

        models_results['approach'].append('DoubleMPLR2')
        models_results['ate'].append(dml_plr_2_ate)

        models_objects['DoubleMPLR2'] = dml_plr_2

    # from econml.dml import CausalForestDML
    # from sklearn.linear_model import LassoCV

    # set parameters for causal forest 
    if "CausalForest" in model_names:
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
                            
        # # fit train data to causal forest model 
        causal_forest = causal_forest.fit(y_train, treat_train, X=X_train, W=None)
        # estimate the CATE with the test set 
        causal_forest_ate = causal_forest.ate(X_test, y=y_test)

        models_results['approach'].append('CausalForest')
        models_results['ate'].append(causal_forest_ate)

    if "DoubleMLIRM1" in model_names:
        #  DoubleML IRM 1
        dml_irm_1 = DoubleMLIRM(df_doubleml, 
                            ml_g=classifier(**classifier_params), 
                            ml_m=classifier(**classifier_params), 
                            dml_procedure='dml1',
                            n_folds=5
                            )
        dml_irm_1 = dml_irm_1.fit(store_predictions = True, store_models = True)
        dml_irm_1_score = dml_irm_1.summary['coef'].values[0]

        models_results['approach'].append('DoubleMLIRM1')
        models_results['ate'].append(dml_irm_1_score)

        models_objects['DoubleMLIRM1'] = dml_irm_1

    # DOuble ML IRM 2
    if "DoubleMLIRM2" in model_names:
        dml_irm_2 = DoubleMLIRM(df_doubleml, 
                            ml_g=classifier(**classifier_params), 
                            ml_m=classifier(**classifier_params), 
                            dml_procedure='dml2',
                            n_folds=5
                            )
        dml_irm_2=dml_irm_2.fit(store_predictions = True, store_models = True)
        dml_irm_2_score = dml_irm_2.summary['coef'].values[0]

        models_results['approach'].append('DoubleML2')
        models_results['ate'].append(dml_irm_2_score)

        models_objects['DoubleMLIRM2'] = dml_irm_2
    # # print("Double ML done")

    # print timedifferences
    print('Time taken: {}'.format(datetime.now() - start_time), )

    return pd.DataFrame(models_results), models_objects

# %% XGBoost in all the models
xgb_modelling_outputs={}

xgb_classifier_params = { "n_estimators":100, "max_depth":8, "min_child_weight":2, "learning_rate":0.001, #"subsample":0.8, 
                            "objective":'binary:logistic', "random_state":42,
                            "eval_metric":"auc", #"disable_default_eval_metric":1,
                        #   "enable_categorical":True, "feature_types":['c' if i in categorical_features else 'q' for i in X_train.columns.values]
                            }
xgb_regressor_params = { "n_estimators":100, "max_depth":8, "min_child_weight":2, "learning_rate":0.01, #"subsample":0.8, 
                            "objective":'reg:squarederror', "random_state":42, 
                        #  "enable_categorical":True, "feature_types":['c' if i in categorical_features else 'q' for i in X_train.columns.values]
                        }

for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    # X, y, treat = modelling_df_dict[treatment]['X'], modelling_df_dict[treatment]['y'], modelling_df_dict[treatment]['treat']
    X_train, X_test, y_train, y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']

    # import xgb classifier and regressor
    from xgboost import XGBClassifier, XGBRegressor

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


# # check what's size of a variable
# import sys
import sys
print(sys.getsizeof(xgb_modelling_outputs))


#%% save xgb_modelling_outputs to pickle
import pickle
import time
# add current date to file name
with open(f'runs/xgboost/modelling_outputs_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
    pickle.dump(xgb_modelling_outputs, f)

#%% load xgb_modelling_outputs from pickle
# import pickle
with open('reports/xgboost/modelling_outputs_20230511_164000.pkl', 'rb') as f:
    xgb_modelling_outputs = pickle.load(f)

# check what's size of a variable



# %% XGB Average Treatment Effects 

xgb_ates = pd.concat([xgb_modelling_outputs[treatment]['results_df']['ate'] for treatment in treatments], axis = 1)
model_names = [i for i in model_names if i not in ['OR','DoublyRobust',]]
xgb_ates.index = model_names
xgb_ates.columns = treatments

[pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_"+treatment).agg("mean").T for treatment in treatments]

conversion_rates= []
for treatment in treatments:
    conversion_rate = pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_" + treatment).agg("mean").T
    conversion_rate.index = [treatment]
    conversion_rate.columns = ["before_change", "after_change"]
    # calculate differences rate between before and after change
    conversion_rate["conversion_rates_diff"] = np.abs(conversion_rate["after_change"] - conversion_rate["before_change"])
    conversion_rates.append(conversion_rate)

conversion_rates = pd.concat(conversion_rates, axis = 0)

time_windows_df = pd.DataFrame({
    "time_window" : [treatments_dict[treatment]['time_window'] for treatment in treatments],
    "test_sample_size" : [modelling_df_dict[treatment]['y_test'].shape[0] for treatment in treatments],
    "conversion_rate_diff" : [conversion_rates.loc[treatment]["conversion_rates_diff"] for treatment in treatments],

    }, index = treatments)

xgb_ates = pd.concat([xgb_ates.T,time_windows_df], axis = 1,  )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


red_white_green = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.0, "red"),
        (0.5, "white"),
        (1.0, "green"),
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
xgb_ates.sort_index()[cols_to_show].style.background_gradient(cmap=red_white_green, vmin=-0.1, vmax=0.1, low=0, high=0, axis=None, subset=model_names).format("{:.4f}")

# %%
# xgb_ates['ModelsAverage'] = xgb_ates[models_subset].mean(axis=1)
# models_subset = models_subset + ['ModelsAverage']

xgb_ates.sort_index()['CausalForest'].plot(kind='bar', figsize=(10, 6), color='blue', alpha=0.5, align='center')
plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
plt.xlabel('Model')
plt.ylabel('ATE')
plt.title('XGBoost ATEs')
plt.show()


# %%
print(xgb_uplifts)
xgb_uplifts.to_clipboard()
# %% Digging deeper into Double ML
xgb_uplifts_doubleML1 = pd.concat([xgb_modelling_outputs[treatment]['models_obj']['DoubleMLPLR2'].summary['P>|t|'] for treatment in treatments], axis=0  )
xgb_uplifts_doubleML1.index = np.char.replace(xgb_uplifts_doubleML1.index.values.tolist(), 'treatment_', '')
xgb_doubleml_summaries = pd.concat([xgb_uplifts_doubleML1, time_windows_df], axis = 1,  )
xgb_doubleml_summaries[['coef', 'P>|t|', "time_window"]].sort_values('P>|t|', ascending=False).style.background_gradient(
    cmap='RdYlGn', axis=0, subset=['coef',"time_window"]
)

xgb_doubleml_summaries[['coef', 'P>|t|', "time_window"]].to_clipboard()

# copy xgb_doubleml_summaries[['coef', 'P>|t|', "time_window"]] to clipboard

# %% Checkingoutputs from DML

# Extracting the data
data = pd.Series([i[0][0] for i in xgb_modelling_outputs['payment_mobile.pay.support.Denmark']['models_obj']['DoubleMLPLR2'].predictions['ml_l']])

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


#%% p adjusting for Double ML

xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML1'].summary
xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML2'].score


xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML'].p_adjust(method='bonferroni')
xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML'].p_adjust(method='holm')
xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML'].dml_procedure

xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML2'].summary
plot_feature_importances(xgb_modelling_outputs['layout_remove_luckyorange']['models_obj']['DoubleML2'].models['ml_l']['treatment_layout_remove_luckyorange'][0][3], 
                         modelling_df_dict['layout_remove_luckyorange']['X_train'])

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
    
    models_results, models_objects = uplift_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=LGBMClassifier, classifier_params=lgb_classifier_params,
                                            regressor=LGBMRegressor, regressor_params=lgb_regressor_params,
                                            selected_features=X_train.columns.values.tolist(), y_col=y_col, d_col=treatment_var_name)
    modelling_outputs_dict = {
        "results_df":models_results,
        "models_obj":models_objects
    }
    lgb_modelling_outputs[treatment] = modelling_outputs_dict
print("Learning finished")


# %% Testing overfitting with 5-fold cross-validation for random forest
# --------------------
# import rf regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

treatment = 'payment_mobile_support'

X_train = modelling_df_dict[treatment]['X_train']
y_train = modelling_df_dict[treatment]['y_train']

from sklearn.metrics import mean_squared_error
import pandas as pd

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Fold', 'Train MSE', 'Test MSE'])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_regressor_params = { "n_estimators":100, "max_depth":7, "min_samples_leaf":2, "max_features":20 }
rf_classifier_params = { "n_estimators":100, "max_depth":7, "min_samples_leaf":2, "max_features":20 }

fold = 0
for train_index, test_index in kf.split(X_train):
    fold += 1
    # Split the data into training/testing sets
    X_train_kf = X_train.iloc[train_index]
    y_train_kf = y_train.iloc[train_index]  
    
    X_test_kf = X_train.iloc[test_index]
    y_test_kf = y_train.iloc[test_index]

    rf = RandomForestRegressor(**rf_regressor_params)
    rf.fit(X_train_kf, y_train_kf)

    # Predict and calculate MSE for the training data
    train_pred = rf.predict(X_train_kf)
    train_mse = mean_squared_error(y_train_kf, train_pred)
    
    # Predict and calculate MSE for the test data
    test_pred = rf.predict(X_test_kf)
    test_mse = mean_squared_error(y_test_kf, test_pred)

    # Append the results to the DataFrame
    results = results.append({'Fold': 'Fold ' + str(fold), 'Train MSE': train_mse, 'Test MSE': test_mse}, ignore_index=True)

# Calculate the average MSE over all folds
average_train_mse = results['Train MSE'].mean()
average_test_mse = results['Test MSE'].mean()

# Append the averages to the DataFrame
results = results.append({'Fold': 'Average', 'Train MSE': average_train_mse, 'Test MSE': average_test_mse}, ignore_index=True)

# Display the results
print(results)







# %% Examininng the relation between covariates and treatment variable

# merging all the dataframes from modelling_df_dict[treatment]['X_train'] by treatment into one dataframe
X_train = pd.concat([modelling_df_dict[treatment]['X_train'] for treatment in modelling_df_dict.keys()])
# merging all the dataframes from modelling_df_dict[treatment]['y_train'] by treatment into one dataframe
y_train = pd.concat([modelling_df_dict[treatment]['y_train'] for treatment in modelling_df_dict.keys()])

# creating the model of Random Forest Regressor
# import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# creating the model of Random Forest Regressor
rf_model_all = RandomForestRegressor(**rf_regressor_params)

# fitting the model
rf_model_all.fit(X_train, y_train)
# xgb_model_fit = xgb_model.fit(modelling_df_dict[treatment]['X_train'], random_treat)
# %% RF all plot base variable importances 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl

# Assuming you have a fitted model 'rf_model_all'
# And the feature names stored in a variable 'feature_names'

# Calculating feature frequencies
feature_freq = np.zeros((rf_model_all.n_features_in_,))
for tree in rf_model_all.estimators_:
    feature_freq += np.bincount(tree.tree_.feature[tree.tree_.feature>=0], minlength=rf_model_all.n_features_in_)

# Normalize frequencies to [0, 1] range
feature_freq = feature_freq / rf_model_all.n_estimators

# Sorting features by frequency
sorted_indices = np.argsort(feature_freq)#[::-1]

# Set the font family to a LaTeX compliant font
mpl.rcParams['font.family'] = 'serif'

# Create the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 8))

# Set the background color to white
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plot the feature frequency
ax.barh(X_train.columns[sorted_indices], feature_freq[sorted_indices], color='blue')

# Remove the grid
ax.grid(False)

# Set the font color to black
ax.xaxis.label.set_color('black')
ax.yaxis.label.set_color('black')
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

# Adjust the height of the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Add title "Feature importance"
plt.title('Feature importance of covariates: Random Forrest Regressor', fontsize=16, color='black')

# y axis and x axis title font size to 16
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# change font of x axis title 
ax.set_xlabel('Frequency', fontsize=14, color='black')
# change y axis title font size 
ax.set_ylabel('Features', fontsize=14, color='black')

# Display the plot
plt.show()





# calculate roc_auc_score
roc_auc = roc_auc_score(modelling_df_dict[treatment]['treat_test'], y_pred_proba)
print("ROC_AUC: %.2f%%" % (roc_auc * 100.0))

#  calculate roc curve
fpr, tpr, thresholds = roc_curve(modelling_df_dict[treatment]['treat_test'], y_pred)

# calculate AUC
auc = auc(fpr, tpr)
print('AUC: %.3f' % auc)

# plot the roc curve for the model
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
plt.plot(fpr, tpr, marker='.', label='XGB')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
# show the plot
plt.show()

# %% Random Forrest in all the models
# import RandomForest classifier and regressor

rf_modelling_outputs={}

# change order of treatments to be __placebo_8 to be first
treatments

actual_treatments = [i for i in treatments if "placebo" not in i]
placebo_treatments = [i for i in treatments if "placebo" in i]

for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    X_train, X_test, y_train, y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']
    
    # import rf classifier and regressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    models_results, models_objects = ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=RandomForestClassifier, classifier_params=rf_classifier_params,
                                            regressor=RandomForestRegressor, regressor_params=rf_regressor_params,
                                            selected_features=X_train.columns.values.tolist(), y_col=y_col, d_col=treatment_var_name)
    modelling_outputs_dict = {
        "results_df":models_results,
        "models_obj":models_objects
    }
    rf_modelling_outputs[treatment] = modelling_outputs_dict
print("Learning finished")

#%% save xgb_modelling_outputs to pickle
import pickle
import time
# add current date to file name
with open(f'runs/rf/modelling_outputs_5days{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
    pickle.dump(rf_modelling_outputs, f)

# open pickle file
with open('runs/rf/modelling_outputs_20230519_173552.pkl', 'rb') as f:
    rf_modelling_outputs_alldays = pickle.load(f)

with open('runs/rf/modelling_outputs_5days20230524_135024.pkl', 'rb') as f:
    rf_modelling_outputs_5days = pickle.load(f)

with open('runs/rf/modelling_outputs_3day20230523_001113.pkl', 'rb') as f:
    rf_modelling_outputs_3days = pickle.load(f)

with open('runs/rf/modelling_outputs_1day20230522_234930.pkl', 'rb') as f:
    rf_modelling_outputs_1day = pickle.load(f)

# %% Linear regression in all the models
# import RandomForest classifier and regressor

lr_modelling_outputs={}

# change order of treatments to be __placebo_8 to be first
treatments

for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    X_train, X_test, y_train, y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']
    
    # import rf classifier and regressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
    models_results, models_objects = ate_modeling(X_train, y_train, X_test, y_test, treat_train, treat_test,
                                            classifier=RandomForestClassifier, classifier_params=rf_regressor_params,
                                            regressor=RandomForestRegressor, regressor_params=rf_classifier_params,
                                            selected_features=X_train.columns.values.tolist(), y_col=y_col, d_col=treatment_var_name)
    modelling_outputs_dict = {
        "results_df":models_results,
        "models_obj":models_objects
    }
    lr_modelling_outputs[treatment] = modelling_outputs_dict
print("Learning finished")

# %%
rf_ates = pd.DataFrame([rf_modelling_outputs_alldays[treatment]['results_df']['ate'][1] for treatment in treatments], index=treatments, columns=['DML PLR ATE'])

# For rf_modelling_outputs_alldays
DMLPLR_pvalue_alldays = pd.concat([rf_modelling_outputs_alldays[treatment]['models_obj']['DoubleMPLR2'].summary['P>|t|'] for treatment in treatments], axis=0)
DMLPLR_pvalue_alldays.index = treatments
rf_ates["DML PLR all days p-value"] = DMLPLR_pvalue_alldays

LR_ate = pd.Series([rf_modelling_outputs_alldays[treatment]['models_obj']['LinearRegression'].params[-1] for treatment in treatments])
LR_ate.index = treatments
LR_pvalue = pd.Series([rf_modelling_outputs_alldays[treatment]['models_obj']['LinearRegression'].pvalues[-1] for treatment in treatments])
LR_pvalue.index = treatments
rf_ates['OLS ATE'] = LR_ate
rf_ates["OLS p-value"] = LR_pvalue

DMLPLR_pvalue = pd.concat([rf_modelling_outputs_alldays[treatment]['models_obj']['DoubleMPLR2'].summary['P>|t|'] for treatment in treatments], axis=0)
DMLPLR_pvalue.index = treatments
rf_ates["DML PLR p-value"] = DMLPLR_pvalue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

red_white_green = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.0, "red"),
        (0.5, "white"),
        (1.0, "green"),
    ],
)
rf_ates.sort_index().style.background_gradient(cmap=red_white_green, vmin=-0.01, vmax=0.01, low=0, high=0, axis=None).format("{:.4f}")

# %% Digging deeper into Double ML
# import sm.OLS
import statsmodels.api as sm
rf_doubleml_summaries = pd.concat([rf_modelling_outputs[treatment]['models_obj']['DoubleMPLR2'].summary for treatment in treatments], axis=0  )
# rf_doubleml_summaries['P>|t|'] = rf_doubleml_summaries['P>|t|'].apply(lambda x: "{:.6f}".format(x))
rf_doubleml_summaries.sort_index().to_clipboard()

# from skleanr import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rf_doubleml_mse = [rf_modelling_outputs[treatment]['models_obj']['DoubleMPLR2'].evaluate_learners(metric = mean_squared_error) for treatment in treatments]
rf_doubleml_mae = [rf_modelling_outputs[treatment]['models_obj']['DoubleMPLR2'].evaluate_learners(metric = mean_absolute_error) for treatment in treatments]

[rf_modelling_outputs[treatment]['models_obj']['LinearRegression'] for treatment in treatments]

rf_doubleml_mse = [i['ml_l'][0][0] for i in rf_doubleml_mse]
rf_doubleml_mae = [i['ml_l'][0][0] for i in rf_doubleml_mae]


lr_mse = {}
for treatment in treatments:
    X_input = sm.add_constant(
            pd.concat([
                modelling_df_dict[treatment]['X_train'], 
                modelling_df_dict[treatment]['treat_train']], axis=1)
            )
    preds = rf_modelling_outputs[treatment]['models_obj']['LinearRegression'].predict(X_input)
    lr_mse[treatment] = mean_squared_error(modelling_df_dict[treatment]['y_train'], preds)

lr_mae = {}
for treatment in treatments:
    X_input = sm.add_constant(
            pd.concat([
                modelling_df_dict[treatment]['X_train'], 
                modelling_df_dict[treatment]['treat_train']], axis=1)
            )
    preds = rf_modelling_outputs[treatment]['models_obj']['LinearRegression'].predict(X_input)
    lr_mae[treatment] = mean_absolute_error(modelling_df_dict[treatment]['y_train'], preds)

# create a dataframe from lr_mse and change values name to mse
lr_mse_df = pd.DataFrame.from_dict(lr_mse, orient='index')
lr_mse_df.columns = ['OLS MSE']

# create a dataframe from lr_mae and change values name to mae
lr_mae_df = pd.DataFrame.from_dict(lr_mae, orient='index')
lr_mae_df.columns = ['OLS MAE']

# MERGE mse and mae dataframes
errors_models = pd.merge(lr_mse_df, lr_mae_df, left_index=True, right_index=True)

errors_models['DML PLR MSE'] = rf_doubleml_mse
errors_models['DML PLR MAE'] = rf_doubleml_mae

errors_models.sort_index()

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

# Loading workspace from saved_workspaces/workspace_name.pkl
import dill

# Load workspace from file
filename = "saved_workspaces\workspace_name.pkl"
with open(filename, "rb") as f:
    dill.load_session(f)

# Workspace variables are now available in this script
print("Workspace loaded.")
 
# %% Loading workspace from saved_workspaces/workspace_name.pkl
import dill

# Load workspace from file
filename = "/saved_workspaces/workspace_2023-04-18_21-41-55.pkl"
with open(os.getcwd() + filename, "rb") as f:
    dill.load_session(f)

# Workspace variables are now available in this script
print("Workspace loaded.")

# %% merge y_means with time_windows_df
conversion_rates= []
for treatment in treatments:
    conversion_rate = pd.concat([modelling_df_dict[treatment]['treat_test'], modelling_df_dict[treatment]['y_test']], axis = 1).groupby("treatment_" + treatment).agg("mean").T
    conversion_rate.index = [treatment]
    conversion_rate.columns = ["before_change", "after_change"]
    # calculate differences rate between before and after change
    conversion_rate["conversion_rates_diff"] = conversion_rate["after_change"] - conversion_rate["before_change"]
    conversion_rates.append(conversion_rate)

conversion_rates = pd.concat(conversion_rates, axis = 0)

time_windows_df = pd.DataFrame({
    "time_window" : [treatments_dict[treatment]['time_window'] for treatment in treatments],
    "test_sample_size" : [modelling_df_dict[treatment]['y_test'].shape[0] for treatment in treatments],
    }, index = treatments)

# %% Results table fro differences only
results_diff_table = pd.merge(y_means, time_windows_df, left_index=True, right_index=True)
results_diff_table.sort_index().style.background_gradient(cmap=red_white_green, vmin=-0.0, vmax=0.1, low=0, high=0, axis=None).format("{:.3f}")

table_order_index = [i for i in treatments if "placebo" not in i] + [i for i in treatments if "placebo" in i]
# replace "_placebo"with "placebo" in index
results_diff_table.time_window = results_diff_table.time_window.astype(int)
results_diff_table = results_diff_table.loc[table_order_index]
results_diff_table.index = results_diff_table.index.str.replace("_placebo", "placebo")

# in columns means_ctrl, means_treat, means_diff apply formatiing with percent 
# results_diff_table['means_p_value'] = results_diff_table.apply(lambda x: significance_level(x['means_p_value']), axis=1)
# results_diff_table_print['means_ctrl'] = results_diff_table_print['means_ctrl'].apply(lambda x: f"{x:.2%}")
# results_diff_table_print['means_treat'] = results_diff_table_print['means_treat'].apply(lambda x: f"{x:.2%}")
# results_diff_table_print['means_diff'] = results_diff_table_print['means_diff'].apply(lambda x: f"{x:.3}")

results_diff_table = results_diff_table.drop(columns=["means_ctrl", "means_treat"], inplace=False)
results_diff_table = results_diff_table.rename(columns={"means_diff": "Differences ATE", "means_p_value": "Differences p-value", "time_window": "Time window", "test_sample_size":"N"})
results_diff_table_print = results_diff_table.copy()
results_diff_table_print['Differences p-value'] = results_diff_table['Differences p-value'].apply(lambda x: significance_level(x))
results_diff_table_print['Differences ATE'] = results_diff_table_print['Differences ATE'].apply(lambda x: f"{100*x:.2f}")


results_diff_table_print.to_clipboard()


# %% Results table for all the models
# --------------------

# results_all_table = results_diff_table

# results_all_table = results_all_table.merge(rf_ates, left_index=True, right_index=True)

# # add column with placebo if "placevo in index else "real"
# results_all_table['Category'] =  ["Actual" if "placebo" not in x else "Placebo" for x in results_all_table.index]

# results_all_table = results_all_table[['Category'] + [i for i in results_all_table.columns if i not in ["Category", "N", "Time window"]] + ["N", "Time window"] ]

# results_all_table_print = results_all_table.copy()

# results_all_table_print['Differences ATE'] = results_all_table_print['Differences ATE'].apply(lambda x: f"{100*x:.2f}")
# results_all_table_print['OLS ATE'] = results_all_table_print['OLS ATE'].apply(lambda x: f"{100*x:.2f}")
# results_all_table_print['DML PLR ATE'] = results_all_table_print['DML PLR ATE'].apply(lambda x: f"{100*x:.2f}")


# results_all_table_print['Differences p-value'] = results_all_table_print['Differences p-value'].apply(lambda x: significance_level(x))
# results_all_table_print['OLS p-value'] = results_all_table_print['OLS p-value'].apply(lambda x: significance_level(x))
# results_all_table_print['DML PLR p-value'] = results_all_table_print['DML PLR p-value'].apply(lambda x: significance_level(x))
# # columns "Time window" and "N" to the end of the table

# results_all_table_print = results_all_table_print.drop(columns = ["Category"], axis=1)

# results_all_table_print = results_all_table_print[[
#     'Differences ATE', 'Differences p-value',
#     'OLS ATE', 'OLS p-value',
#     'DML PLR ATE','DML PLR p-value',
#       'N', 'Time window',
#         ]]
# model_names = ['Differences', 'OLS', 'DML PLR']
# sub_columns = ['ATE', 'p-value']
# multi_index = pd.MultiIndex.from_tuples([(model, sub_column) for model in model_names for sub_column in sub_columns] + [('N', ''), ('Time window', '')])
# results_all_table_print.columns = multi_index

# results_all_table_print.to_clipboard()

# %% Applying Holm's procedure to the results table All changes
# --------------------

import pandas as pd
from statsmodels.stats.multitest import multipletests

results_all_table = results_diff_table

results_all_table = results_all_table.merge(rf_ates, left_index=True, right_index=True)

# add column with placebo if "placevo in index else "real"
results_all_table['Category'] =  ["Actual" if "placebo" not in x else "Placebo" for x in results_all_table.index]

results_all_table = results_all_table[['Category'] + [i for i in results_all_table.columns if i not in ["Category", "N", "Time window"]] + ["N", "Time window"] ]

results_all_table_print = results_all_table.drop(columns = ["Category"], axis=1)
# assuming your DataFrame is named results_all_table
# results_all_table_actual = results_all_table[results_all_table['Category'] == 'Actual']
results_all_table['Category'] =  ["Actual" if "placebo" not in x else "Placebo" for x in results_all_table.index]
pvals_diff = results_all_table['Differences p-value'].values
pvals_dml_plr = results_all_table['DML PLR p-value'].values
pvals_ols = results_all_table['OLS p-value'].values

# Applying Holm's method
_, pvals_diff_holm, _, _ = multipletests(pvals_diff, alpha=0.1, method='holm')
_, pvals_dml_plr_holm, _, _ = multipletests(pvals_dml_plr, alpha=0.1, method='holm')
_, pvals_ols_holm, _, _ = multipletests(pvals_ols, alpha=0.1, method='holm')

# Creating new columns with Holm's corrected p-values
results_all_table['Differences Holm p-value'] = pvals_diff_holm.copy()
results_all_table['DML PLR Holm p-value'] = pvals_dml_plr_holm.copy()
results_all_table['OLS Holm p-value'] = pvals_ols_holm.copy()


results_all_table_print['Differences Holm p-value'] = pvals_diff_holm.copy()
results_all_table_print['DML PLR Holm p-value'] = pvals_dml_plr_holm.copy()
results_all_table_print['OLS Holm p-value'] = pvals_ols_holm.copy()

results_all_table_print['Differences ATE'] = results_all_table_print['Differences ATE'].apply(lambda x: f"{100*x:.2f}")
results_all_table_print['OLS ATE'] = results_all_table_print['OLS ATE'].apply(lambda x: f"{100*x:.2f}")
results_all_table_print['DML PLR ATE'] = results_all_table_print['DML PLR ATE'].apply(lambda x: f"{100*x:.2f}")

results_all_table_print['Differences p-value'] = results_all_table_print['Differences p-value'].apply(lambda x: significance_level(x))
results_all_table_print['OLS p-value'] = results_all_table_print['OLS p-value'].apply(lambda x: significance_level(x))
results_all_table_print['DML PLR p-value'] = results_all_table_print['DML PLR p-value'].apply(lambda x: significance_level(x))

results_all_table_print['Differences Holm p-value'] = results_all_table_print['Differences Holm p-value'].apply(lambda x: significance_level(x))
results_all_table_print['OLS Holm p-value'] = results_all_table_print['OLS Holm p-value'].apply(lambda x: significance_level(x))
results_all_table_print['DML PLR Holm p-value'] = results_all_table_print['DML PLR Holm p-value'].apply(lambda x: significance_level(x))


results_all_table_print = results_all_table_print[[
    'Differences ATE', 
    # 'Differences p-value', 
    'Differences Holm p-value',
    'OLS ATE', 
    # 'OLS p-value',
    'OLS Holm p-value',
    'DML PLR ATE',
    # 'DML PLR p-value',
    'DML PLR Holm p-value',
      'N', 'Time window',
        ]]

import pandas as pd

# Existing column names

# Define the multi-index
model_names = ['Differences', 'OLS', 'DML PLR']
sub_columns = ['ATE', 'Holm p-value']
multi_index = pd.MultiIndex.from_tuples([(model, sub_column) for model in model_names for sub_column in sub_columns] + [('N', ''), ('Time window', '')])

# Create a new DataFrame with the multi-index columns
results_all_table_print.columns = multi_index

# Display the updated DataFrame
results_all_table_print.to_clipboard()
# %% Applying Holm's procedure to the results table Actual changes
# --------------------

import pandas as pd
from statsmodels.stats.multitest import multipletests

# assuming your DataFrame is named results_all_table
results_all_table_actual = results_all_table[results_all_table['Category'] == 'Actual']
pvals_diff = results_all_table_actual['Differences p-value'].values
pvals_dml_plr = results_all_table_actual['DML PLR p-value'].values
pvals_ols = results_all_table_actual['OLS p-value'].values

# Applying Holm's method
_, pvals_diff_holm, _, _ = multipletests(pvals_diff, alpha=0.1, method='holm')
_, pvals_dml_plr_holm, _, _ = multipletests(pvals_dml_plr, alpha=0.1, method='holm')
_, pvals_ols_holm, _, _ = multipletests(pvals_ols, alpha=0.1, method='holm')

# Creating new columns with Holm's corrected p-values
results_all_table_actual['Differences Holm p-value'] = pvals_diff_holm.copy()
results_all_table_actual['DML PLR Holm p-value'] = pvals_dml_plr_holm.copy()
results_all_table_actual['OLS Holm p-value'] = pvals_ols_holm.copy()

results_all_table_actual['Differences ATE'] = results_all_table_actual['Differences ATE'].apply(lambda x: f"{100*x:.2f}")
results_all_table_actual['OLS ATE'] = results_all_table_actual['OLS ATE'].apply(lambda x: f"{100*x:.2f}")
results_all_table_actual['DML PLR ATE'] = results_all_table_actual['DML PLR ATE'].apply(lambda x: f"{100*x:.2f}")

results_all_table_actual['Differences p-value'] = results_all_table_actual['Differences p-value'].apply(lambda x: significance_level(x))
results_all_table_actual['OLS p-value'] = results_all_table_actual['OLS p-value'].apply(lambda x: significance_level(x))
results_all_table_actual['DML PLR p-value'] = results_all_table_actual['DML PLR p-value'].apply(lambda x: significance_level(x))

results_all_table_actual['Differences Holm p-value'] = results_all_table_actual['Differences Holm p-value'].apply(lambda x: significance_level(x))
results_all_table_actual['OLS Holm p-value'] = results_all_table_actual['OLS Holm p-value'].apply(lambda x: significance_level(x))
results_all_table_actual['DML PLR Holm p-value'] = results_all_table_actual['DML PLR Holm p-value'].apply(lambda x: significance_level(x))


results_all_table_actual = results_all_table_actual.drop(columns = ["Category"], axis=1)
results_all_table_actual = results_all_table_actual[[
    'Differences ATE', 
    # 'Differences p-value', 
    'Differences Holm p-value',
    'OLS ATE', 
    # 'OLS p-value',
    'OLS Holm p-value',
    'DML PLR ATE',
    # 'DML PLR p-value',
    'DML PLR Holm p-value',
      'N', 'Time window',
        ]]

import pandas as pd

# Existing column names

# Define the multi-index
model_names = ['Differences', 'OLS', 'DML PLR']
sub_columns = ['ATE', 'Holm p-value']
multi_index = pd.MultiIndex.from_tuples([(model, sub_column) for model in model_names for sub_column in sub_columns] + [('N', ''), ('Time window', '')])

# Create a new DataFrame with the multi-index columns
results_all_table_actual.columns = multi_index

# Display the updated DataFrame
results_all_table_actual.to_clipboard()

# %% Rf modelling outputs for different time windows
# --------------------
rf_ates_days = pd.DataFrame([rf_modelling_outputs_1day[treatment]['results_df']['ate'][1] for treatment in treatments], index=treatments, columns=['DML PLR 1 day ATE'])
rf_ates_days['DML PLR 3 days ATE'] = [rf_modelling_outputs_3days[treatment]['results_df']['ate'][1] for treatment in treatments]
rf_ates_days['DML PLR 5 days ATE'] = [rf_modelling_outputs_5days[treatment]['results_df']['ate'][1] for treatment in treatments]
rf_ates_days['DML PLR all days ATE'] = [rf_modelling_outputs_alldays[treatment]['results_df']['ate'][1] for treatment in treatments]

DMLPLR_pvalue_1day = pd.concat([rf_modelling_outputs_1day[treatment]['models_obj']['DoubleMPLR2'].summary['P>|t|'] for treatment in treatments], axis=0)
DMLPLR_pvalue_1day.index = treatments
rf_ates_days["DML PLR 1 day p-value"] = DMLPLR_pvalue_1day

# For rf_modelling_outputs_3days
DMLPLR_pvalue_3days = pd.concat([rf_modelling_outputs_3days[treatment]['models_obj']['DoubleMPLR2'].summary['P>|t|'] for treatment in treatments], axis=0)
DMLPLR_pvalue_3days.index = treatments
rf_ates_days["DML PLR 3 days p-value"] = DMLPLR_pvalue_3days

# For rf_modelling_outputs_5days
DMLPLR_pvalue_5days = pd.concat([rf_modelling_outputs_5days[treatment]['models_obj']['DoubleMPLR2'].summary['P>|t|'] for treatment in treatments], axis=0)
DMLPLR_pvalue_5days.index = treatments
rf_ates_days["DML PLR 5 days p-value"] = DMLPLR_pvalue_5days

# For rf_modelling_outputs_alldays
DMLPLR_pvalue_alldays = pd.concat([rf_modelling_outputs_alldays[treatment]['models_obj']['DoubleMPLR2'].summary['P>|t|'] for treatment in treatments], axis=0)
DMLPLR_pvalue_alldays.index = treatments
rf_ates_days["DML PLR all days p-value"] = DMLPLR_pvalue_alldays
# rf_ates_days = rf_ates.loc[actual_treatments]

from statsmodels.stats.multitest import multipletests

# Extracting the p-values for each time period
pvals_1day = rf_ates_days["DML PLR 1 day p-value"].values
pvals_3days = rf_ates_days["DML PLR 3 days p-value"].values
pvals_5days = rf_ates_days["DML PLR 5 days p-value"].values
pvals_alldays = rf_ates_days["DML PLR all days p-value"].values

# Applying Holm's method to each
_, pvals_1day_holm, _, _ = multipletests(pvals_1day, alpha=0.1, method='holm')
_, pvals_3days_holm, _, _ = multipletests(pvals_3days, alpha=0.1, method='holm')
_, pvals_5days_holm, _, _ = multipletests(pvals_5days, alpha=0.1, method='holm')
_, pvals_alldays_holm, _, _ = multipletests(pvals_alldays, alpha=0.1, method='holm')

# Adding Holm's corrected p-values to the dataframe
rf_ates_days['DML PLR 1 day Holm p-value'] = pvals_1day_holm
rf_ates_days['DML PLR 3 days Holm p-value'] = pvals_3days_holm
rf_ates_days['DML PLR 5 days Holm p-value'] = pvals_5days_holm
rf_ates_days['DML PLR all days Holm p-value'] = pvals_alldays_holm

rf_ates_days['DML PLR 1 day ATE'] = rf_ates_days['DML PLR 1 day ATE'].apply(lambda x: f"{100*x:.2f}")
rf_ates_days['DML PLR 3 days ATE'] = rf_ates_days['DML PLR 3 days ATE'].apply(lambda x: f"{100*x:.2f}")
rf_ates_days['DML PLR 5 days ATE'] = rf_ates_days['DML PLR 5 days ATE'].apply(lambda x: f"{100*x:.2f}")
rf_ates_days['DML PLR all days ATE'] = rf_ates_days['DML PLR all days ATE'].apply(lambda x: f"{100*x:.2f}")

rf_ates_days['DML PLR 1 day Holm p-value'] = rf_ates_days['DML PLR 1 day Holm p-value'].apply(lambda x: significance_level(x))
rf_ates_days['DML PLR 3 days Holm p-value'] = rf_ates_days['DML PLR 3 days Holm p-value'].apply(lambda x: significance_level(x))
rf_ates_days['DML PLR 5 days Holm p-value'] = rf_ates_days['DML PLR 5 days Holm p-value'].apply(lambda x: significance_level(x))
rf_ates_days['DML PLR all days Holm p-value'] = rf_ates_days['DML PLR all days Holm p-value'].apply(lambda x: significance_level(x))

rf_ates_days['DML PLR 1 day p-value'] = rf_ates_days['DML PLR 1 day p-value'].apply(lambda x: significance_level(x))
rf_ates_days['DML PLR 3 days p-value'] = rf_ates_days['DML PLR 3 days p-value'].apply(lambda x: significance_level(x))
rf_ates_days['DML PLR 5 days p-value'] = rf_ates_days['DML PLR 5 days p-value'].apply(lambda x: significance_level(x))
rf_ates_days['DML PLR all days p-value'] = rf_ates_days['DML PLR all days p-value'].apply(lambda x: significance_level(x))

columns_order_days = [
    'DML PLR 1 day ATE', 
    # 'DML PLR 1 day p-value',
    'DML PLR 1 day Holm p-value',
    'DML PLR 3 days ATE', 
    # 'DML PLR 5 days p-value',
    'DML PLR 3 days Holm p-value', 
    'DML PLR 5 days ATE',
    # 'DML PLR 3 days p-value', 
    'DML PLR 5 days Holm p-value',
    'DML PLR all days ATE', 
    # 'DML PLR all days p-value', 
    'DML PLR all days Holm p-value'
    ]
rf_ates_days = rf_ates_days[columns_order_days]

time_intervals = ['1 day', '3 days', '5 days', 'all days']
sub_columns = ['ATE',  'Holm p-value']
multi_index = pd.MultiIndex.from_tuples([("DML PLR (" + time + ")", sub_column) for time in time_intervals for sub_column in sub_columns])

rf_ates_days.columns = multi_index
# add mu
rf_ates_days.loc[table_order_index].to_clipboard()
# %%Double ML PLR 
preds = pd.Series([i[0][0] for i in rf_modelling_outputs['payment_mobile_support']['models_obj']['DoubleMPLR2'].predictions['ml_l']])

# Creating the histogram
plt.hist(preds, bins=100)
plt.title('Histogram of Data')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Displaying the histogram
plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Applying the sigmoid function to the data
sigmoid_preds = preds.apply(sigmoid)

# Creating the histogram
plt.hist(sigmoid_preds, bins='auto')
plt.title('Histogram of Sigmoid Transformed Data')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Displaying the histogram
plt.show()

# plot roc aauc curve
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot

# calculate roc curve
fpr, tpr, thresholds = roc_curve(modelling_df_dict['payment_mobile_support']['y_train'], sigmoid_preds)

# calculate AUC
auc = roc_auc_score(modelling_df_dict['payment_mobile_support']['y_train'], sigmoid_preds)
print('AUC: %.3f' % auc)

# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')

# show the plot
pyplot.show()



 # %% CHecking propensity scores differencess between treatment and control groups
 # --------------------

from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# create a dictionary to store propensity scores for each treatment

covariates = [
       'clicks_itinerary_direct_flight',
       'clicks_itinerary_sales_price_pax',
       'clicks_itinerary_segment_count', 'clicks_itinerary_totaldistance',
       'clicks_itinerary_travel_timehours',
       'clicks_itinerary_with_baggage', 'clicks_mobile',
       'clicks_passengers_count', 'ratio_sales_price_travel_time',
       'ratio_distance_passenger', 'ratio_travel_time_distance',
       'ratio_sales_price_distance', 'carriers_marketing_ratings_count',
       'carriers_marketing_ratings_mean',
       'ratio_sales_price_carrier_rating_max',
       'ratio_sales_price_carrier_rating_count',
       'clicks_itinerary_sales_price_if_cheapest',
       'clicks_itinerary_sales_price_if_best',
       'clicks_itinerary_sales_price_if_fastest',
       'clicks_itinerary_sales_price_diff_cheapest',
       'clicks_itinerary_sales_price_diff_best',
       'clicks_itinerary_sales_price_diff_fastest',
       'clicks_count',
    #    'google_trends_weekly_DK',
    #    'google_trends_weekly_DK_lag_7',
    #    'ratio_clicks_google_trends',
    #    'diff_clicks_google_trends',
       ]

propensity_scores_dict = {}
propensity_models = {}

for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    X_train, X_test, y_train, y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']

    # calculate propensity scores for each treatment group 
    # import random forest classifier
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(**rf_classifier_params)
    model.fit(X_train, treat_train)
    
    # store propensity scores
    propensity_scores_dict[treatment] = model.predict_proba(X_train)[:, 1]
    propensity_models[treatment] = model
    
# create a figure to plot the propensity score distributions
# %%plot the distribution of propensity scores for each treatment
for treatment in treatments:
    plt.figure(figsize=(10, 6))
    # histogram plot of the propensity scores for each treatment
    sns.distplot(propensity_scores_dict[treatment], hist=True, kde=False,
                    bins=int(180/5), color = 'darkblue', label=treatment)
    plt.title('Propensity Score Distributions by Treatment')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    # legend in upper center position 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    plt.show()
# %% Random forest classifier on Y output variable without treatment
# --------------------
pred_proba_dict = {}
pred_proba_models = {}
for treatment in treatments:
    print(treatment)
    treatment_var_name = 'treatment_' + treatment
    X_train, X_test, y_train, y_test, treat_train, treat_test = modelling_df_dict[treatment]['X_train'], modelling_df_dict[treatment]['X_test'], modelling_df_dict[treatment]['y_train'], modelling_df_dict[treatment]['y_test'], modelling_df_dict[treatment]['treat_train'], modelling_df_dict[treatment]['treat_test']

    # calculate propensity scores for each treatment group 
    # import random forest classifier
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(**rf_classifier_params)
    model.fit(X_train, y_train)
    
    # store propensity scores
    pred_proba_dict[treatment] = model.predict_proba(X_train)[:, 1]
    pred_proba_models[treatment] = model

# %% plot the distribution of predicted probabilities for each treatment
for treatment in treatments:
    plt.figure(figsize=(10, 6))
    # histogram plot of the propensity scores for each treatment
    sns.distplot(pred_proba_dict[treatment], hist=True, kde=False,
                    bins=int(180/5), color = 'darkblue', label=treatment)
    sns.distplot(pred_proba_dict[treatment], hist=True, kde=False,
                    bins=int(180/5), color = 'darkblue', label=treatment)
    plt.title('Propensity Score Distributions by Treatment')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    # legend in upper center position 
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    plt.show()


for treatment in treatments:
    # calculate accuracy and auc score for each treatment
    from sklearn.metrics import roc_auc_score, accuracy_score
    y_train = modelling_df_dict[treatment]['y_train']
    print(treatment)
    print("AUC", round(roc_auc_score(y_train, pred_proba_dict[treatment]), 2), "Accuracy", round(accuracy_score(y_train, pred_proba_dict[treatment].round()), 2))
    


# %%plor feature importance for each treatment

for treatment in actual_treatments:
    # for propensity_models[treatment]plot variable importance in random forest classifier
    # get importance
    importance = propensity_models[treatment].feature_importances_
    # summarize feature importance
    # plot feature importance in horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(X_train.columns, importance)
    plt.title('Feature Importance for Treatment: ' + treatment)
    plt.xlabel('Relative Importance')
    plt.ylabel('Feature')
    plt.show()


# %% PLot 

def plot_scatter(x_dim, y_dim, data, if_pvalue = True, save = False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from adjustText import adjust_text

    # Set the default theme to white
    sns.set_style("white")

    # Set the default font to Latex font
    plt.rcParams['font.family'] = 'serif'

    # Change figure size
    plt.rcParams['figure.figsize'] = [9, 9]

    # Scatter plot xgb_ates_means['diff'] vs xgb_ates_means['ate']
    scatter_plot = sns.scatterplot(x=x_dim, y=y_dim, hue='Category', data=data, palette=['blue', 'orange'])

    # Add horizontal line for 0
    plt.axhline(y=0, color='black', linestyle='-')
    plt.legend(loc='lower right')

    # add a 45 degrees line

    # Create a list to store the text objects
    texts = []
    for i, txt in enumerate(data.index):
        texts.append(plt.text(data[x_dim].iat[i], data[y_dim].iat[i], txt, ha='center', va=np.random.choice(["bottom", "top"])))

    # Use adjust_text to iteratively adjust text positions
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

    # set x lim
    if if_pvalue:
        plt.xlim(0, 1)
    else:
        x_lim = round(data[x_dim].abs().max(), 3) + 0.002
        # round to ceiling

        plt.xlim(-x_lim, x_lim)

    # set y lim
    y_lim = round(data[y_dim].abs().max(), 3) + 0.002
    plt.ylim(-y_lim, y_lim)
    # 45 degree line
    plt.plot([-x_lim, x_lim], [-y_lim, y_lim], ls="--", c=".3")
    # for each point draw a dashed grey line to the 45 degree line
    for i in range(len(data)):
        plt.plot([data[x_dim].iat[i], data[x_dim].iat[i]], [data[y_dim].iat[i], data[x_dim].iat[i]], ls="--", c=".3")
    # Add a title to the plot
    # plt.title("Scatter Plot")

    # log y scale
    # plt.yscale('log')

    # Show the plot
    plt.show()

    # save this scatter plot with relevant name
    if save:
        scatter_plot.figure.savefig(f'plots/{y_dim} vs {x_dim}.png', bbox_inches='tight')

plot_scatter(x_dim = "DML PLR Holm p-value", y_dim="DML PLR ATE", data=results_all_table, if_pvalue=True, save=True)

plot_scatter(x_dim="Differences ATE", y_dim="DML PLR ATE", data=results_all_table, if_pvalue=False, save=True)

plot_scatter("N", "DML PLR p-value", data=results_all_table, if_pvalue=True, save=False)


clicks_count_treatments = pd.DataFrame({"clicks_count_mean" : [modelling_df_dict[treatment]['X_train']['clicks_count'].mean() for treatment in treatments]})
clicks_count_treatments.index = treatments
results_all_table_clicks_count = pd.concat([results_all_table, clicks_count_treatments], axis=1)

conversion_rates_treatments = pd.DataFrame({"conversion_rate" : [modelling_df_dict[treatment]['y_train'].mean() for treatment in treatments]})
conversion_rates_treatments.index = treatments
results_all_table_conversion_rates = pd.concat([results_all_table, conversion_rates_treatments], axis=1)

plot_scatter("clicks_count_mean", "DML PLR p-value", results_all_table_clicks_count, if_pvalue=False, save=False)

plot_scatter(x_dim="conversion_rate", y_dim="DML PLR p-value", data=results_all_table_conversion_rates, if_pvalue=False, save=False)

y_means

# %%
xgb_ates_means[['IPW', 'diff']].assign(ratio=xgb_ates_means['IPW'] / xgb_ates_means['diff'])
# %%
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Calculate d_i^2
d_i_squared = d_i ** 2

# Calculate the sum of d_i^2
sum_d_i_squared = np.sum(d_i_squared)

d_i_y_i = d_i * y_train
d_i_y_i_sum = np.sum(d_i_y_i)

# Calculate tau
tau = d_i_y_i_sum / sum_d_i_squared


# Calculate d_i * g(x_i)
d_g_x = d_i * g(X_i)

# Calculate d_i * y_i
d_y = d_i * y_i

# Calculate tau_hat
tau_hat = tau + np.sum(d_g_x) / sum_d_i_squared + np.sum(d_y) / sum_d_i_squared


# Assuming these are your data
X_i = np.array(X_train)  # your X_train data
d_i = np.array(treat_train)  # your d_i data
y_i = np.array(y_train)  # your y_i data

 # %% Partially linear regression 
 # --------------------
plr_outputs = {}

for treatment in treatments:
    X_train = modelling_df_dict[treatment]['X_train']
    y_train = modelling_df_dict[treatment]['y_train']
    treat_train = modelling_df_dict[treatment]['treat_train']

    # Initialize the model
    # g_model = RandomForestClassifier(**rf_classifier_params)
    g_model = RandomForestRegressor(**rf_regressor_params)

    def g(X_train, y_train):
        # Fit the model
        g_model.fit(X_train,y_train)
    
        # Return the predicted values
        return g_model.predict(X_train)

    input = pd.DataFrame({
            "d":treat_train,
            "g":g(X_train, y_train)
            })
    input = sm.add_constant(input)
    
    import statsmodels.api as sm


    # Fit the model
    plr = sm.OLS(y_train, input).fit()
    # add intercept

    # show coefficients of the model

    plr_outputs[treatment] = {
        'models_obj': plr,
        "ate": plr.params['d'],
    }
    # return summary of the model with coefficients and pvalues

    print("learning finished for treatment: " + treatment)

# %%
plr_ates = pd.DataFrame({"PLR":[plr_outputs[treatment]['ate']  for treatment in treatments]}, treatments)
plr_ates.sort_index().style.background_gradient(cmap=red_white_green, vmin=-.1, vmax=.1, low=0, high=0, axis=None, subset = ['PLR'])

[plr_outputs[treatment]['ate']  for treatment in treatments]

for treatment in treatments:
    rf_modelling_outputs[treatment]['models_obj']['PLR'] = plr_outputs[treatment]['models_obj']
    rf_modelling_outputs[treatment]['results_df']['ate'] = plr_outputs[treatment]['ate']
# %%
