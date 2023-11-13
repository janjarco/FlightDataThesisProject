
# ### Import libraries

# %%
import os
import pandas as pd
import numpy as np
# from src.data.filter_columns_by_regex import filter_columns_by_regex

#set working directory to this from the workspace
os.chdir('/Users/janjarco/Programming/PrivateRepository/FlightDataThesisProject')

#list all the files in current working directory
print(os.listdir("data"))


# %% [markdown]
# #### Reading the merged data frame of orders and clicks

clicks_orders_merge_currency = pd.read_csv("data/processed/clicks_orders_orderlines_merged.csv")

# %%
# viewfirst 5 rows of the clicks_orders_merge_currency
clicks_orders_merge_currency.shape

# %%
clicks_orders_merge_currency['orders_if_order'] = np.where(clicks_orders_merge_currency['orders_order_id'].notna() & clicks_orders_merge_currency['orders_cancelled'] == False,1, 0)
# groupby orders_if_order and orders_cancelled
clicks_orders_merge_currency.groupby(['orders_if_order', 'orders_cancelled']).size()
clicks_orders_merge = clicks_orders_merge_currency[clicks_orders_merge_currency.orders_if_order == 1]
# del clicks_orders_merge_currency
clicks_orders_merge.shape

# %% [markdown]
# ### Calculating distances between airports    

# %%
import pandas as pd
import numpy as np
from geopy.distance import geodesic

airports = pd.read_csv("https://raw.githubusercontent.com/ip2location/ip2location-iata-icao/master/iata-icao.csv")
airports = airports[['iata', 'latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})

# adding some missing airports
# {'SDZ', 'IOM', 'QKL', 'JER', 'ZYR'}
airports = airports.append({'iata': 'LON', 'lat': airports.loc[airports['iata'] == 'LHR', 'lat'].values[0], 'lon': airports.loc[airports['iata'] == 'LHR', 'lon'].values[0]}, ignore_index=True)
airports = airports.append({'iata': 'PRN', 'lat': 42.6629, 'lon': 21.1655}, ignore_index=True)
airports = airports.append({'iata': 'UBN', 'lat': 47.6514, 'lon': 106.8216}, ignore_index=True)
airports = airports.append({'iata': 'SDZ', 'lat': 43.4355556, 'lon': -1.5358333}, ignore_index=True)
airports = airports.append({'iata': 'IOM', 'lat': 54.083333, 'lon': -4.623889}, ignore_index=True)
airports = airports.append({'iata': 'QKL', 'lat': 50.84722, 'lon': 6.9577}, ignore_index=True)
airports = airports.append({'iata': 'JER', 'lat': 49.207947, 'lon': -2.195508}, ignore_index=True)
airports = airports.append({'iata': 'ZYR', 'lat': 50.837091, 'lon': 4.338753}, ignore_index=True)

airports = airports.append({'iata': 'BNH', 'lat': 50.8178, 'lon': -0.2833}, ignore_index=True)
airports = airports.append({'iata': 'GCI', 'lat': 49.434959, 'lon': -2.602139}, ignore_index=True)
airports = airports.append({'iata': 'PLH', 'lat': 50.423889, 'lon': -4.105278}, ignore_index=True)
airports = airports.append({'iata': 'QPP', 'lat': -33.9216, 'lon': 151.1805}, ignore_index=True)
airports = airports.append({'iata': 'XWG', 'lat': 53.3333, 'lon': -2.85}, ignore_index=True)
airports = airports.append({'iata': 'XYD', 'lat': 44.222, 'lon': -0.5989}, ignore_index=True)
airports = airports.append({'iata': 'ZRB', 'lat': 54.3333, 'lon': -3.4}, ignore_index=True)
airports = airports.append({'iata': 'ZYA', 'lat': 52.08, 'lon': 5.1222}, ignore_index=True)

airports = airports.append({'iata': 'QKL', 'lat': 50.84722, 'lon': 6.9577}, ignore_index=True)
airports = airports.append({'iata': 'JER', 'lat': 49.207947, 'lon': -2.195508}, ignore_index=True)
airports = airports.append({'iata': 'ZYR', 'lat': 50.837091, 'lon': 4.338753}, ignore_index=True)


airports[['lat', 'lon']] = airports[['lat', 'lon']].apply(pd.to_numeric)

print(airports)

def distance_airports(str):
    if pd.isna(str):
        return np.nan
    else:
        origin, destination = str.split('-')
        origin_lon = airports.loc[airports['iata'] == origin, 'lon'].values[0]
        origin_lat = airports.loc[airports['iata'] == origin, 'lat'].values[0]
        destination_lon = airports.loc[airports['iata'] == destination, 'lon'].values[0]
        destination_lat = airports.loc[airports['iata'] == destination, 'lat'].values[0]
        try:
            return round(geodesic((origin_lat, origin_lon), (destination_lat, destination_lon)).km, 2)
        except:
            return None

print(distance_airports('CPH-WAW'))

# %% [markdown]
# Cleaning clicks_itinerary_string string

# %%
clicks_orders_merge['clicks_itinerary_string'] = clicks_orders_merge['clicks_itinerary_string'].str.replace(' ','').str.strip()
clicks_orders_merge['clicks_itinerary_string'].apply(lambda x: len(x.split(','))).max()

clicks_itinerary_string_cols = clicks_orders_merge['clicks_itinerary_string'].str.replace(' ','').str.strip().str.split(',').apply(lambda x: pd.Series(x))
clicks_itinerary_string_cols.columns = ['clicks_itinerary_string_' + str(i+1) for i in clicks_itinerary_string_cols.columns]


# %%
clicks_itinerary_string_cols = clicks_itinerary_string_cols[['clicks_itinerary_string_' + str(i) for i in range(1,7)]]
clicks_itinerary_string_cols[clicks_itinerary_string_cols.isna().any(axis=1)]


# %% [markdown]
# Check if there are any missing airports in the database

# %%
from itertools import chain

#split column of dataframe into 2 by "-"
airports_all = [item for sublist in clicks_itinerary_string_cols.unstack().drop_duplicates().dropna().str.split('-') for item in sublist]
print(airports_all)
# list airpoirts in airports['iata'] that are not in airports_all
airports_notin_itineraries = set([x for x in airports_all if x not in airports['iata'].values])
# print(airports_notin_itineraries[:20])
airports_notin_itineraries
# %% [markdown]
# #### Final calculation of distances

# %%
itineraries_series = clicks_itinerary_string_cols.unstack().drop_duplicates().dropna().to_list()
itinerary_string_cols_dist = [distance_airports(i) for i in itineraries_series]

# %%
# create a dictionary mapping itinerary strings to their assigned values
itinerary_dist_dict = dict(zip(itineraries_series, itinerary_string_cols_dist))

# replace the values in all columns of clicks_clicks_orders_merge_currency using applymap
clicks_itinerary_string_cols_dist = clicks_itinerary_string_cols.applymap(lambda x: itinerary_dist_dict.get(x, x))
# adding suffix "dist" to each column
clicks_itinerary_string_cols_dist.columns = clicks_itinerary_string_cols_dist.columns + '_dist'

clicks_itinerary_string_cols_dist.head()


# %%
# merging three dataframes together 
clicks_orders_merge = pd.concat([clicks_orders_merge, clicks_itinerary_string_cols, clicks_itinerary_string_cols_dist], axis=1)

# calculating total distance of the journey
clicks_orders_merge['clicks_itinerary_totaldistance'] = clicks_orders_merge.filter(regex='_dist$').sum(axis=1, skipna=True)

# showing histogram of total distance
clicks_orders_merge['clicks_itinerary_totaldistance'].hist(bins=100)

# %% [markdown]
# #### Exchaning currencies

# %%
import requests
from requests.structures import CaseInsensitiveDict
import pandas as pd

# Define the URL of the API endpoint
URL = "https://api.freecurrencyapi.com/v1/historical?apikey=MfhFIDkrj37M0NRsdRSLlzxDu9kuXBYG8XNn2jf0"

# Define the parameters for the API request
parameters = {
    "date_from" : "2022-01-01", 
    "date_to" : "2022-12-31", 
    "base_currency" : "SEK"
}

# Make the API request
resp = requests.get(url = URL, params = parameters)

# Check the status code of the response to ensure it was successful
if resp.status_code != 200:
    print(f"Request failed with status code {resp.status_code}")
    exit(1)

# Load the response data as a JSON object
currency_data_2022 = resp.json()["data"]

# %%

# Define the URL of the API endpoint
URL = "https://api.freecurrencyapi.com/v1/historical?apikey=MfhFIDkrj37M0NRsdRSLlzxDu9kuXBYG8XNn2jf0"

# Define the parameters for the API request
parameters = {
    "date_from" : "2023-01-01", 
    "date_to" : "2023-03-20", 
    "base_currency" : "SEK"
}

# Make the API request
resp = requests.get(url = URL, params = parameters)

# Check the status code of the response to ensure it was successful
if resp.status_code != 200:
    print(f"Request failed with status code {resp.status_code}")
    exit(1)

# Load the response data as a JSON object
currency_data_2023 = resp.json()['data']

# %%
# merge currency_data_2022 and currency_data_2023
currency_data = {**currency_data_2022, **currency_data_2023}

# %%

# Normalize the data using Pandas and filter for data from 2021-04-27
currency_data_norm = pd.json_normalize(currency_data)
currency_data_norm

# %%
currency_data_melt = pd.melt(currency_data_norm, value_vars=currency_data_norm.columns.values, var_name='date_currency')
currency_data_melt[['date', 'currency']] = currency_data_melt['date_currency'].str.split('.', expand=True)
currency_data_melt = currency_data_melt[["currency", "date", "value"]].rename(columns={"value": "exchange_rate"})
currency_data_melt["date"] = currency_data_melt["date"]
currency_data_melt

# %%
# save currency data to csv file
currency_data_melt.to_csv('data/external/currency_data.csv', index=False)

# %%
# read data/external/currency_data.csv to currency_data_melt
currency_data_melt = pd.read_csv('data/external/currency_data.csv')

# %%
clicks_orders_merge.shape

# %%
clicks_orders_merge['clicks_itinerary_currency'].value_counts()


# %%
from datetime import datetime, timezone
# extracting order date
clicks_orders_merge["orders_created_at_date_original"] = clicks_orders_merge.orders_created_at_date
clicks_orders_merge["orders_created_at_datetime"] = pd.to_datetime(clicks_orders_merge.orders_created_at_date_original, format='%Y-%m-%dT%H:%M:%S%z')

clicks_orders_merge["orders_created_at_datetime"] = pd.to_datetime(clicks_orders_merge.orders_created_at, unit='s')
# clicks_orders_merge["orders_created_at_datetime"]
# clicks_orders_merge["orders_created_at_datetime"] = pd.to_datetime(clicks_orders_merge["orders_created_at"], origin='unix' )
clicks_orders_merge["orders_created_at_date"] = clicks_orders_merge["orders_created_at_datetime"].dt.date.astype(str)
clicks_orders_merge["orders_created_at_time"] = clicks_orders_merge["orders_created_at_datetime"].dt.time.astype(str)

# if clicks_orders_merge["orders_created_at_date"] is NaT replace with None
clicks_orders_merge["orders_created_at_date"] = clicks_orders_merge["orders_created_at_date"].apply(lambda x: None if x == 'NaT' else x)
clicks_orders_merge["orders_created_at_date"] 
# %%
from datetime import datetime, timezone
# extracting order date
clicks_orders_merge["clicks_created_at_original"] = clicks_orders_merge.clicks_created_at
clicks_orders_merge["clicks_created_at_datetime"] = pd.to_datetime(clicks_orders_merge.clicks_created_at_original, format='%Y-%m-%dT%H:%M:%S%z', utc=True).astype('datetime64[ns]')

# clicks_orders_merge["clicks_created_at_datetime"] = pd.to_datetime(clicks_orders_merge.clicks_created_at, unit='s')
# clicks_orders_merge["clicks_created_at_datetime"]
# clicks_orders_merge["clicks_created_at_datetime"] = pd.to_datetime(clicks_orders_merge["clicks_created_at"], origin='unix' )
clicks_orders_merge["clicks_created_at_date"] = clicks_orders_merge["clicks_created_at_datetime"].dt.date.astype(str)
clicks_orders_merge["clicks_created_at_time"] = clicks_orders_merge["clicks_created_at_datetime"].dt.time.astype(str)

clicks_orders_merge["clicks_created_at_date"]


# %% [markdown]
# Charts of distribution of orders and time in day

# %%
#visualize chart by time with number of clicks per day
clicks_orders_merge["clicks_created_at_date"].value_counts().sort_index().plot()

# %%
#visualize bar chart by time (hours:minute:seconds with number of orders per hour
clicks_orders_merge.clicks_created_at_datetime.dt.hour.value_counts().sort_index().plot(kind='bar')

# %%
#merge clicks_orders_merge and currency_data_melt on date and currency
clicks_orders_merge_currency = pd.merge(
    # clicks_orders_merge.rename(columns={'orders_created_at_date': 'date', "clicks_itinerary_currency": "currency"}), 
    clicks_orders_merge,
    currency_data_melt, 
    how='left', 
    left_on=['clicks_created_at_date', 'clicks_itinerary_currency'], 
    right_on=['date', 'currency'],
    # on=['date', 'currency'],
    suffixes=('_clicks', '_currencies')
    )
clicks_orders_merge_currency.columns.values

# %%
# check if there are any null values in exchange_rate
len(clicks_orders_merge_currency[clicks_orders_merge_currency['exchange_rate'].isnull()]) / len(clicks_orders_merge_currency)

# %% [markdown]
# There are 1.8% values without information about currency at this moment they are filtered

# %%
# filter clicks_orders_merge_currency where exchange_rate is not null
clicks_orders_merge_currency = clicks_orders_merge_currency[clicks_orders_merge_currency['exchange_rate'].notnull()]

# %%
# filter clicks_orders_merge_currency.columns.values  by regex
from src.data.filter_columns_by_regex import filter_columns_by_regex

experiment_vars = filter_columns_by_regex(clicks_orders_merge_currency, 'experiments')

# %%
# summarize clicks_orders_merge_currency by orders_created_at_date and caluculate counts and visualize in plot 
clicks_orders_merge_currency.groupby(["orders_created_at_date"])['clicks_index'].count().plot()


# %%
# calculate total price in SEK
clicks_orders_merge_currency['orders_total_price_SEK'] = clicks_orders_merge_currency['clicks_itinerary_sales_price'] * clicks_orders_merge_currency['exchange_rate']
# clicks_orders_merge_currency['orders_total_price_SEK'].hist(bins=100)

# histogram of clicks_itinerary_travel_time in hours
clicks_orders_merge_currency['clicks_itinerary_travel_timehours'] = round(clicks_orders_merge_currency['clicks_itinerary_travel_time'] / 60,2)
# clicks_orders_merge_currency['clicks_itinerary_travel_timehours'].hist(bins=100)
# %%
addon_vars = filter_columns_by_regex(clicks_orders_merge_currency, 'orders_addon')
# from a list of variables addon_vars create a sum of all the variables
clicks_orders_merge_currency['orders_addon_totalsum'] = clicks_orders_merge_currency[addon_vars].sum(axis=1)


# %%
import src.features.build_features as build_features


for var in experiment_vars:
    build_features.perform_t_test(
    dataframe=clicks_orders_merge_currency, 
    group_var=var, 
    var_test = 'orders_addon_totalsum', 
    group1='control', 
    group2 = 'hypothesis')

# %%
from datetime import datetime

from collections import OrderedDict

search_engine_changes = {
    'gateway.migration': datetime(2022, 6, 13, 13, 44),
    'lhg.gateway.migration': datetime(2022, 7, 25, 11, 22),
    'bump.BFM.version': datetime(2022, 8, 12, 9, 29),
    'add.price.override': datetime(2022, 8, 10, 7, 23),
    'mixed.itineraries': datetime(2022, 8, 10, 11, 54),
    'avoid.cabin.downgrade': datetime(2022, 8, 19, 9, 14),
    'hack.bagprice.override.Altea.FLX': datetime(2022, 9, 15, 7, 50),
    'remove.itineraries.departure.too.close': datetime(2022, 9, 28, 11, 9),
    'immediate.payment.if.less.36.hours.departure': datetime(2022, 9, 26, 13, 38),
    'support.20ITINS': datetime(2022, 12, 21, 16, 23),
    'add.support.unacceptable.connections': datetime(2023, 2, 16, 12, 55),
    'mobile.pay.support.Denmark': datetime(2023, 2, 21, 11, 31),
    'fix.reordering.after.maximum.results': datetime(2023, 1, 5, 11, 50)
}

# %%
from datetime import timedelta

def before_after_change(clicks_created_at, changes_dict, change_name):
    try:
        change_at = changes_dict[change_name]
        clicks_created_at = clicks_created_at.replace(tzinfo=None)
        if change_at - timedelta(weeks=1) <= clicks_created_at <= change_at + timedelta(weeks=1):
            return 'before_change' if clicks_created_at <= change_at else 'after_change'
        else:
            return None
    except:
        return None
# %%

# create a new column 'before_after_change' and apply the function before_after_change
search_changes_clicks_orders_merge_currency = pd.DataFrame()
for key in search_engine_changes:
    search_changes_clicks_orders_merge_currency['search_changes_'+key] = clicks_orders_merge_currency.apply(lambda row: before_after_change(
        row['clicks_created_at_datetime'], 
        search_engine_changes,
        key,
        ), axis=1)

# %%
# search_engine_changes for each variable in data frame show value_counts
for col in search_changes_clicks_orders_merge_currency.columns:
    print(search_changes_clicks_orders_merge_currency[col].value_counts())

# %%
# concat the search_changes_clicks_orders_merge_currency to the clicks_orders_merge_currency and save as clicks_orders_merge_currency
clicks_orders_merge_currency = pd.concat([clicks_orders_merge_currency, search_changes_clicks_orders_merge_currency], axis=1)

# %%
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# filter_columns_by_regex(clicks_orders_merge_currency, '_if_')
search_changes_conv_rates = []
for col in search_changes_clicks_orders_merge_currency.columns:
    # print(col)
    # groupby the column, summarize orders_if_orders, number of clicks and calculate the conversion rate
    # Calculate conversion rate and save as 'conv_rate'
    search_changes_conv_rates.append(clicks_orders_merge_currency.groupby(col)['orders_if_order'].agg(orders=('sum'), clicks=('count')).assign(conv_rate=lambda x: x['orders'] / x['clicks']))
    
search_changes_conv_rates
# %% [markdown]
# #### Tweak with clicks_orders_merge_currency['orders_if_order']
# %% 


# %%
# save clicks_orders_merge_currency to csv file
clicks_orders_merge_currency.to_csv('data/processed/clicks_orders_merge_currency.csv', index=False)

# %%
import time
start_time = time.time()
clicks_orders_merge_currency = pd.read_csv("data/processed/clicks_orders_merge_currency.csv", low_memory=True, parse_dates=['clicks_created_at_datetime', 'orders_created_at_datetime'])
end_time = time.time()
elapsed_time = end_time - start_time
print('Time elapsed (in seconds):', elapsed_time)
# %%

# fixing orders_if_order variable
# clicks_orders_merge_currency['orders_if_order'] = np.where(((clicks_orders_merge_currency['orders_if_order']==1) & clicks_orders_merge_currency['orders_cancelled'] == False),1, 0)
clicks_orders_merge_currency['orders_if_order'] = np.where((clicks_orders_merge_currency['orders_if_order'] == 1) & (clicks_orders_merge_currency['orders_cancelled'] == False), 1, 0)

clicks_orders_merge_currency.groupby(['orders_if_order', 'orders_cancelled']).size()
#%%
# save to csv names of columns with first 5 values by value counts for each column

columns_values = [clicks_orders_merge_currency[col].value_counts().head(5).index.tolist() for col in clicks_orders_merge_currency.columns]

# columns_values is a list of lists concatenate them row by row to the dataframe 
clicks_orders_merge_currency_columns_values = pd.DataFrame(columns_values)
import openpyxl
pd.concat([pd.Series(clicks_orders_merge_currency.columns), clicks_orders_merge_currency_columns_values], axis=1).to_excel("data/processed/columns_values.xlsx", index=False)

# %% [markdown]
# read pickle file search_engine_changes
import pickle
search_engine_changes = pickle.load(open("data/processed/search_engine_changes.pkl", "rb"))
    
# %%
# clicks_orders_merge_currency create a dummy variable before and after search_engine_changes['mobile.pay.support.Denmark']
# clicks_orders_merge_currency_gtrends = (clicks_orders_merge_currency.query('currency == "DKK"')
#         .assign(before_mobile_support_denmark=lambda x: np.where(x['clicks_created_at_datetime'] < search_engine_changes['mobile.pay.support.Denmark'], "before_change", "after_change")))
#%%
# read data/external/google_trends_interest.csv
google_trends = pd.read_csv("data/external/google_trends_interest.csv")

# for each variable create a lag of variable with suffix _lag
for col in google_trends.columns[google_trends.columns != 'date']:
    for i in range(7):
        google_trends[col+'_lag_'+str(i+1)] = google_trends[col].shift(i+1)

google_trends = google_trends.bfill()

# left join clicks_orders_merge_currency_gtrends and clicks_orders_merge_currency_google_trends on clicks_created_at_datetime = date
# clicks_orders_merge_currency_gtrends = clicks_orders_merge_currency_gtrends.assign(clicks_created_at_date=lambda x: x['clicks_created_at_datetime'].dt.date)
clicks_orders_merge_currency_gtrends = clicks_orders_merge_currency.merge(google_trends, how='left', left_on='clicks_created_at_date', right_on='date', )


filter_columns_by_regex(clicks_orders_merge_currency_gtrends, 'trends')

# clicks_orders_merge_currency_gtrends check NAs in google_trends_DK
clicks_orders_merge_currency_gtrends['google_trends_weekly_DK'].isna().sum()


# %%
from pandas.api.types import CategoricalDtype
from tzlocal import get_localzone
from datetime import datetime
import matplotlib.pyplot as plt

# time_filter_mobile_pay_support_Denmark = search_engine_changes['mobile.pay.support.Denmark'] - 1*(max(clicks_orders_merge_currency_gtrends['clicks_created_at_datetime']) - search_engine_changes['mobile.pay.support.Denmark'])
# time_filter_mobile_pay_support_Denmark = datetime(2023, 1, 1, 0, 0, 0)
# dkk add column orders_if_order_bin as boolean from orders_if_order
clicks_orders_merge_currency_gtrends = clicks_orders_merge_currency_gtrends.assign(orders_if_order_bin=lambda x: x['orders_if_order'] == 1)
# clicks_orders_merge_currency_gtrends = clicks_orders_merge_currency_gtrends.assign(before_mobile_support_denmark_bin=lambda x: x['before_mobile_support_denmark']  == "after_change")

# group by clicks_created_at_date by days of the week and visualize
clicks_orders_merge_currency_gtrends['clicks_created_at_datetime_weekday'] = clicks_orders_merge_currency_gtrends['clicks_created_at_datetime'].dt.dayofweek + 1

# Extracting weekend information
clicks_orders_merge_currency_gtrends['clicks_created_at_datetime_weekend'] = np.where(clicks_orders_merge_currency_gtrends['clicks_created_at_datetime_weekday'].isin([6, 7]), 1, 0)

# Extracting hour information
clicks_orders_merge_currency_gtrends['clicks_created_at_datetime_hour'] = clicks_orders_merge_currency_gtrends['clicks_created_at_datetime'].dt.hour

clicks_orders_merge_currency_gtrends['clicks_passengers_count']  =  clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price'] / clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_pax']

# visualize value counts clicks_orders_merge_currency_gtrends['clicks_created_at_datetime_weekday']
# clicks_orders_merge_currency_gtrends['clicks_created_at_datetime_weekday'].value_counts().sort_index().plot(kind='bar')

# group by clicks_created_at_datetime_weekday and caluclate ratio of orders_if_order_bin to number of clicks
clicks_orders_merge_currency_gtrends.groupby(['clicks_created_at_datetime_weekday', 'orders_if_order_bin']).size().unstack().rename(columns={0: 'no_order', 1: 'order'}).assign(ratio=lambda x: x['order'] / (x['order'] + x['no_order']))['ratio'].plot(kind='bar', stacked=True)

# %%
# Exploring marketing carriers
nested_carriers = clicks_orders_merge_currency_gtrends['clicks_itinerary_marketing_carriers']
carriers_unique = [value.replace("'", "") for sublist in nested_carriers for value in sublist[1:-1].split(', ')]

# save to pickle nested_carriers
import pickle
pickle.dump(nested_carriers, open("data/interim/nested_carriers.pkl", "wb"))

# %%
# read pickle file nested_carriers  data/interim/nested_carriers_df.pkl
import pickle
nested_carriers_df = pickle.load(open("data/interim/nested_carriers_df.pkl", "rb"))
nested_carriers_df.shape[0] == clicks_orders_merge_currency_gtrends.shape[0]

nested_carriers_df[nested_carriers_df['nested_carriers_marketing_ratings'].apply(lambda x: any(pd.isnull(i) for i in x))]

# concat nested_carriers_df to clicks_orders_merge_currency_gtrends
clicks_orders_merge_currency_gtrends = pd.concat([clicks_orders_merge_currency_gtrends, nested_carriers_df], axis=1)
filter_columns_by_regex(clicks_orders_merge_currency_gtrends, 'carriers')
#%% [markdown]
#### Feature engineering
# clicks_orders_merge_currency_gtrends['ratio_distance_distance'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_totaldistance'] / clicks_orders_merge_currency_gtrends['clicks_passengers_count']
clicks_orders_merge_currency_gtrends['ratio_sales_price_travel_time'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_pax'] / clicks_orders_merge_currency_gtrends['clicks_itinerary_travel_timehours']
clicks_orders_merge_currency_gtrends['ratio_distance_passenger'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_totaldistance'] / clicks_orders_merge_currency_gtrends['clicks_passengers_count']
clicks_orders_merge_currency_gtrends['ratio_travel_time_distance'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_travel_timehours'] / clicks_orders_merge_currency_gtrends['clicks_itinerary_totaldistance']
clicks_orders_merge_currency_gtrends['ratio_sales_price_distance'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_pax'] / clicks_orders_merge_currency_gtrends['clicks_itinerary_totaldistance']
clicks_orders_merge_currency_gtrends['carriers_marketing_ratings_count'] = [len(i)for i in clicks_orders_merge_currency_gtrends['nested_carriers_marketing_list']]
clicks_orders_merge_currency_gtrends['ratio_sales_price_carrier_rating_max'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_pax'] / clicks_orders_merge_currency_gtrends['carriers_marketing_ratings_max']
clicks_orders_merge_currency_gtrends['ratio_sales_price_carrier_rating_min'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_pax'] / clicks_orders_merge_currency_gtrends['carriers_marketing_ratings_min']
clicks_orders_merge_currency_gtrends['ratio_sales_price_carrier_rating_avg'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_pax'] / clicks_orders_merge_currency_gtrends['carriers_marketing_ratings_mean']
clicks_orders_merge_currency_gtrends['ratio_sales_price_carrier_rating_count'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_pax'] / clicks_orders_merge_currency_gtrends['carriers_marketing_ratings_count']

# %%
# create difference between clicked price and searhc results price
clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_if_cheapest'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price'] == clicks_orders_merge_currency_gtrends['clicks_result_set_cheapest_price']
clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_if_best'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price'] == clicks_orders_merge_currency_gtrends['clicks_result_set_best_price']
clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_if_fastest'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price'] == clicks_orders_merge_currency_gtrends['clicks_result_set_fastest_price']

clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_category'] = np.where(
    clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_if_cheapest'], 'cheapest',
    np.where(clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_if_best'], 'best',
    np.where(clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_if_fastest'], 'fastest',
'other')))

clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_diff_cheapest'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price'] - clicks_orders_merge_currency_gtrends['clicks_result_set_cheapest_price']
clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_diff_best'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price'] - clicks_orders_merge_currency_gtrends['clicks_result_set_best_price']
clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price_diff_fastest'] = clicks_orders_merge_currency_gtrends['clicks_itinerary_sales_price'] - clicks_orders_merge_currency_gtrends['clicks_result_set_fastest_price']

# df_copy['interaction_google_trends_mobile'] = df_copy[f"google_trends"] * df_copy['clicks_mobile']
# df_copy['interaction_google_trends_distance'] = df_copy[f"google_trends"] * df_copy['clicks_itinerary_totaldistance']
# df_copy['interaction_google_trends_passengers'] = df_copy[f"google_trends"] * df_copy['clicks_passengers_count']
# df_copy['interaction_google_trends_sales_price_pax'] = df_copy[f"google_trends"] * df_copy['clicks_itinerary_sales_price_pax']
# df_copy['interaction_google_trends_sales_price'] = df_copy[f"google_trends"] * df_copy['clicks_itinerary_sales_price_pax']
# df_copy['interaction_google_trends_travel_time'] = df_copy[f"google_trends"] * df_copy['clicks_itinerary_travel_timehours']
# df_copy['interaction_google_trends_direct_flight'] = df_copy[f"google_trends"] * df_copy['clicks_itinerary_direct_flight']
# df_copy['interaction_google_trends_weekend'] = df_copy[f"google_trends"] * df_copy['clicks_created_at_datetime_weekend']
# df_copy['interaction_google_trends_baggage'] = df_copy[f"google_trends"] * df_copy['clicks_itinerary_with_baggage']
# clicks_orders_merge_currency_gtrends['interaction_passengers_weekend'] = clicks_orders_merge_currency_gtrends['clicks_passengers_count'] * clicks_orders_merge_currency_gtrends['clicks_created_at_datetime_weekend']
# clicks_orders_merge_currency_gtrends['interaction_mobile_direct_flight'] = clicks_orders_merge_currency_gtrends['clicks_mobile'] * clicks_orders_merge_currency_gtrends['clicks_itinerary_direct_flight']
clicks_orders_merge_currency_gtrends=clicks_orders_merge_currency_gtrends.replace({True: 1, False: 0})

# save to csv
clicks_orders_merge_currency_gtrends.to_csv('data/processed/clicks_orders_merge_currency_gtrends.csv', index=False)
# filter_columns_by_regex(clicks_orders_merge_currency_gtrends, 'ratio')
#%%
clicks_orders_merge_currency_gtrends = pd.read_csv('data/processed/clicks_orders_merge_currency_gtrends.csv', date_parser=True)
# generate separate datasets by currency
clicks_orders_merge_currency_gtrends_dkk = clicks_orders_merge_currency_gtrends[clicks_orders_merge_currency_gtrends['clicks_itinerary_currency'] == 'DKK']
clicks_orders_merge_currency_gtrends_eur = clicks_orders_merge_currency_gtrends[clicks_orders_merge_currency_gtrends['clicks_itinerary_currency'] == 'EUR']
clicks_orders_merge_currency_gtrends_nok = clicks_orders_merge_currency_gtrends[clicks_orders_merge_currency_gtrends['clicks_itinerary_currency'] == 'NOK']
clicks_orders_merge_currency_gtrends_sek = clicks_orders_merge_currency_gtrends[clicks_orders_merge_currency_gtrends['clicks_itinerary_currency'] == 'SEK']

df_currency_dict={
    "DK": clicks_orders_merge_currency_gtrends_dkk,
    "DE": clicks_orders_merge_currency_gtrends_eur,
    "NO": clicks_orders_merge_currency_gtrends_nok,
    "SE": clicks_orders_merge_currency_gtrends_sek
}
#%%
# for key in df_currency_dict.keys():
#     df_copy = df_currency_dict[key].copy()
#     df_copy['google_trends_weekly'] = df_copy[f"google_trends_weekly_{key}"]
#     df_copy['google_trends_smooth'] = df_copy[f"google_trends_smooth_{key}"]
#     df_copy = df_copy.drop(columns=filter_columns_by_regex(df_copy, "google_trends_.*_"))
#     df_currency_dict[key] = df_copy
from src.data.filter_columns_by_regex import filter_columns_by_regex
filter_columns_by_regex(df_currency_dict['DK'], "trends")

# show percentage of missing values for each column in orders_merge_currency_gtrends_dkk

# %% Selecting columns for the conversion modelling
x_cols = [
    'clicks_created_at_datetime', 
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
    'ratio_sales_price_travel_time',
    'ratio_distance_passenger',
    'ratio_travel_time_distance',
    'ratio_sales_price_distance',
    'carriers_marketing_ratings_count',
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
    'clicks_itinerary_sales_price_category',
    'clicks_itinerary_sales_price_diff_cheapest',
    'clicks_itinerary_sales_price_diff_best',
    'clicks_itinerary_sales_price_diff_fastest',
    ]

x_cols = x_cols + filter_columns_by_regex(df_currency_dict['DK'], "trends_.*DK")
y_col = 'orders_if_order'

# select columns for the model with x_cols, y_col and d_cols
for key in df_currency_dict.keys():
    df_copy = df_currency_dict[key].copy()
    df_currency_dict[key] = df_copy[x_cols + [y_col]]

#%% save pickles for conversion modelling
# save each df from df_currency_dict to separate pickle file
import pickle
for key in df_currency_dict.keys():
    df_copy = df_currency_dict[key].copy()
    filename = f"data/processed/modelling_dataset_clicks_{key}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(df_copy, f)

# %% Selecting columns for the cancellation modelling
import numpy as np
clicks_orders_merge_currency_gtrends_dkk['orders_order_started'] = np.where(clicks_orders_merge_currency_gtrends_dkk['orders_order_id'].notna().copy(), True, False)
orders_merge_currency_gtrends_dkk = clicks_orders_merge_currency_gtrends_dkk[clicks_orders_merge_currency_gtrends_dkk['orders_order_started'] == True]

# show percentage of missing values for each column in orders_merge_currency_gtrends_dkk
missing_values = pd.DataFrame(orders_merge_currency_gtrends_dkk.isnull().mean().sort_values(ascending=False))
missing_values.columns = ['missing_value_percentage']

missing_values.loc[filter_columns_by_regex(orders_merge_currency_gtrends_dkk, "orders_"), :].sort_values(by='missing_value_percentage', ascending=False).iloc[40:80]


x_cols = [
    'clicks_created_at_datetime', 
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
    'ratio_sales_price_travel_time',
    'ratio_distance_passenger',
    'ratio_travel_time_distance',
    'ratio_sales_price_distance',
    'carriers_marketing_ratings_count',
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
    'clicks_itinerary_sales_price_category',
    'clicks_itinerary_sales_price_diff_cheapest',
    'clicks_itinerary_sales_price_diff_best',
    'clicks_itinerary_sales_price_diff_fastest',
    'orders_search_data.search_parameters.children', 
    'orders_search_data.search_parameters.infants', 
    'orders_search_data.search_parameters.adults',
    'orders_order_type',
    'orders_addon_totalsum',
    'orders_search_data.gate_name',
    ]

x_cols = x_cols + filter_columns_by_regex(orders_merge_currency_gtrends_dkk, "trends.*_DK")
y_col = 'orders_cancelled'

orders_merge_currency_gtrends_dkk = orders_merge_currency_gtrends_dkk[x_cols + [y_col]]

import pickle
filename = f"data/processed/modelling_dataset_orders_DK.pkl"
with open(filename, 'wb') as f:
    pickle.dump(orders_merge_currency_gtrends_dkk, f)

# %% [markdown]
# Saving workspace with user input

# %%
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


# %% [markdown]
# Loading workspace from the file

# %%
import dill

# Load workspace from file
filename = "workspace.pkl"
with open(filename, "rb") as f:
    dill.load_session(f)

# Workspace variables are now available in this script
print("Workspace loaded.")
clicks_orders_merge_currency.columns.values
