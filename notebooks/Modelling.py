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


# %% [markdown]
# #### Reading the merged data frame of orders and clicks

# %%
import time
start_time = time.time()
df = pd.read_csv("data/processed/clicks_orders_merge_currency.csv")
end_time = time.time()
elapsed_time = end_time - start_time
print('Time elapsed (in seconds):', elapsed_time)

# %%
# filter df.columns.values  by regex
from src.data.filter_columns_by_regex import filter_columns_by_regex


experiment_vars = filter_columns_by_regex(df, '.*experiments')
experiment_vars

# %%
from src.features.build_features import perform_t_test
for var in experiment_vars:
    perform_t_test(
    dataframe=df, 
    group_var=var, 
    var_test = 'orders_total_price_SEK', 
    group1='control', 
    group2 = 'hypothesis')


# %%

for var in search_engine_changes.keys():
    perform_t_test(
    dataframe=df, 
    group_var="search_changes_"+var, 
    var_test = 'orders_if_order',
    group1='before_change', 
    group2='after_change')

# %%
filter_columns_by_regex(df, 'if_')
# %%
