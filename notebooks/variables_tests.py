 %% 
# filter df if orders_if_order == 1
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
# %%
for var in search_engine_changes.keys():
    perform_t_test(
    dataframe=df[df['orders_if_order'] == 1], 
    group_var="search_changes_"+var, 
    var_test = 'orders_addon_totalsum', 
    group1='before_change',
    group2='after_change')

# %%
# from src.features.build_features import perform_t_test

for var in experiment_vars:
    perform_t_test(
    dataframe=df, 
    group_var=var, 
    var_test = 'orders_total_price_SEK', 
    group1='control', 
    group2 = 'hypothesis')


# %% Tests for change in mobile payment support in Denmark 

# orders_if_order
perform_t_test(
    dataframe=df[df['currency'] == 'DKK'], 
    group_var="search_changes_mobile.pay.support.Denmark",
    var_test = 'orders_total_price_SEK',
    group1='before_change', 
    group2='after_change'
    )
# order_total sales price
perform_t_test(
    dataframe=df[df['currency'] == 'DKK'], 
    group_var="search_changes_mobile.pay.support.Denmark",
    var_test = 'orders_if_order',
    group1='before_change', 
    group2='after_change'
    )

perform_t_test(
    dataframe=(df.query("currency == 'DKK' & orders_if_order == 1")
               .assign(orders_cancelled_bin = lambda x: np.where(x['orders_cancelled'] == True, 1, 0))),
    group_var="search_changes_mobile.pay.support.Denmark",
    var_test = 'orders_cancelled_bin',
    group1='before_change', 
    group2='after_change'
    )


# %%
for var in experiment_vars:
    perform_t_test(
    dataframe=df, 
    group_var=var, 
    var_test = 'orders_addon_totalsum', 
    group1='control', 
    group2 = 'hypothesis')


# %%

for var in search_engine_changes.keys():
    perform_t_test(
    dataframe=df[df['orders_if_order'] == 1], 
    group_var="search_changes_"+var, 
    var_test = 'orders_total_price_SEK', 
    group1='before_change',
    group2='after_change')

# %%
# groupby by clicks_created_at_date and summarize orders_if_orders, number of clicks and calculate the conversion rate
# Calculate conversion rate and save as 'conv_rate'
conv_rates = (df
              .groupby('clicks_created_at_date')['orders_if_order']
              .agg(orders=('sum'), clicks=('count'))
              .assign(conv_rate=lambda x: x['orders'] / x['clicks']))
conv_rates
# %% [markdown]
import datetime
import matplotlib.pyplot as plt

# plot conv_rate with matplotlib and change x ticks to be less frequent - once per month
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(conv_rates.index, conv_rates['conv_rate'])
ax.set_xticks(conv_rates.index[::30])
ax.set_xticklabels(conv_rates.index[::30], rotation=90)

filter_columns_by_regex(df, '.*experiments')

# add to the plot vertical lines with the dates of the search engine changes
# for date in search_engine_changes.values():
#     plt.axvline(datetime.datetime.combine(date, datetime.time.min), color='r', linestyle='--')

# show the plot
plt.show()

# %% [Conversion rate weekly plot]

# groupby by weeks calculated from clicks_created_at_date and summarize orders_if_orders, number of clicks and calculate the conversion rate
# Calculate conversion rate and save as 'conv_rate'
conv_rates_weekly = (df
                     # add a column with the week calculated from clicks_created_at_datetime to be the monday of the week
                    .assign(clicks_created_at_week=lambda x: x['clicks_created_at_datetime'].dt.to_period('W').dt.start_time)
                    .groupby('clicks_created_at_week')['orders_if_order']
                    .agg(orders=('sum'), clicks=('count'))
                    .assign(conv_rate=lambda x: x['orders'] / x['clicks'])
                    .reset_index())
conv_rates_weekly.head()

# %%
import matplotlib.pyplot as plt

# plot conv_rate_skk_weekly by date
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(conv_rates_weekly['clicks_created_at_week'], conv_rates_weekly['conv_rate'])

# add vertical line for the date of the search engine change
ax.axvline(search_engine_changes['mobile.pay.support.Denmark'], color='r', linestyle='--')
# add a label for the search engine change named 'mobile.pay.support.Denmark'
ax.text(search_engine_changes['mobile.pay.support.Denmark'], 0.032, 'mobile.pay.support.Denmark', rotation=90)
# set title and axis labels
ax.set_title('Weekly Conversion Rates for all currencies')
ax.set_xlabel('Date')
ax.set_ylabel('Conversion Rate')

# display the plot
plt.show()
# %%
# calculate the conversion rate 
# Calculate conversion rate and save as 'conv_rate'
conv_rates_dkk = (df[df['currency'] == 'DKK']
                .groupby('search_changes_mobile.pay.support.Denmark')['orders_if_order']
                .agg(orders=('sum'), clicks=('count'))
                .assign(conv_rate=lambda x: x['orders'] / x['clicks']))

conv_rates_dkk

# %%
conv_rates_dkk_weekly = (df[df['currency'] == 'DKK']
                         # add a column with the week calculated from clicks_created_at_datetime to be the monday of the week
                .assign(clicks_created_at_week=lambda x: x['clicks_created_at_datetime'].dt.to_period('W').dt.start_time)
                .groupby('clicks_created_at_week')['orders_if_order']
                .agg(orders=('sum'), clicks=('count'))
                .assign(conv_rate=lambda x: x['orders'] / x['clicks'])
                .reset_index())
conv_rates_dkk_weekly.head()
# %%
import matplotlib.pyplot as plt

# plot conv_rate_skk_weekly by date
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(conv_rates_dkk_weekly['clicks_created_at_week'], conv_rates_dkk_weekly['conv_rate'])

# add vertical line for the date of the search engine change
ax.axvline(search_engine_changes['mobile.pay.support.Denmark'], color='r', linestyle='--')
# add a label for the search engine change named 'mobile.pay.support.Denmark'
ax.text(search_engine_changes['mobile.pay.support.Denmark'], 0.032, 'mobile.pay.support.Denmark', rotation=90)

# set title and axis labels
ax.set_title('Weekly Conversion Rates for DKK')
ax.set_xlabel('Date')
ax.set_ylabel('Conversion Rate')

# display the plot
plt.show()

# %%
# %%
# calculate the conversion rate 
# Calculate conversion rate and save as 'conv_rate'
conv_rates_dkk = (df[df['currency'] == 'DKK']
                .groupby('search_changes_mobile.pay.support.Denmark')['orders_if_order']
                .agg(orders=('sum'), clicks=('count'))
                .assign(conv_rate=lambda x: x['orders'] / x['clicks']))

conv_rates_dkk

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
# -------------------------------
# Following Estimation of Heterogeneous Treatment Effects
# Susan Athey, Stefan Wager, Vitor Hadad, Sylvia Klosin, Nicolaj Muhelbach, Xinkun Nie, Matt Schaelling
# May 07, 2020
# https://gsbdbi.github.io/ml_tutorial/hte_tutorial/hte_tutorial.html
# -------------------------------


# %%
from CTL.causal_tree_learn import CausalTree

# honest CT (Athey and Imbens, PNAS 2016)
ct_honest = CausalTree(honest=True, weight=0.0, split_size=0.0)
ct_honest.fit(x_train, y_train.to_numpy().astype(np.float), treat_train.to_numpy().astype(np.float))
ct_honest.prune()
ct_honest_predict = ct_honest.predict(x_test)
ct_honest.plot_tree(features=x_cols, filename="output/bin_tree_honest", show_effect=True)

# Check the resulting type and values
print(type(treat_train)) 

# %%
# honest CT (Athey and Imbens, PNAS 2016)
ct_honest = CausalTree(honest=True, weight=0.0, split_size=0.0)
ct_honest.fit(x_train, y_train, treat_train.astype(np.float64))

