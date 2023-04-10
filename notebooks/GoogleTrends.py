import os
import pandas as pd
import numpy as np

#set working directory to this from the workspace
os.chdir('/Users/janjarco/Programming/PrivateRepository/FlightDataThesisProject')

# GoogleTrends.py
import pytrends
from pytrends.request import TrendReq
import pandas as pd

# Connect to Google Trends API
pytrends = TrendReq()
pytrends = TrendReq(
    hl='en-US', 
    tz=60,
    # proxies = {'https': 'https://34.203.233.13:80'}
                    )

# Set the keyword you want to extract trends for
keywords = {
    "Danish": ["flybilletter", "billige flybilletter", "direkte fly", "skyscanner",  "flybilletter billige"],
    "Swedish": ["flygbiljetter", "billiga flygbiljetter", "flyg", "sas flygbiljetter", "skyscanner"],
    "German": ["flugtickets",  "günstige flugtickets", "flüge", "flugtickets buchen", "lufthansa",],
    "Norwegian": ["flybilletter", "billige flybilletter", "flybilletter oslo", "norwegian flybilletter", "flybilletter billig"]
}

[len(keywords[k]) for k in keywords.keys()]
# Set the time period you want to extract trends for
start_date = "2022-01-01"
end_date = "2023-03-20"

start_year = 2022
start_mon = 1
stop_year = 2023
stop_mon = 3

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pytrends import dailydata


# trends = dailydata.get_daily_data(
#         keywords, 
#         start_year = start_year, 
#         start_mon = start_mon, 
#         stop_year = stop_year, 
#         stop_mon = stop_mon, 
#         geo=geo)

def get_google_trends(keywords, start_date, end_date, geo):
    """
    This function uses the Google Trends API to retrieve the interest over time for the given keywords (kw_list) in Denmark.
    """
    # Build the payload and retrieve the trends data
    pytrends.build_payload(kw_list=keywords, timeframe=f"{start_date} {end_date}", geo=geo)
    trends = pytrends.interest_over_time()

    # Drop the isPartial column
    trends = trends.drop(columns="isPartial")

    # Rename the columns
    trends = trends.rename(columns={kw: f"google_trends_{geo}_{kw}" for kw in keywords})

    # Interpolate missing values
    # for col in trends.columns:
    #     if col.startswith("google_trends_"):
    #         trends[col] = trends[col].replace(0, np.nan).interpolate(method="linear", limit_direction="both")

    # Return date and sum of google_trends columns
    trends_sum = trends.loc[:, trends.columns.str.startswith("google_trends_")].sum(axis=1)
    trends_sum = trends_sum.reset_index(name="google_trends_"+geo)
    # take date coulumn to index
    trends_sum = trends_sum.set_index("date")

    return trends_sum

# pytrends.build_payload(kw_list=keywords["Danish"], timeframe=f"{start_date} {end_date}", geo="DK")
# trends = pytrends.interest_over_time()
# pytrends.suggestions(keyword='flybilletter')
# pd.DataFrame(pytrends.suggestions(keyword='flybilletter'))

# read csv "multiTimeline DK.csv"
dk_trends = pd.read_csv("data/external/multiTimeline DK.csv", skiprows=1)
# add sum of trends and anme it "google_trends_DK"
dk_trends["google_trends_DK"] = dk_trends.iloc[:, 1:].sum(axis=1)

dk_trends = pd.DataFrame(dk_trends.set_index("Week" )['google_trends_DK'])
# rename index to "date"
dk_trends.index.name = "date"
# change index to datetime
dk_trends.index = pd.to_datetime(dk_trends.index)

# leave only column "google_trends_DK"
# dk_trends = dk_trends["google_trends_DK"]
# dk_trends = get_google_trends(keywords["Danish"], start_date, end_date, "DK")
se_trends = get_google_trends(keywords["Swedish"], start_date, end_date, "SE")
no_trends = get_google_trends(keywords["Norwegian"], start_date, end_date, "NO")
de_trends = get_google_trends(keywords["German"], start_date, end_date, "DE")

# join by date trends dataframes, reset index and sort by date
trends_df = pd.concat([ dk_trends, se_trends, no_trends, de_trends], axis=1, join="inner")

# temporary table with all the dates by day from 2022-01-01 to 2023-03-20
dates = pd.date_range(start_date, end_date, freq="D")
# dates to dafrmae with column name "date"
dates = pd.DataFrame(dates, columns=["date"])

# left join dates with trends_df
trends_df = dates.merge(trends_df, how="left", on="date").ffill().bfill().set_index("date")
# apply kernel smoothing to trends_df
trends_df = trends_df.rolling(7, win_type="gaussian", min_periods=1, center=True).mean(std=2).round()

# Save the trends data to a csv file
trends_df.to_csv("data/external/google_trends_interest.csv", index=True)

# Print the first 5 rows of the DataFrame
trends_df.head()

#%% visualize all the trends with legend in dk_trends
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
plt.plot(trends_df)
plt.legend(trends_df.columns, loc='upper left')
plt.show()
# %%
