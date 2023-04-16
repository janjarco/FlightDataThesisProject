# Airline information 


import requests
from bs4 import BeautifulSoup
import pandas as pd
import tqdm
# read from pkl data/interim/nested_carriers.pkl
import os

os.chdir('/Users/janjarco/Programming/PrivateRepository/FlightDataThesisProject')

nested_carriers = pd.read_pickle('data/interim/nested_carriers.pkl')
carriers_unique = set([value.replace("'", "") for sublist in nested_carriers for value in sublist[1:-1].split(', ')])

nested_carriers_list = []
for sublist in tqdm.tqdm(nested_carriers):
    new_sublist = []
    for value in sublist[1:-1].split(', '):
        new_sublist.append(value.replace("'", ""))
    nested_carriers_list.append(new_sublist)


def scrape_airline_codes(url):
    # Send a request to the URL and get the content
    response = requests.get(url)
    content = response.content

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')

    # Find the table containing the airline codes
    table = soup.find('table', {'class': 'wikitable'})

    # Extract the table headers
    headers = [header.text.strip() for header in table.findAll('th')]

    # Extract the table rows
    rows = table.findAll('tr')[1:]  # Exclude the first row, which contains the headers

    # Extract the data from the rows
    data = [[cell.text.strip() for cell in row.findAll('td')] for row in rows]

    # Create a DataFrame with the extracted data
    df = pd.DataFrame(data, columns=headers)

    return df

base_url = "https://en.wikipedia.org/wiki/List_of_airline_codes_("
values = ['0–9'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

all_dataframes = []

for value in values:
    url = f"{base_url}{value})"
    print(f"Scraping {url}")
    df = scrape_airline_codes(url)
    all_dataframes.append(df)

# Combine all the DataFrames into a single DataFrame
airlines = pd.concat(all_dataframes, ignore_index=True)

# Create a DataFrame with the extracted data
airlines = airlines[['IATA', 'Airline', 'Country']][airlines['IATA'] != ''].reset_index(drop=True)

# save to pkl data/interim/airlines_wikipedia.pkl
airlines.to_pickle('data/interim/airlines_wikipedia.pkl')

airlines = pd.read_pickle('data/interim/airlines_wikipedia.pkl')
# join carries_unique with airlines
carriers_unique = pd.DataFrame(carriers_unique, columns=['IATA']).drop_duplicates(subset='IATA', keep='first')
carriers_unique = carriers_unique.merge(right =  airlines, on='IATA', how='left')

# return carriers_unique where Airline is null
carriers_unique[carriers_unique['Airline'].isna()]

airlines_list_nested = [list[1:-1].split(', ') for list in [value.replace("'", "") for value in nested_carriers]]

# %%
# airlines raatings

# read ratings from airlines_ratings_skytrax.txt separated by comma 
airlines_ratings = pd.read_csv('data/external/airlines_ratings_skytrax.txt', sep=',', header=1,index_col=1 , names=['IATA', 'Star Rating']).reset_index().rename(columns={'index': 'Airline'})

# airlines_ratings trim whitespaces for all columns
airlines_ratings = airlines_ratings.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# airlines_ratings change Star Rating to numeric if it is nan return None
airlines_ratings['Star Rating'] = pd.to_numeric(airlines_ratings['Star Rating'], errors='coerce')
# airlines_ratings if IATA is duplicated keep the first
airlines_ratings = airlines_ratings.drop_duplicates(subset='IATA', keep='first')

# merge  carriers_unique with airlines_ratings with left join
airlines = carriers_unique.merge(airlines_ratings, on='IATA', how='left')
airlines = airlines.replace({'Star Rating': {' None': None}})
airlines[airlines['Star Rating'].isna()]

# %%
import tqdm
import random

nested_carriers_ratings = []
# nested_carriers_sample = random.sample(nested_carriers_list, 1890)
for row in tqdm.tqdm(nested_carriers_list):
    new_row = []
    for value in row:
        # return index of value in airlines['IATA']
        to_append = airlines[airlines['IATA'] == value]['Star Rating'].values
        if len(to_append)>0:
            new_row.append(to_append[0])
        else:
            new_row.append("XD")
    nested_carriers_ratings.append(new_row)


# nested_carriers_ratings = nested_carriers_ratings_save
nested_carriers_list_nas = []
for i in nested_carriers_ratings:
    # return number of NA OR NONE IN LIST
    nested_carriers_list_nas.append(i.count(None))

# %%
# return length of each element in the list nested_carriers_ratings
nested_carriers_ratings_len = [len(i) for i in nested_carriers_ratings]
sum(nested_carriers_ratings_len)

nested_carriers_list_len = [len(i) for i in nested_carriers_list]
sum(nested_carriers_list_len)

# create dataframe with two columns from lists: nested_carriers_ratings and nested_carriers_list
import numpy as np

nested_carriers_df = pd.DataFrame({'nested_carriers_marketing_ratings': nested_carriers_ratings, 'nested_carriers_marketing_list': nested_carriers_list})
nested_carriers_df['carriers_marketing_ratings_min'] = nested_carriers_df['nested_carriers_marketing_ratings'].apply(lambda x: min(x))
nested_carriers_df['carriers_marketing_ratings_max'] = nested_carriers_df['nested_carriers_marketing_ratings'].apply(lambda x: max(x))
nested_carriers_df['carriers_marketing_ratings_mean'] = nested_carriers_df['nested_carriers_marketing_ratings'].apply(lambda x: np.mean(x))

# check where nested_carriers_df has Nan values
nested_carriers_df[nested_carriers_df['nested_carriers_marketing_ratings'].apply(lambda x: any(pd.isnull(i) for i in x))]

# to pickle nested_carriers_df
nested_carriers_df.to_pickle('data/interim/nested_carriers_df.pkl')

# unnest nested list nested_carriers_list
from collections import Counter
carriers_unnested = [carrier for sublist in nested_carriers_list for carrier in sublist]
counts_carriers = Counter(carriers_unnested)    

airlines_ratings[airlines_ratings.IATA.isin([i[0] for i in counts_carriers.most_common(40) if i not in counts_carriers.most_common(30)])]

# %%
# processing of nested_carriers


# %%
import requests
from lxml import html

url = 'https://www.tripadvisor.com/Airlines'
response = requests.get(url)
tree = html.fromstring(response.content)

airline_names = tree.xpath('//*[@id="taplc_airlines_lander_main_0"]/div/div[4]/div/div[1]/div[1]/a[2]/div[1]/div/span/text()')
airline_ratings = tree.xpath('//*[@id="taplc_airlines_lander_main_0"]/div/div[4]/div/div[1]/div[1]/a[2]/div[2]/span[1]/@class')

for name, rating in zip(airline_names, airline_ratings):
    rating_value = rating.split('_')[-1]
    print(f'{name}: {rating_value}/5')
sum([tuple[1] for tuple in counts_carriers.most_common(30)]) / len(carriers_unnested)