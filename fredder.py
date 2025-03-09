import os
import requests
import pandas as pd
from dotenv import load_dotenv
import utils

# Load environment variables from the .env file
load_dotenv()
api_key = os.getenv('FRED_API_KEY')
if not api_key:
    raise ValueError("FRED_API_KEY not found in the .env file")


def fetch_series_data(series_id, api_key):
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()
    observations = data.get('observations', [])

    df = pd.DataFrame(observations)
    if df.empty:
        print(f"No data returned for series: {series_id}")
        return df

    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.set_index('date', inplace=True)
    return df


def run():
    series_dict = {
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',  # Interest rate yield example
        '30Y': 'DGS30',
        '5YBEI': 'T5YIE',
        'USD': 'DTWEXAFEGS',
        'VIX': 'VIXCLS',  # VIX index
        'SP500': 'SP500',  # S&P 500 price index
        'BTC': 'CBBTCUSD'
    }

    dataframes = []
    # Fetch and display the first few rows of data for each series
    for name, series_id in series_dict.items():
        print(f"Fetching data for {name} (Series ID: {series_id})...")
        df = fetch_series_data(series_id, api_key)[['value']]
        df.rename(columns={'value':name}, inplace=True)
        dataframes.append(df)

    df = dataframes[0]
    for d in dataframes[1:]:
        df = pd.merge(df, d, how='left', left_index=True, right_index=True)

    df.index = pd.to_datetime(df.index).date

    print("FRED data:")
    utils.pdf(df.tail(3))
    df.to_parquet('data/fred.parquet')

if __name__ == '__main__':
    run()
