import random, io, glob, os, time, re
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from datetime import datetime
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import vix_futures_exp_dates
import utils

aws_access= utils.AWS_ACCESS
aws_secret= utils.AWS_SECRET

headers = {'User-Agent': 'XYZ/3.0'}


def pull_vix(exp, backfill=False):
    print('Downloading VIX Futures data...')
    utils.make_dir('data/VIX')
    reg_url = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_"
    # archive_url = "https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/archive/VX/VX_"

    dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in exp]
    today = datetime.now().date()
    
    if (backfill): dates = [date.strftime("%Y-%m-%d") for date in dates]
    else: dates = [date.strftime("%Y-%m-%d") for date in dates if date.date() >= today]

    gateway = ApiGateway("https://cdn.cboe.com",
                         regions=EXTRA_REGIONS,
                         access_key_id=aws_access,
                         access_key_secret=aws_secret)
    gateway.start()
    session = requests.Session()
    session.mount("https://cdn.cboe.com", gateway)

    for date in tqdm(dates):
        time.sleep(random.uniform(1, 3))

        try:
            response = session.get(f'{reg_url}{date}.csv', headers=headers)

            file_object = io.StringIO(response.content.decode('utf-8'))
            df = pd.read_csv(file_object)
            
            if len(df) > 1:
                df.to_parquet(f'data/VIX/{date}.parquet')

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None

    gateway.shutdown()
    print('Done downloading VIX Futures data.')


def process_vix():
    exp = vix_futures_exp_dates.run_over_time_frame(2013)

    path = 'data/VIX'
    p_files = glob.glob(os.path.join(path, "*.parquet"))

    files = []
    for i in p_files:
        files.append(pd.read_parquet(i))

    df = pd.concat(files)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # df.dropna(axis=1, how='all', inplace=True)
    from datetime import datetime
    expiry_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in exp]
    month_mapping = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    # Create a dictionary to map (month, year) to expiry dates
    month_year_to_date = {(date.month, date.year): date for date in expiry_dates}

    def extract_month_year(futures_str):
        if pd.isna(futures_str):
            return None  # Handle NaN values
        match = re.search(r'\((\w+)\s+(\d{4})\)', futures_str)
        if match:
            month_str, year_str = match.groups()
            month = month_mapping[month_str]
            year = int(year_str)
            return month, year
        else:
            return None

    # Create a new column with the corresponding expiry dates
    df['Expiry'] = df['Futures'].apply(lambda x: month_year_to_date.get(extract_month_year(x)))
    df['Expiry'] = pd.to_datetime(df['Expiry'])
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    # df['DTM'] = (df['Expiry'] - df['Trade Date']).dt.days
    df['DTM'] = df.apply(lambda row: (row['Expiry'] - row['Trade Date']).days, axis=1)

    def calculate_dfr(row, expiries):
        future_expiries = expiries[expiries <= row['Trade Date']]
        if not future_expiries.empty: return (row['Trade Date']-future_expiries.max()).days
        else: return None
    df['DFR'] = df.apply(lambda row: calculate_dfr(row, df['Expiry']), axis=1)

    def calculate_dtr(row, expiries):
        future_expiries = expiries[expiries >= row['Trade Date']]
        if not future_expiries.empty: return (future_expiries.min()-row['Trade Date']).days
        else: return None
    df['DTR'] = df.apply(lambda row: calculate_dtr(row, df['Expiry']), axis=1)

    df['Expiry'] = df['Expiry'].dt.date
    df['Trade Date'] = df['Trade Date'].dt.date

    df.sort_values(by=['Expiry','Trade Date'], inplace=True)

    df.rename(columns={'VIX': 'Close_Px', 'Trade Date': 'Trade_Date'}, inplace=True)
    df = df.sort_values(by=['Trade_Date', 'DTM'])
    df['Calendar_Num'] = df.groupby('Trade_Date')['DTM'].rank(ascending=True)

    df.to_parquet('data/vix.parquet')
    return df


def create_calendar_spread_df(df):
    new_data = []
    for trade_date, group in df.groupby('Trade_Date'):
        group = group.sort_values('Calendar_Num')
        for i in range(len(group) - 1):
            fut_1, fut_2 = group.iloc[i], group.iloc[i + 1]
            new_row = {
                'Trade_Date': trade_date,'Fut_1': fut_1['Futures'],'Open_1': fut_1['Open'],
                'High_1': fut_1['High'],'Low_1': fut_1['Low'],'Close_1': fut_1['Close'],
                'Total_Volume_1': fut_1['Total Volume'],'Open_Interest_1': fut_1['Open Interest'],
                'DTM_1': fut_1['DTM'],
                'Fut_2': fut_2['Futures'],'Open_2': fut_2['Open'],
                'High_2': fut_2['High'],'Low_2': fut_2['Low'],'Close_2': fut_2['Close'],
                'Total_Volume_2': fut_2['Total Volume'],'Open_Interest_2': fut_2['Open Interest'],
                'DTM_2': fut_2['DTM'],
                'DFR': fut_2['DFR'],'DTR': fut_2['DTR'],
                'Calendar_Spread': f"{int(fut_1['Calendar_Num'])}-{int(fut_2['Calendar_Num'])}",
                'spread': fut_2['Close'] - fut_1['Close']
            }
            new_data.append(new_row)
    new_df = pd.DataFrame(new_data)

    return new_df


def interpolate_missing_and_zero_vix(df_, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    nan_filled_counts = {}
    zero_filled_counts = {}

    def fill_nans_and_zeros_in_live_period(series):
        live_period = series.first_valid_index(), series.last_valid_index()
        live_series = series.loc[live_period[0]:live_period[1]].copy()
        nan_count = live_series.isna().sum()
        zero_count = (live_series == 0).sum()
        live_series.replace(0, np.nan, inplace=True)
        filled_series = live_series.interpolate(method='linear', limit_direction='both')
        series.update(filled_series)
        return series, nan_count, zero_count

    df = df_.copy()
    for column in df.columns:
        if column in exclude_columns:
            continue  # Skip interpolation for excluded columns

        df[column], nan_filled_counts[column], zero_filled_counts[column] = fill_nans_and_zeros_in_live_period(
            df[column])

    # print(f'Interpolated NaNs: {nan_filled_counts}')
    # print(f'Interpolated Zeros: {zero_filled_counts}')
    return df


def pivot_vix(df):
    pivot_df = df.pivot_table(index='Trade_Date', columns='Calendar_Num', values='Settle', aggfunc='first')
    pivot_df.columns = [f"{col}_Settle" for col in pivot_df.columns]

    df_ = df[['Trade_Date','DTR','DFR']]
    df_ = df_.drop_duplicates()
    df_ = df_.set_index('Trade_Date').sort_index()

    pivot_df = pd.merge(pivot_df, df_, how='left', left_index=True, right_index=True)
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df[[
        'Trade_Date', '1.0_Settle', '2.0_Settle', '3.0_Settle',
        '4.0_Settle', '5.0_Settle', '6.0_Settle', '7.0_Settle', '8.0_Settle',
        'DTR', 'DFR'
    ]].set_index('Trade_Date')

    pivot_df = interpolate_missing_and_zero_vix(pivot_df, exclude_columns=['DTR','DFR'])

    # pivot_df.to_parquet(utils.get_abs_path('data/vix_pivot.parquet'))

    return pivot_df


def vix_spreads(pivot_df):
    df = pd.DataFrame(index=pivot_df.index)
    for i in range(1, len([i for i in pivot_df.columns if 'Settle' in i])+1):
        for j in range(i + 1, len([i for i in pivot_df.columns if 'Settle' in i])+1):
            df[f'{i}-{j}'] =  pivot_df[f'{j}.0_Settle'] - pivot_df[f'{i}.0_Settle']
    # df.to_parquet(utils.get_abs_path('data/vix_spreads.parquet'))
    return df


def vix_active_contracts():
    vix_contracts = glob.glob(os.path.join(utils.get_abs_path('data/VIX'), "*.parquet"))
    lst = []
    for i in vix_contracts:
        lst.append(pd.read_parquet(i))
    vix_contracts = pd.concat(lst).pivot(columns='Futures', index='Trade Date', values='Settle')

    def convert_column_name(column_name):
        year = column_name.split()[-1][:-1]
        month = column_name.split()[1][1:]
        month_number = \
        {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
         'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}[month]
        return year + month_number

    vix_contracts.columns = [convert_column_name(col) for col in vix_contracts.columns]
    vix_contracts.replace(0, np.nan, inplace=True)
    vix_contracts.index = pd.to_datetime(vix_contracts.index).date

    vix_key = pd.DataFrame(None, columns=['1', '2', '3', '4', '5', '6', '7', '8'], index=vix_contracts.index)
    vix_key.index = pd.to_datetime(vix_key.index).date
    for index, row in vix_contracts.iterrows():
        # print(row.dropna())
        temp = list(row.dropna().index)
        temp.sort()
        temp = temp[:8] + [np.nan] * (8 - len(temp))
        vix_key.loc[index] = temp[:8]
    return vix_key, vix_contracts


def rt_vix_cleaner(df_):
    df = df_.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    latest_date = df['date'].iloc[-1]
    df['Expiration'] = pd.to_datetime(df['Expiration']).dt.date
    df = df[df.date==latest_date].sort_values(by='Expiration').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['Calendar_Num'] = df.groupby('date')['Expiration'].rank(ascending=True)

    def calculate_dfr(row, expiries):
        future_expiries = expiries[expiries <= row['date']]
        if not future_expiries.empty:
            return (row['Trade Date'] - future_expiries.max()).days
        else:
            return None
    df['DFR'] = df.apply(lambda row: calculate_dfr(row, df['Expiration']), axis=1)

    def calculate_dtr(row, expiries):
        future_expiries = expiries[expiries >= row['date']]
        if not future_expiries.empty:
            return (future_expiries.min() - row['date']).days
        else:
            return None
    df['DTR'] = df.apply(lambda row: calculate_dtr(row, df['Expiration']), axis=1)

    df = df[['date','close','LocalSymbol','Expiration','Calendar_Num','DTR','DFR']]
    df.rename(columns={'date':'Trade_Date','close':'Settle','Expiration':'Expiry'},inplace=True)

    return df


def main():
    pass


if __name__ == "__main__":
    main()