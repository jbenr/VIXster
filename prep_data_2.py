import pandas as pd
import numpy as np
import utils
import yahoooo
from yahoooo import get_price
from datetime import datetime

def load_data(live=False):
    print("\n=== Loading Data ===")

    macro_data = pd.read_parquet('data/fred.parquet')
    print(f"Pulled fred data. shape: {macro_data.shape}")
    macro_data.index = pd.to_datetime(macro_data.index).date
    # macro_data.dropna(inplace=True)

    # spx_options_df = pd.read_parquet("data/spx_options.parquet")
    # print(f"Pulled SPX options data. shape: {spx_options_df.shape}")
    # spx_options_df["QUOTE_DATE"] = pd.to_datetime(spx_options_df["QUOTE_DATE"]).dt.date
    # spx_options_df["EXPIRE_DATE"] = pd.to_datetime(spx_options_df["EXPIRE_DATE"]).dt.date

    vix_futures_df = pd.read_parquet("data/vix.parquet")
    print(f"Pulled VIX futures data. shape: {vix_futures_df.shape}")
    vix_futures_df["Trade_Date"] = pd.to_datetime(vix_futures_df["Trade_Date"]).dt.date

    vix_spreads = pd.read_parquet("data/vix_spreads.parquet")[
        ['1-2','2-3','3-4','4-5','5-6','6-7','7-8']
    ]
    print(f"Pulled VIX spreads data. shape: {vix_spreads.shape}")

    # common_trade_dates = sorted(set(spx_options_df["QUOTE_DATE"])
    #                             .intersection(vix_spreads.index)
    #                             .intersection(vix_futures_df['Trade_Date'])
    #                             .intersection(macro_data.index))
    common_trade_dates = sorted(set(vix_spreads.index)
                                .intersection(vix_futures_df['Trade_Date'])
                                .intersection(macro_data.index))
    print(f"Filtered to first {len(common_trade_dates)} common trade dates")

    macro_data = macro_data.loc[common_trade_dates]
    # spx_options_df = spx_options_df[spx_options_df["QUOTE_DATE"].isin(common_trade_dates)]
    vix_futures_df = vix_futures_df[vix_futures_df["Trade_Date"].isin(common_trade_dates)]
    vix_spreads = vix_spreads.loc[common_trade_dates]

    fut_feats = vix_futures_df[['Trade_Date','DTR','DFR']].drop_duplicates()
    fut_feats.set_index('Trade_Date', inplace=True)

    df = pd.merge(vix_spreads, macro_data, left_index=True, right_index=True)
    df = pd.merge(df, fut_feats, left_index=True, right_index=True)

    ### Insert live data logic
    if live:
        vx = yahoooo.get_price('^VIX')
        vx.columns = ['VIX']
        sp = yahoooo.get_price('^GSPC')
        sp.columns = ['SP500']
        print(vx, sp)

        live_spreads = pd.read_parquet('sheet/spreads.parquet')
        max_date = live_spreads['Last Update'].max()
        live_spreads[max_date] = ((live_spreads['Bid Size'] * live_spreads['Bid Price'])\
        + (live_spreads['Ask Size'] * live_spreads['Ask Price']))\
        / (live_spreads['Bid Size'] + live_spreads['Ask Size'])
        live_spreads = live_spreads[['Spread',max_date]].set_index("Spread").T
        live_spreads.index = pd.to_datetime(live_spreads.index).date

        live_spreads = pd.merge(live_spreads, vx, left_index=True, right_index=True, how='left')
        live_spreads = pd.merge(live_spreads, sp, left_index=True, right_index=True, how='left')

        max_date_ = datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S").date()
        for col in live_spreads.columns:
            if pd.isna(live_spreads.loc[max_date_, col]):
                try:
                    price_df = get_price(col)
                    if price_df is not None:
                        live_price = price_df["Close"].iloc[0]
                        live_spreads.at[max_date_, col] = live_price
                        print(f"Filled {col} on {max_date_} → {live_price}")
                except Exception as e:
                    print(f"Couldn’t get a live price for {col}: {e}")

        df = df[['1-2','2-3','3-4','4-5','5-6','6-7','7-8','VIX','SP500','DTR','DFR']]
        df = pd.concat([df,live_spreads])

        new_date = df.index[-1]

        prev_date = df.index[-2]
        days_diff = (pd.to_datetime(new_date) - pd.to_datetime(prev_date)).days

        df.at[new_date, 'DTR'] = df.at[prev_date, 'DTR'] - days_diff
        df.at[new_date, 'DFR'] = df.at[prev_date, 'DFR'] + days_diff

        df = df.fillna(0)

    utils.pdf(df.tail(3))
    print("=== Data Loaded & Dates Parsed ===\n")

    return df


import talib

def feature_engineer(df):
    data = df.copy()
    data.index = pd.to_datetime(data.index)

    data['SP500_5d'] = data['SP500'].pct_change(5)
    data['SP500_1d'] = data['SP500'].pct_change(1)
    data['SP500_peak'] = data['SP500'].cummax()
    data['SP500_drawdown'] = (data['SP500_peak'] - data['SP500']) / data['SP500']  # drawdown from peak

    data['Month'] = data.index.month
    month_dummies = pd.get_dummies(data['Month'], prefix='Month', drop_first=True)
    data = pd.concat([data.drop('Month', axis=1), month_dummies], axis=1)

    def realized_volatility(data, column='SP500', window=21):
        log_returns = np.log(data[column] / data[column].shift(1))  # Calculate log returns
        realized_vol = log_returns.rolling(window=window).std() * np.sqrt(252)  # Apply formula
        return realized_vol

    for vol_period in [7,14,21,50]:
        data[f'SP500_realized_vol_{vol_period}d'] = realized_volatility(data, column='SP500', window=vol_period)

    data.dropna(inplace=True)
    return data


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_regimes(data, n_clusters=4, cluster_features=['VIX', 'SP500_drawdown', 'IG']):
    X_cluster = StandardScaler().fit_transform(data[cluster_features])  # Standardize

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['ClusterRegime'] = kmeans.fit_predict(X_cluster)

    cluster_counts = data['ClusterRegime'].value_counts()  # Number of samples per cluster
    cluster_percentages = (cluster_counts / len(data)) * 100  # Convert to percentage

    data['ClusterPercentage'] = data['ClusterRegime'].map(cluster_percentages)

    cluster_means = data.groupby('ClusterRegime')[cluster_features].mean()

    cluster_summary = cluster_means.copy()
    cluster_summary['SampleCount'] = cluster_counts
    cluster_summary['ClusterPercentage'] = cluster_percentages.map(lambda x: f"{x:.1f}%")
    utils.pdf(cluster_summary.round(3))

    return data


if __name__ == "__main__":
    df = load_data(live=True)
    print("+++ Lock and load")
    utils.pdf(df.tail(3))
    data = feature_engineer(df)
    # data.to_parquet('data/live_features.parquet')
    utils.pdf(data.tail(5))
    cluster_features = ['VIX', 'SP500_5d', 'SP500_drawdown']
    cluster_regimes(data, n_clusters=7, cluster_features=cluster_features)