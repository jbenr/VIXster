import pandas as pd
import numpy as np
import utils

def load_data():
    print("\n=== Loading Data ===")

    macro_data = pd.read_parquet('data/fred.parquet').dropna()
    print(f"Pulled fred data. shape: {macro_data.shape}")
    macro_data.index = pd.to_datetime(macro_data.index).date

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

    # utils.pdf(spx_options_df.tail(10))
    # utils.pdf(macro_data.tail(10))
    # print(spx_options_df.columns)
    utils.pdf(vix_futures_df.tail(10))
    # utils.pdf(vix_spreads.tail(10))

    fut_feats = vix_futures_df[['Trade_Date','DTR','DFR']]
    fut_feats.set_index('Trade_Date', inplace=True)
    fut_feats.drop_duplicates(inplace=True)

    df = pd.merge(vix_spreads, macro_data, left_index=True, right_index=True)
    df = pd.merge(df, fut_feats, left_index=True, right_index=True)
    utils.pdf(df.tail(10))

    print("\n=== Data Loaded & Dates Parsed ===")

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

    vol_period = 21
    data[f'SP500_realized_vol_{vol_period}d'] = realized_volatility(data, column='SP500', window=vol_period)

    data.dropna(inplace=True)
    return data


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_regimes(data, n_clusters=4, cluster_features=['VIX', 'SP500_drawdown', 'IG']):
    X_cluster = StandardScaler().fit_transform(data[cluster_features])  # Standardize

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['ClusterRegime'] = kmeans.fit_predict(X_cluster)

    # Compute cluster percentages
    cluster_counts = data['ClusterRegime'].value_counts(normalize=True) * 100  # Convert to %
    data['ClusterPercentage'] = data['ClusterRegime'].map(cluster_counts)

    # Interpret clusters by looking at average feature values
    cluster_means = data.groupby('ClusterRegime')[cluster_features+['ClusterPercentage']].mean()
    utils.pdf(cluster_means.round(3))

    return data


if __name__ == "__main__":
    df = load_data()
    data = feature_engineer(df)
    utils.pdf(data.tail(5))
    cluster_features = ['VIX', 'SP500_5d', 'SP500_drawdown', 'IG']
    cluster_regimes(data, n_clusters=7, cluster_features=cluster_features)