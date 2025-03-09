import pandas as pd
import numpy as np
import utils
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import talib
import joblib


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
    utils.pdf(macro_data.tail(10))
    # print(spx_options_df.columns)
    utils.pdf(vix_futures_df.tail(10))
    utils.pdf(vix_spreads.tail(10))

    print("\n=== Data Loaded & Dates Parsed ===")

    return (macro_data,
            # spx_options_df,
            vix_futures_df, vix_spreads)


from hmmlearn.hmm import GaussianHMM
def cluster_macro_hmm(macro_scaled):
    best_k, best_bic = 2, np.inf  # Start with 2 regimes
    print("=== Running HMM Clustering ===")

    for k in tqdm(range(2, 20), desc="Finding optimal clusters (HMM)", unit="state"):
        hmm = GaussianHMM(n_components=k, covariance_type="full", n_iter=100, random_state=42)
        hmm.fit(macro_scaled)  # Fit model
        bic = hmm.bic(macro_scaled)  # Compute BIC
        if bic < best_bic:
            best_k, best_bic = k, bic

    print(f"Optimal number of macro clusters (HMM): {best_k}")

    # Train final HMM with best_k clusters
    hmm = GaussianHMM(n_components=best_k, covariance_type="full", n_iter=100, random_state=42)
    hmm.fit(macro_scaled)
    joblib.dump(hmm, 'data/features/macro_hmm.pkl')

    predicted_regimes = hmm.predict(macro_scaled)
    return predicted_regimes, best_k  # Return cluster assignments


def compute_macro_features(macro_df, cols):
    print("=== Computing Macro Features ===")
    macro_df = macro_df.copy()

    # Compute Daily Changes
    macro_df['10Y_chg'] = macro_df['10Y'].diff()
    macro_df['VIX_pct_chg'] = macro_df['VIX'].pct_change()
    macro_df['SP500_pct_chg'] = macro_df['SP500'].pct_change()
    macro_df['2s10s'] = macro_df['10Y'] - macro_df['2Y']
    macro_df['BTC_pct_chg'] = macro_df['BTC'].pct_change()

    # Compute Rolling Averages & Volatility
    window_sizes = [5, 20]
    for window in window_sizes:
        for col in ['10Y_chg', 'VIX_pct_chg', 'SP500_pct_chg', '2s10s', 'BTC_pct_chg']:
            macro_df[f'{col}_mean_{window}'] = macro_df[col].rolling(window).mean()
            macro_df[f'{col}_std_{window}'] = macro_df[col].rolling(window).std()

    # Compute Macro Shock Features (Z-Scores)
    z_score_period = 30
    for col in ['10Y_chg', 'VIX_pct_chg', 'SP500_pct_chg', '2s10s', 'BTC_pct_chg']:
        macro_df[f'{col}_{z_score_period}d_zscore'] = (macro_df[col] - macro_df[col].rolling(z_score_period).mean()) / macro_df[col].rolling(z_score_period).std()

    # Compute Streak Features (Consecutive Up/Down Days)
    for col in ['SP500_pct_chg', 'VIX_pct_chg', '10Y_chg', 'BTC_pct_chg']:
        streaks = np.zeros(len(macro_df))
        for i in range(1, len(macro_df)):
            if np.isnan(macro_df[col].iloc[i]):
                continue
            if macro_df[col].iloc[i] > 0:
                streaks[i] = streaks[i - 1] + 1 if streaks[i - 1] > 0 else 1
            elif macro_df[col].iloc[i] < 0:
                streaks[i] = streaks[i - 1] - 1 if streaks[i - 1] < 0 else -1
            else:
                streaks[i] = 0  # Reset on zero change
        macro_df[f'{col}_streak'] = streaks

    macro_df.dropna(inplace=True)
    print(macro_df.columns)

    moar = [
        'SP500_pct_chg_30d_zscore','VIX_pct_chg_30d_zscore','10Y_chg_30d_zscore','BTC_pct_chg_30d_zscore',
        'SP500_pct_chg_streak','VIX_pct_chg_streak','10Y_chg_streak','BTC_pct_chg_streak',
        'SP500_pct_chg_std_20','VIX_pct_chg_std_20'
    ]

    # Determine Optimal Number of Clusters for Macro Regimes
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df[cols+moar].dropna(subset=cols))
    joblib.dump(scaler, 'data/features/macro_scaler.pkl')

    clustered_df, best_k = cluster_macro_hmm(macro_scaled)
    print(f"Optimal number of macro clusters: {best_k}")

    macro_df.loc[macro_df.dropna().index, 'Macro_Regime'] = clustered_df

    cluster_counts = macro_df['Macro_Regime'].value_counts(normalize=True) * 100
    cluster_summary = macro_df[cols + moar + ['Macro_Regime']].groupby('Macro_Regime').mean()
    cluster_summary['Cluster_Percentage'] = cluster_summary.index.map(cluster_counts)
    print(cluster_summary)

    print("=== Macro Features Computed ===")
    return macro_df.dropna()[cols + moar + ['Macro_Regime']]


def compute_spread_features(spreads_df, lookback_windows=[5, 10, 20]):
    print("=== Computing Spread Features ===")
    spreads_df = spreads_df.copy()

    spread_cols = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']

    new_features = {}  # Store new columns in a dict to avoid fragmentation

    # Compute rolling features for each spread
    for spread in spread_cols:
        # Momentum & Volatility
        for window in lookback_windows:
            new_features[f'{spread}_mean_{window}'] = spreads_df[spread].rolling(window).mean()
            new_features[f'{spread}_std_{window}'] = spreads_df[spread].rolling(window).std()
            new_features[f'{spread}_zscore_{window}'] = (
                    (spreads_df[spread] - new_features[f'{spread}_mean_{window}']) /
                    (new_features[f'{spread}_std_{window}'] + 1e-6)  # Avoid div by zero
            )

        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = talib.BBANDS(spreads_df[spread], timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
        new_features[f'{spread}_upper_bb'] = upper_bb
        new_features[f'{spread}_lower_bb'] = lower_bb

        # RSI
        new_features[f'{spread}_rsi'] = talib.RSI(spreads_df[spread], timeperiod=14) - 50

    # Add time-based features
    new_features['day_of_week'] = pd.to_datetime(spreads_df.index).dayofweek
    new_features['day_of_month'] = pd.to_datetime(spreads_df.index).day
    new_features['month'] = pd.to_datetime(spreads_df.index).month
    # new_features['quarter'] = pd.to_datetime(spreads_df.index).quarter

    # ** Efficiently join all features back into the original DataFrame **
    spreads_df = pd.concat([spreads_df, pd.DataFrame(new_features, index=spreads_df.index)], axis=1)

    print("=== Spread Features Computed ===")
    return spreads_df.dropna()


import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
def evaluate_feature_importance(spreads_df):
    print("=== Evaluating Feature Importance (Time-Series Aware) ===")

    target_spreads = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']

    # Ensure 'Trade_Date' is handled properly
    if 'Trade_Date' in spreads_df.columns:
        spreads_df = spreads_df.set_index('Trade_Date')

    features = spreads_df.drop(columns=target_spreads, errors='ignore').select_dtypes(include=[np.number])

    important_features_dict = {}

    for spread in target_spreads:
        print(f"\n🔍 Evaluating features for {spread} spread...")

        target = spreads_df[spread]

        # Drop rows with missing values
        data = pd.concat([features, target], axis=1).dropna()
        X, y = data.drop(columns=[spread]), data[spread]

        # Time-Series Split (instead of random split)
        tscv = TimeSeriesSplit(n_splits=5)

        feature_importance = np.zeros(X.shape[1])

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train XGBoost model
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            # Compute SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Aggregate feature importance
            feature_importance += np.abs(shap_values).mean(axis=0)

        # Normalize importance
        feature_importance /= tscv.get_n_splits()

        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        # Keep only the top 10 features
        top_features = feature_importance_df.head(20)['Feature'].tolist()
        important_features_dict[spread] = top_features

        print(f"✅ Top Features for {spread} Spread:")
        print(feature_importance_df.head(20))

        # Plot SHAP Feature Importance
        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Feature Importance for {spread} Spread")
        plt.show()

    return important_features_dict


def prepare_model_data(macro_df, spreads_df):
    print("=== Preparing Model Data ===")

    if spreads_df.index.name == 'Trade_Date': spreads_df = spreads_df.reset_index()
    macro_df.index.name = 'Trade_Date'
    if macro_df.index.name == 'Trade_Date': macro_df = macro_df.reset_index()

    # Check columns before melting
    print(f"✅ Columns in spreads_df BEFORE melt: {spreads_df.columns}")
    print(f"✅ Columns in macro_df: {macro_df.columns}")

    # Melt spread features into long format
    spreads_df_long = spreads_df.melt(
        id_vars=['Trade_Date', 'DTR', 'DFR'],  # Keep trade date, days-to-roll, and days-from-roll
        value_vars=['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8'],
        var_name='Spread_ID',
        value_name='Spread_Value'
    )

    # Melt spread-specific technical indicators (mean, std, z-score, RSI, Bollinger Bands, etc.)
    spread_features = [col for col in spreads_df.columns if
                       any(f"{s}_" in col for s in ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8'])]
    print("Spread features: ",spread_features)

    feature_dfs = []
    for feature in tqdm(spread_features, desc="Processing Spread Features"):
        temp_df = spreads_df.melt(
            id_vars=['Trade_Date'],
            value_vars=[feature],
            var_name='Spread_ID',
            value_name=feature
        )
        feature_dfs.append(temp_df)

    # Merge all melted feature DataFrames
    spread_features_long = feature_dfs[0]
    for df in tqdm(feature_dfs[1:], desc="Merging Spread Feature DataFrames"):
        spread_features_long = spread_features_long.merge(df, on=['Trade_Date', 'Spread_ID'], how='left')

    # Merge macro data into the spread dataset
    final_df = spreads_df_long.merge(spread_features_long, on=['Trade_Date', 'Spread_ID'], how='left')
    final_df = final_df.merge(macro_df, on='Trade_Date', how='left')

    # Check final dataset
    print(f"✅ Columns in spreads_df_long AFTER melt: {spreads_df_long.columns}")
    print(f"✅ Columns in final_df AFTER merging macro: {final_df.columns}")

    return final_df


def main():
    (macro_df,
     # spx_options_df,
     vix_futures_df, vix_spreads_df) = load_data()
    utils.make_dir('data/features')

    macro_cols = [
        '10Y_chg','VIX_pct_chg','SP500_pct_chg','2s10s','BTC_pct_chg'
    ]
    macro_df = compute_macro_features(macro_df, macro_cols)
    print(list(macro_df.columns))
    # print(macro_df.tail(5))
    macro_df.to_parquet('data/features/macro_features.parquet')

    spreads_df = compute_spread_features(vix_spreads_df)
    # merge DFR and DTR on spreads
    roll_days = vix_futures_df[['Trade_Date','DTR','DFR']].drop_duplicates()
    roll_days.set_index('Trade_Date', inplace=True)
    spreads_df = pd.merge(spreads_df, roll_days, how='left', left_index=True, right_index=True)
    # print(list(spreads_df.columns))

    important_features = evaluate_feature_importance(spreads_df)
    for key in important_features.keys():
        print(key, important_features[key])

    # print(spreads_df.tail(5))
    spreads_df.to_parquet('data/features/spread_features.parquet')

    print("\n=== Data Preprocessing Complete ===")

    final_df = prepare_model_data(macro_df, spreads_df)
    print(final_df.tail(3))
    # utils.pdf(final_df.tail(3))

    final_df.to_csv('data/features/final_df.csv')

if __name__ == "__main__":
    main()
