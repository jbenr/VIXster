import math

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


from sklearn.mixture import GaussianMixture
def cluster_macro_gmm(macro_df, n_components=3):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    regimes = gmm.fit_predict(macro_df)

    return regimes

def find_optimal_gmm_clusters(macro_scaled, max_clusters=10):
    print("🔍 Finding Optimal GMM Clusters...")

    bic_scores = []
    aic_scores = []
    n_components_range = range(1, max_clusters+1)

    for n in tqdm(n_components_range, desc="Evaluating GMM clusters"):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(macro_scaled)

        bic_scores.append(gmm.bic(macro_scaled))
        aic_scores.append(gmm.aic(macro_scaled))

    # Optimal cluster count (lowest BIC)
    optimal_n = n_components_range[np.argmin(bic_scores)]
    print(f"✅ Optimal number of clusters based on BIC: {optimal_n}")

    return optimal_n


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

    # clustered_df, best_k = cluster_macro_hmm(macro_scaled)
    # print(f"Optimal number of macro clusters: {best_k}")
    optimal_clusters = find_optimal_gmm_clusters(macro_scaled)
    clustered_df = cluster_macro_gmm(macro_scaled, optimal_clusters)
    print(clustered_df)

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
    print(spreads_df.tail(3))

    print("=== Spread Features Computed ===")
    return spreads_df.dropna()


import shap
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
def evaluate_feature_importance(spreads_df, corr_threshold, importance_threshold):
    # print("=== Evaluating Feature Importance (Time-Series Aware) ===")

    target_spreads = ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']
    if 'Trade_Date' in spreads_df.columns: spreads_df = spreads_df.set_index('Trade_Date')
    features = spreads_df.drop(columns=target_spreads, errors='ignore').select_dtypes(include=[np.number])
    important_features_dict = {}

    for spread in target_spreads:
        target = spreads_df[spread]
        data = pd.concat([features, target], axis=1).dropna()
        X, y = data.drop(columns=[spread]), data[spread]

        tscv = TimeSeriesSplit(n_splits=5)

        feature_importance = np.zeros(X.shape[1])

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            feature_importance += np.abs(shap_values).mean(axis=0)

        # Normalize importance
        feature_importance /= tscv.get_n_splits()

        # Create a DataFrame with feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        # Keep only features with importance above threshold
        feature_importance_df = feature_importance_df[feature_importance_df['Importance'] >= importance_threshold]

        # Compute correlation matrix
        selected_features = feature_importance_df['Feature'].tolist()
        if len(selected_features) > 1:
            corr_matrix = X[selected_features].corr().abs()
        else:
            corr_matrix = None

        # Find and remove highly correlated features
        to_remove = set()
        if corr_matrix is not None:
            for i in range(len(selected_features)):
                for j in range(i + 1, len(selected_features)):
                    if corr_matrix.iloc[i, j] > corr_threshold:
                        # Drop the feature with the lower importance score
                        if feature_importance_df.loc[feature_importance_df['Feature'] == selected_features[i], 'Importance'].values[0] < \
                           feature_importance_df.loc[feature_importance_df['Feature'] == selected_features[j], 'Importance'].values[0]:
                            to_remove.add(selected_features[i])
                        else:
                            to_remove.add(selected_features[j])

        # Remove selected features
        refined_features = feature_importance_df[~feature_importance_df['Feature'].isin(to_remove)]

        # Store in dictionary
        important_features_dict[spread] = refined_features['Feature'].tolist()

        # print(f"✅ Top Features for {spread} Spread (After Correlation Filtering):")
        # print(refined_features)

    return important_features_dict


def generate_feature_selection_dict(spreads_df, macro_df, corr_threshold=0.8, importance_threshold=0.03):
    print("\n=== Generating Feature Selection Dictionary ===")

    feature_selection_dict = {spread: set() for spread in ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']}
    spreads_df_feat = pd.merge(spreads_df, macro_df, how='left', left_index=True, right_index=True)

    # Get unique non-NaN Macro_Regimes
    macro_regimes = [c for c in spreads_df_feat['Macro_Regime'].unique() if not math.isnan(c)]

    # Track progress using tqdm
    for clust in tqdm(macro_regimes, desc="Processing features for Macro Regimes"):
        # print(f"\n🔍 Evaluating Feature Importance for Macro Regime: {clust}")

        cluster_df = spreads_df_feat[spreads_df_feat['Macro_Regime'] == clust]
        important_features = evaluate_feature_importance(cluster_df, corr_threshold, importance_threshold)

        for spread, features in important_features.items():
            feature_selection_dict[spread].update(features)

    feature_selection_dict = {spread: list(features) for spread, features in feature_selection_dict.items()}

    print("\n✅ Final Feature Selection Dictionary:")
    for spread, features in feature_selection_dict.items():
        print(f"{spread}: {features}")

    return feature_selection_dict


def prepare_model_data(macro_df, spreads_df, optimal_features):
    print("=== Preparing Model Data ===")

    if spreads_df.index.name == 'Trade_Date': spreads_df = spreads_df.reset_index()
    macro_df.index.name = 'Trade_Date'
    if macro_df.index.name == 'Trade_Date': macro_df = macro_df.reset_index()

    # Melt spread values into long format
    spreads_df_long = spreads_df.melt(
        id_vars=['Trade_Date', 'DTR', 'DFR', 'day_of_week', 'day_of_month', 'month'],  # Retain key columns
        value_vars=optimal_features.keys(),  # Use only spreads in optimal_features
        var_name='Spread_ID',
        value_name='Spread_Value'
    )

    # Filter spread features dynamically based on optimal_features
    feature_dfs = []
    for spread, features in optimal_features.items():
        selected_features = [f for f in features if f in spreads_df.columns]  # Ensure features exist
        if not selected_features:
            continue  # Skip if no valid features for this spread

        temp_df = spreads_df[['Trade_Date'] + selected_features].melt(
            id_vars=['Trade_Date'],
            value_vars=selected_features,
            var_name='Feature_Name',
            value_name='Feature_Value'
        )
        temp_df['Spread_ID'] = spread  # Assign corresponding spread

        feature_dfs.append(temp_df)

    # Merge all feature DataFrames into one long-form DataFrame
    if feature_dfs:
        spread_features_long = pd.concat(feature_dfs, ignore_index=True)
    else:
        raise ValueError("No valid spread features found for any spread in optimal_features!")

    # Merge spreads, features, and macro data
    final_df = spreads_df_long.merge(spread_features_long, on=['Trade_Date', 'Spread_ID'], how='left')
    final_df = final_df.merge(macro_df, on='Trade_Date', how='left')

    final_df.sort_values(by=['Trade_Date', 'Spread_ID', 'Feature_Name'], inplace=True)
    return final_df



def main():
    (macro_df,
     # spx_options_df,
     vix_futures_df, vix_spreads_df) = load_data()

    utils.make_dir('data/features')

    macro_cols = [
        '10Y_chg','VIX_pct_chg','SP500_pct_chg','2s10s','BTC_pct_chg'
    ]
    macro_df = compute_macro_features(macro_df, macro_cols).dropna()
    macro_df.to_parquet('data/features/macro_features.parquet')

    spreads_df = compute_spread_features(vix_spreads_df)
    # merge DFR and DTR on spreads
    roll_days = vix_futures_df[['Trade_Date','DTR','DFR']].drop_duplicates()
    roll_days.set_index('Trade_Date', inplace=True)
    spreads_df = pd.merge(spreads_df, roll_days, how='left', left_index=True, right_index=True)

    optimal_features = generate_feature_selection_dict(spreads_df, macro_df, corr_threshold=0.9, importance_threshold=0.02)
    for key in optimal_features.keys():
        print(f"{key} in spreads_df.columns: {key in spreads_df.columns}")

    # print(spreads_df.tail(5))
    spreads_df.to_parquet('data/features/spread_features.parquet')

    print("\n=== Data Preprocessing Complete ===")

    final_df = prepare_model_data(macro_df, spreads_df, optimal_features)
    utils.pdf(final_df.tail(75))

    final_df.to_csv('data/features/final_df.parquet')

if __name__ == "__main__":
    main()
