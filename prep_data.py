import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import vixy
import utils
import talib
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from xgboost import XGBRegressor


def prep_X_y(params):
    # Process and pivot the VIX data
    y = vixy.process_vix()
    y = vixy.pivot_vix(y)
    y.index = pd.to_datetime(y.index).date

    # Initial feature selection from VIX data
    X = y[['DTR', 'DFR']]
    y = vixy.vix_spreads(y)
    y = y[['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']]

    # Load and process external market data
    fred = pd.read_parquet('data/fred.parquet')
    fred.index = pd.to_datetime(fred.index).date
    fred.index.name = 'Trade_Date'
    stonk = fred[['VIX', 'SP500', '10Y']].copy()
    for col in ['VIX', '10Y']:
        stonk[f'{col}_chg'] = fred[col].diff()
    for col in ['SP500']:
        stonk[f'{col}_pct_chg'] = fred[col].pct_change(fill_method=None)

    # Align and merge datasets
    shared_indexes = X.index.intersection(stonk.index)
    stonk = stonk.loc[shared_indexes]
    X = pd.merge(X, stonk, how='left', left_index=True, right_index=True).dropna()
    shared_indexes = X.index.intersection(y.index)
    X = X.loc[shared_indexes]
    y = y.loc[shared_indexes]

    # Generate technical indicators for each target spread
    bb_period = params.get("BB_period", 14)
    rsi_period = params.get("RSI_period", 14)
    dic_dat = {}
    for col in y.columns:
        tech = y[[col]].copy()
        tech[f'Upper_BB_{bb_period}'], tech[f'Middle_BB_{bb_period}'], tech[f'Lower_BB_{bb_period}'] = talib.BBANDS(
            tech[col], timeperiod=bb_period, nbdevup=2, nbdevdn=2, matype=0)
        tech[f'RSI_{rsi_period}'] = talib.RSI(tech[col], rsi_period) - 50
        tech.drop(columns=col, inplace=True)
        dic_dat[col] = tech.dropna()

    # Re-align X and y after technical indicators
    shared_indexes = dic_dat[y.columns[0]].index.intersection(X.index)
    X = X.loc[shared_indexes]
    y = y.loc[shared_indexes]

    return X, y, dic_dat


def load_spx_options_data(folder_path="data/spx_opt/"):
    dataframes = []

    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                all_files.append(os.path.join(root, file))

    for file_path in tqdm(all_files, desc="Loading SPX Option Files", unit="file"):
        try:
            df = pd.read_csv(file_path, delimiter=",", low_memory=False)
            df.dropna(axis=1, how="all", inplace=True)  # Drop empty columns
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if dataframes:
        full_df = pd.concat(dataframes, ignore_index=True)
        full_df.columns = full_df.columns.str.strip(" []")

        full_df["QUOTE_DATE"] = pd.to_datetime(full_df["QUOTE_DATE"], errors='coerce').dt.date
        full_df["EXPIRE_DATE"] = pd.to_datetime(full_df["EXPIRE_DATE"], errors='coerce').dt.date

        nums = ["DTE", "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA",
                "P_RHO", "C_IV", "P_IV", "STRIKE", "C_BID", "C_ASK", "P_BID", "P_ASK"]
        for num in nums:
            if num in full_df.columns:
                full_df[num] = pd.to_numeric(full_df[num], errors="coerce")

        for col in full_df.columns:
            if full_df[col].dtype == 'object':
                full_df[col] = full_df[col].astype(str)

        full_df.fillna(0, inplace=True)

        full_df.sort_values(by=['QUOTE_DATE','EXPIRE_DATE'], inplace=True)
        return full_df
    else:
        print("No valid data files found.")
        return pd.DataFrame()


def compute_option_features(spx_options_df):
    features = []
    for trade_date in tqdm(spx_options_df["Trade_Date"].unique(), desc="Computing Option Features"):
        daily_options = spx_options_df[spx_options_df["Trade_Date"] == trade_date]
        unique_expiries = daily_options["EXPIRE_DATE"].unique()

        for expiry in unique_expiries:
            expiry_options = daily_options[daily_options["EXPIRE_DATE"] == expiry]

            # ATM IV Calculation
            atm_strike = expiry_options.loc[
                (expiry_options["C_DELTA"].abs() - 0.5).abs().idxmin(), "STRIKE"
            ]
            atm_options = expiry_options[expiry_options["STRIKE"] == atm_strike]
            atm_iv = atm_options[["C_IV", "P_IV"]].mean().mean()

            # 25 Delta Skew & Risk Reversal
            calls_25d = expiry_options[(expiry_options["C_DELTA"] >= 0.23) & (expiry_options["C_DELTA"] <= 0.27)]
            puts_25d = expiry_options[(expiry_options["P_DELTA"] <= -0.23) & (expiry_options["P_DELTA"] >= -0.27)]
            skew_25 = calls_25d["C_IV"].mean() - puts_25d["P_IV"].mean() if not calls_25d.empty and not puts_25d.empty else np.nan

            # Vega Weighted IV
            vega_weighted_iv = (
                (expiry_options["C_IV"] * expiry_options["C_VEGA"] + expiry_options["P_IV"] * expiry_options["P_VEGA"]).sum()
                / (expiry_options["C_VEGA"].sum() + expiry_options["P_VEGA"].sum())
            )

            # Smile Slope (IV Difference Between 10 Delta and ATM)
            iv_10d = expiry_options[expiry_options["C_DELTA"] >= 0.10]["C_IV"].mean()
            smile_slope = iv_10d - atm_iv

            features.append(
                {
                    "Trade_Date": trade_date,
                    "OPTIONS_EXPIRY_DATE": expiry,
                    "ATM_IV": atm_iv,
                    "Skew_25": skew_25,
                    "Risk_Reversal_25": skew_25,
                    "Vega_Weighted_IV": vega_weighted_iv,
                    "Smile_Slope": smile_slope,
                }
            )

    # Convert to DataFrame
    features_df = pd.DataFrame(features)

    # **Fix Term Structure Calculation**
    term_structure = (
        features_df.groupby("Trade_Date")["ATM_IV"]
        .apply(lambda x: x.diff().bfill().ffill())  # ✅ Replaced deprecated method
        .reset_index(level=0, drop=True)  # ✅ Fix MultiIndex issue
    )

    # Assign Term Structure back to DataFrame
    features_df["Term_Structure"] = term_structure

    return features_df



def interpolate_features_to_vix_expiry(vix_futures_df, option_features_df):
    vix_fair_features = []

    for _, row in tqdm(vix_futures_df.iterrows(), desc="Interpolating Option Features"):
        trade_date = row["Trade_Date"]
        expiry_date = row["EXPIRY_DATE"]

        # Ensure option data exists for this trade date
        feature_data = option_features_df[option_features_df["Trade_Date"] == trade_date]
        if feature_data.empty:
            continue  # Skip if no matching trade date

        # Use closest two expiries for interpolation
        below = feature_data[feature_data["OPTIONS_EXPIRY_DATE"] <= expiry_date]
        above = feature_data[feature_data["OPTIONS_EXPIRY_DATE"] >= expiry_date]

        if below.empty or above.empty:
            continue  # Skip if no valid expiries found

        # Select nearest expiries
        T1, var1 = below.iloc[-1][["OPTIONS_EXPIRY_DATE", "ATM_IV"]]
        T2, var2 = above.iloc[0][["OPTIONS_EXPIRY_DATE", "ATM_IV"]]

        T1, T2 = (T1 - trade_date).days, (T2 - trade_date).days  # Convert to days

        # Interpolate each feature
        interpolated_features = {
            feature: np.interp(
                (expiry_date - trade_date).days,
                [T1, T2],
                [below.iloc[-1][feature], above.iloc[0][feature]],
            )
            for feature in ["ATM_IV", "Skew_25", "Risk_Reversal_25", "Vega_Weighted_IV", "Smile_Slope", "Term_Structure"]
        }

        # Append results
        vix_fair_features.append(
            {
                "Trade_Date": trade_date,
                "EXPIRY_DATE": expiry_date,
                **interpolated_features,
                "VIX_FUTURES_PRICE": row.get("Close", np.nan),  # Ensure column exists
            }
        )

    return pd.DataFrame(vix_fair_features)


def compute_feature_deltas(vix_feature_df, window=5):
    """
    Computes the change in features over a rolling window.
    """
    vix_feature_df = vix_feature_df.sort_values(by=['Trade_Date', 'EXPIRY_DATE'])
    feature_cols = ['ATM_IV', 'Skew_25', 'Risk_Reversal_25', 'Vega_Weighted_IV', 'Smile_Slope', 'Term_Structure']

    for feature in feature_cols:
        vix_feature_df[f"{feature}_chg"] = vix_feature_df.groupby('EXPIRY_DATE')[feature].diff(periods=window)

    vix_feature_df['VIX_FUTURES_PRICE_CHG'] = vix_feature_df.groupby('EXPIRY_DATE')['VIX_FUTURES_PRICE'].diff(
        periods=window)

    return vix_feature_df.dropna()


def train_vix_futures_model(vix_feature_df):
    """
    Train XGBoost model on VIX futures feature deltas.
    """
    X = vix_feature_df[['ATM_IV_chg', 'Skew_25_chg', 'Risk_Reversal_25_chg', 'Vega_Weighted_IV_chg', 'Smile_Slope_chg',
                        'Term_Structure_chg']]
    y = vix_feature_df['VIX_FUTURES_PRICE_CHG']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"VIX Futures Prediction MSE: {mse}")

    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
    print("Model Predictions vs Actuals:")
    print(predictions_df.head(10))

    return model, predictions_df


def main():
    macro_data = pd.read_parquet('data/fred.parquet').dropna()
    print(f"Pulled fred data len: {len(macro_data)}")
    macro_data.index = pd.to_datetime(macro_data.index).date

    spx_options_df = pd.read_parquet("data/spx_options.parquet")
    print(f"Pulled SPX options data len: {len(spx_options_df)}")
    spx_options_df["QUOTE_DATE"] = pd.to_datetime(spx_options_df["QUOTE_DATE"]).dt.date
    spx_options_df["EXPIRE_DATE"] = pd.to_datetime(spx_options_df["EXPIRE_DATE"]).dt.date

    vix_futures_df = pd.read_parquet("data/vix.parquet")
    print(f"Pulled VIX futures data len: {len(vix_futures_df)}")
    vix_futures_df["Trade_Date"] = pd.to_datetime(vix_futures_df["Trade_Date"]).dt.date

    vix_spreads = pd.read_parquet("data/vix_spreads.parquet")[
        ['1-2','2-3','3-4','4-5','5-6','6-7','7-8']
    ]
    print(f"Pulled VIX spreads data len: {len(vix_spreads)}")

    common_trade_dates = sorted(set(spx_options_df["QUOTE_DATE"])
                                .intersection(vix_spreads.index)
                                .intersection(vix_futures_df['Trade_Date'])
                                .intersection(macro_data.index))[300:400]
    print(f"Filtered to first {len(common_trade_dates)} common trade dates")
    macro_data = macro_data.loc[common_trade_dates]
    spx_options_df = spx_options_df[spx_options_df["QUOTE_DATE"].isin(common_trade_dates)]
    vix_futures_df = vix_futures_df[vix_futures_df["Trade_Date"].isin(common_trade_dates)]
    vix_spreads = vix_spreads.loc[common_trade_dates]

    # utils.pdf(spx_options_df.tail(10))
    utils.pdf(macro_data.tail(10))
    print(spx_options_df.columns)
    utils.pdf(vix_futures_df.tail(10))
    utils.pdf(vix_spreads.tail(10))

    # option_features_df = compute_option_features(spx_options_df)
    # print("Computed option features")
    # utils.pdf(option_features_df.tail(10))
    #
    # vix_feature_df = interpolate_features_to_vix_expiry(vix_futures_df, option_features_df)
    # print("Interpolated features to VIX expiry")
    # utils.pdf(vix_feature_df.tail(10))
    #
    # vix_feature_df = compute_feature_deltas(vix_feature_df)
    # print("Computed feature deltas")
    # utils.pdf(vix_feature_df.tail(10))
    #
    # model, predictions_df = train_vix_futures_model(vix_feature_df)
    # print("Generated model predictions")
    # utils.pdf(predictions_df.head(10))


if __name__ == "__main__":
    main()

