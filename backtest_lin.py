"""
backtest_lin.py

A simple backtesting script for saved (pickled) linear models that forecast
VIX calendar spreads. We load each model, generate predictions for the past
2 years, and simulate a basic trading strategy. Customize as needed!
"""

import pickle
import pandas as pd
import numpy as np
import os

# ------------------------------------------------------------------------------------
# HELPER: Expand 'SEASONALITY_BLOCK' into actual columns that start with "Month_"
# ------------------------------------------------------------------------------------
def expand_seasonality_block(feature_cols, df):
    if "SEASONALITY_BLOCK" in feature_cols:
        month_cols = [c for c in df.columns if c.startswith("Month_")]
        new_cols = [c for c in feature_cols if c != "SEASONALITY_BLOCK"]
        new_cols += month_cols
        return new_cols
    else:
        return feature_cols

# ------------------------------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------------------------------
def load_linear_model(pickle_path):
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)
    return model

# ------------------------------------------------------------------------------------
# GENERATE PREDICTIONS
# ------------------------------------------------------------------------------------
def generate_predictions(model, df_features, feature_cols, target_spread_col):
    expanded_feats = expand_seasonality_block(feature_cols, df_features)
    df = df_features.copy()

    valid_idx = df.dropna(subset=expanded_feats + [target_spread_col]).index
    df = df.loc[valid_idx]

    X = df[expanded_feats]
    df["PredictedSpread"] = model.predict(X)
    df.to_parquet('data/backtest/backtest_lin_predictions.parquet')

    return df

# ------------------------------------------------------------------------------------
# BUILD TRADING SIGNAL
# ------------------------------------------------------------------------------------
def build_trading_signal(df, target_spread_col, pred_spread_col, lookback=20, z_threshold=1.0):
    """
    Example strategy:
    if PredictedSpread - ActualSpread > threshold => LONG
    if ActualSpread - PredictedSpread > threshold => SHORT
    else => FLAT
    """
    df = df.copy()
    df["SpreadResid"] = df[pred_spread_col] - df[target_spread_col]

    df["SpreadChange"] = df[target_spread_col].diff()
    df["SpreadVol"] = df["SpreadChange"].rolling(window=lookback).std()

    #    watch out for zero / nan vol
    df["Zscore"] = df["SpreadResid"] / df["SpreadVol"]

    # position
    df["Position"] = 0
    df.loc[df["Zscore"] > z_threshold, "Position"] = 1
    df.loc[df["Zscore"] < -z_threshold, "Position"] = -1

    return df

# ------------------------------------------------------------------------------------
# SIMULATE PNL
# ------------------------------------------------------------------------------------
def simulate_pnl(df, target_spread_col, position_col="Position"):
    """
    A simple daily PnL simulation:
    - Buy/sell spread if Position=1/-1 from day t to t+1
    - PnL = position * (Spread_{t+1} - Spread_t)
    """
    df = df.copy().sort_index()
    df["DailyPnL"] = 0.0

    spread_change = df[target_spread_col].diff()
    position = df[position_col].shift(1)

    df["DailyPnL"] = spread_change * position
    df["DailyPnL"] = df["DailyPnL"].fillna(0)

    df["CumulativePnL"] = df["DailyPnL"].cumsum()

    return df

# ------------------------------------------------------------------------------------
# EVALUATE PERFORMANCE
# ------------------------------------------------------------------------------------
def evaluate_performance(df, pnl_col="DailyPnL"):
    """
    Calculate summary stats: total return, Sharpe, win rate, max drawdown, etc.
    """
    ann_factor = 252  # if daily
    daily_returns = df[pnl_col]

    total_pnl = daily_returns.sum()
    mean_pnl = daily_returns.mean()
    std_pnl = daily_returns.std()
    sharpe = np.nan
    if std_pnl != 0:
        sharpe = (mean_pnl * ann_factor) / (std_pnl * np.sqrt(ann_factor))

    wins = (daily_returns > 0).sum()
    losses = (daily_returns < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) else 0

    cum_pnl = df["CumulativePnL"]
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak
    max_drawdown = drawdown.min()

    perf = {
        "TotalPnL": total_pnl,
        "MeanDailyPnL": mean_pnl,
        "StdDailyPnL": std_pnl,
        "SharpeRatio": sharpe,
        "WinRate": win_rate,
        "MaxDrawdown": max_drawdown
    }
    return perf

# ------------------------------------------------------------------------------------
# RUN BACKTEST
# ------------------------------------------------------------------------------------
def run_backtest(
        model_paths,
        df_data,
        feature_cols_dict,
        target_spread_cols,
        out_path="data/backtest/"
):
    results = []

    for spread, pkl_path in model_paths.items():
        print(f"\n=== Backtesting Model for Spread {spread} ===")

        if not os.path.exists(pkl_path):
            print(f"Model file not found: {pkl_path}. Skipping.")
            continue
        model = load_linear_model(pkl_path)

        if spread not in feature_cols_dict:
            print(f"No feature columns provided for spread {spread}. Skipping.")
            continue
        feats = feature_cols_dict[spread]

        if spread not in df_data.columns:
            print(f"DataFrame has no column for {spread} spread. Skipping.")
            continue

        # 1) Generate predictions
        df_preds = generate_predictions(
            model=model,
            df_features=df_data,
            feature_cols=feats,
            target_spread_col=spread
        )

        # e.g. store columns: [index], [spread], [PredictedSpread], plus any features
        pred_path = f"{out_path}backtest_lin_preds_{spread}.parquet"
        df_preds.to_parquet(pred_path)
        print(f"Saved predictions to {pred_path}")

        # 2) Build trading signals
        df_signals = build_trading_signal(
            df=df_preds,
            target_spread_col=spread,
            pred_spread_col="PredictedSpread",
            z_threshold=1,
            lookback=60
        )

        # e.g. store columns: [index], [spread], [PredictedSpread], Position, SpreadDiff
        signals_path = f"{out_path}backtest_lin_signals_{spread}.parquet"
        df_signals.to_parquet(signals_path)
        print(f"Saved signals to {signals_path}")

        # 3) Simulate PnL
        df_pnl = simulate_pnl(df_signals, target_spread_col=spread)
        # e.g. store columns: [index], [DailyPnL], [CumulativePnL]
        pnl_path = f"{out_path}backtest_lin_pnl_{spread}.parquet"
        df_pnl.to_parquet(pnl_path)
        print(f"Saved PnL timeseries to {pnl_path}")

        # OPTIONAL: Merge them if you prefer a single DF containing predictions, signals, PnL, etc.
        # NOTE: All DFs share the same index (assuming no big changes in generate_predictions, etc.)
        #       so we can do something like:

        df_merged = df_preds.join(df_signals[["Position", "SpreadResid","SpreadVol","Zscore"]]).join(df_pnl[["DailyPnL", "CumulativePnL"]])
        merged_path = f"{out_path}backtest_lin_merged_{spread}.parquet"
        df_merged.to_parquet(merged_path)
        print(f"Saved merged DataFrame to {merged_path}")

        # 4) Evaluate performance
        perf = evaluate_performance(df_pnl, pnl_col="DailyPnL")
        perf["Spread"] = spread

        print("Performance summary:")
        for k, v in perf.items():
            print(f"  {k}: {v}")

        results.append(perf)

    # Summaries of all spreads
    results_df = pd.DataFrame(results)
    results_out = f"{out_path}backtest_results.parquet"
    results_df.to_parquet(results_out, index=False)
    print(f"\nBacktest results written to {results_out}")

    return results_df


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------
from prep_data_2 import load_data, feature_engineer, cluster_regimes
import utils

def main():
    # 1) Load your 2-year historical data
    df = load_data()
    df_data = feature_engineer(df)
    target_spreads = [f"{i}-{i+1}" for i in range(1,8)]

    cluster_features = [
        'VIX',
        'SP500_realized_vol_21d',
        'SP500_1d', 'SP500_5d',
        'SP500_drawdown'
    ]
    clust = cluster_regimes(df_data, n_clusters=4, cluster_features=cluster_features)
    df_data["ClusterRegime"] = clust["ClusterRegime"]

    # 2) Map each spread to the path of its pickled linear model
    model_paths = {
        spread: f"data/models/linear_model_{spread}.pkl"
        for spread in target_spreads
    }

    # 3) Map each spread to the list of features that model used
    best_feats = pd.read_parquet('data/optimal_models.parquet')
    maximizing_by = 'DirectionalAcc'
    feature_cols_dict = {
        spread: list(best_feats[best_feats['Spread'] == spread].loc[
            best_feats[best_feats['Spread'] == spread][maximizing_by].idxmax()
        ]['SubsetBlocks'])
        for spread in target_spreads
    }

    # 4) The actual columns in df_data that hold each spread's time series
    target_spread_cols = target_spreads

    # 5) Run the backtest
    utils.make_dir('data/backtest')
    results_df = run_backtest(
        model_paths=model_paths,
        df_data=df_data,
        feature_cols_dict=feature_cols_dict,
        target_spread_cols=target_spread_cols,
        out_path="data/backtest/"
    )

    print("\n== Final Summary ==")
    utils.pdf(results_df)


if __name__ == "__main__":
    main()
