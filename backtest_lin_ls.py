import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import StandardScaler
from prep_data_2 import load_data, feature_engineer
import utils


def load_models(model_paths):
    models = {}
    for spread, path in model_paths.items():
        with open(path, 'rb') as f:
            models[spread] = pickle.load(f)
    return models


def expand_seasonality_block(feature_cols, df):
    if "SEASONALITY_BLOCK" in feature_cols:
        month_cols = [c for c in df.columns if c.startswith("Month_")]
        return [c for c in feature_cols if c != "SEASONALITY_BLOCK"] + month_cols
    return feature_cols


def generate_all_predictions(models, df_data, feature_cols_dict):
    df_preds = df_data.copy()
    for spread, model in models.items():
        feats = expand_seasonality_block(feature_cols_dict[spread], df_data)
        df_preds[f"Pred_{spread}"] = model.predict(df_data[feats].dropna())
    return df_preds


def front_idx(spread: str) -> int:
    # "1-2" -> 1, "3-4" -> 3
    return int(spread.split('-')[0])

def inv_vol(x: float) -> float:
    return 0.0 if (x is None or np.isnan(x) or x <= 0) else 1.0 / x

def contracts_by_front(long_s: str,
                       short_s: str,
                       vols: dict[str, float],
                       *,
                       min_contract: int = 1,
                       cap: int | None = None,
                       all_spreads: list[str] | None = None) -> dict[str, int]:
    """
    Set the *front* leg (smaller first index) to 1 contract.
    Scale the other leg by inverse-vol ratio, rounded to int.
    Non-active spreads get 0 unless you pass all_spreads to include them as zeros.
    """
    # pick the front leg among the active pair
    front = min([long_s, short_s], key=front_idx)
    back  = short_s if front == long_s else long_s

    u_front = inv_vol(vols.get(front, np.nan))
    u_back  = inv_vol(vols.get(back,  np.nan))

    # base = 1 on front; other = round(u_back / u_front)
    c_front = max(min_contract, 1)  # force at least 1
    ratio   = (u_back / u_front) if u_front > 0 else 1.0
    c_back  = int(round(max(min_contract, ratio)))

    # optional cap
    if cap is not None:
        c_front = min(c_front, cap)
        c_back  = min(c_back,  cap)

    # assign signs: long positive, short negative
    res = {front: +c_front, back: -c_back} if front == long_s else {front: -c_front, back: +c_back}

    # include zeros for all other spreads if requested
    if all_spreads:
        for s in all_spreads:
            res.setdefault(s, 0)

    return res


def ranked_strategy_vol_adjusted(df_preds, target_spreads, vol_lookback=150, z_lookback=20, z_thresh=1.0, exit_z_thresh=1.0):
    df = df_preds.copy()
    df = df.dropna(subset=[f"Pred_{s}" for s in target_spreads] + target_spreads)
    df.sort_index(inplace=True)

    for s in target_spreads:
        df[f"Vol_{s}"] = df[s].diff().rolling(window=vol_lookback).std()
        df[f"Vol_z_{s}"] = df[s].diff().rolling(window=z_lookback).std()
        df[f"Z_{s}"] = (df[f"Pred_{s}"] - df[s]) / df[f"Vol_z_{s}"]

    daily_pnls = []
    trade_log = []
    trade_id = 0
    current_position = None
    current_weights = (0.5, 0.5)

    # for i in range(len(df)-1):
    #     row_today = df.iloc[i]
    #     row_next = df.iloc[i + 1]
    for i in range(len(df)):
        row_today = df.iloc[i]
        if i == len(df) - 1:
            row_next = row_today.copy()
        else:
            row_next = df.iloc[i + 1]

        zscores = {s: row_today[f"Z_{s}"] for s in target_spreads}
        preds = {s: row_today[f"Pred_{s}"] for s in target_spreads}
        vols = {s: row_today[f"Vol_{s}"] for s in target_spreads}

        roll_detected = (row_today.get("DTR", 1) == 0) or (row_today.get("DFR", 1) == 0)

        long_spread = max(zscores, key=zscores.get)
        short_spread = min(zscores, key=zscores.get)

        entry_block = row_today.get("DTR", 1) != 0 and row_today.get("DFR", 1) != 0

        if current_position:
            current_long, current_short = current_position
            if roll_detected or zscores[current_long] < exit_z_thresh or zscores[current_short] > -exit_z_thresh:
                current_position = None
                current_weights = (0.5, 0.5)

        if not current_position and zscores[long_spread] >= z_thresh and zscores[short_spread] <= -z_thresh and entry_block:
            current_position = (long_spread, short_spread)
            vol_long = vols[long_spread]
            vol_short = vols[short_spread]
            if np.isnan(vol_long) or np.isnan(vol_short) or (vol_long + vol_short) == 0:
                weight_long = weight_short = 0.5
            else:
                weight_long = vol_short / (vol_long + vol_short)
                weight_short = vol_long / (vol_long + vol_short)
            current_weights = (weight_long, weight_short)
            trade_id += 1

        if current_position:
            long_s, short_s = current_position
            contracts_all = contracts_by_front(
                long_s, short_s, vols,
                min_contract=1,  # front leg becomes 1
                cap=None,  # or an int cap if you want to limit size
                all_spreads=target_spreads
            )
        else:
            contracts_all = {s: 0 for s in target_spreads}

        if not current_position:
            trade_log.append({
                "Date": row_today.name,
                "Trade": False,
                "TradeID": 0,
                "DailyPnL": 0.0,
                "TradeCumulativePnL": np.nan,
                **{f"Z_{s}": zscores[s] for s in target_spreads},
                **{f"Pred_{s}": preds[s] for s in target_spreads},
                **{f"Spread_{s}": row_today[s] for s in target_spreads},
                **{f"Contracts_{s}": contracts_all[s] for s in target_spreads},
            })
            daily_pnls.append(0.0)
            continue

        long_spread, short_spread = current_position
        weight_long, weight_short = current_weights

        pnl_long = (row_next[long_spread] - row_today[long_spread]) * contracts_all[long_spread]
        pnl_short = (row_next[short_spread] - row_today[short_spread]) * contracts_all[short_spread]
        daily_pnls.append(pnl_long + pnl_short)

        trade_log.append({
            "Date": row_today.name,
            "Trade": True,
            "TradeID": trade_id,
            "LongSpread": long_spread,
            "ShortSpread": short_spread,
            "Weight_Long": weight_long,
            "Weight_Short": weight_short,
            "PnL_Long": pnl_long,
            "PnL_Short": pnl_short,
            "DailyPnL": pnl_long + pnl_short,
            "Vol_Long": vols[long_spread],
            "Vol_Short": vols[short_spread],
            **{f"Z_{s}": zscores[s] for s in target_spreads},
            **{f"Pred_{s}": preds[s] for s in target_spreads},
            **{f"Spread_{s}": row_today[s] for s in target_spreads},
            **{f"Contracts_{s}": contracts_all[s] for s in target_spreads},
        })

    df_result = pd.DataFrame(trade_log).set_index("Date")
    df_result["CumulativePnL"] = df_result["DailyPnL"].cumsum()
    df_result["TradeCumulativePnL"] = df_result.groupby("TradeID")["DailyPnL"].cumsum()

    return df_result


def evaluate_performance(df):
    daily_returns = df["DailyPnL"]
    cum_pnl = df["CumulativePnL"]

    total_pnl = cum_pnl.iloc[-1]
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else np.nan
    annual_return = mean_return * 252
    volatility = std_return * np.sqrt(252)

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else np.nan

    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    trade_pnl = df[df["Trade"]].groupby("TradeID")["TradeCumulativePnL"].last()
    trade_pnl = trade_pnl[trade_pnl.index != 0]
    num_trades = len(trade_pnl)
    num_wins = (trade_pnl > 0).sum()
    win_rate = num_wins / num_trades if num_trades > 0 else np.nan

    avg_win = trade_pnl[trade_pnl > 0].mean() if num_wins > 0 else np.nan
    avg_loss = trade_pnl[trade_pnl <= 0].mean() if num_wins < num_trades else np.nan
    max_win = trade_pnl.max() if num_trades > 0 else np.nan
    max_loss = trade_pnl.min() if num_trades > 0 else np.nan

    trade_durations = df[df["Trade"]].groupby("TradeID").size()
    avg_trade_duration = trade_durations.mean() if num_trades > 0 else np.nan

    print("\n📊 Strategy Performance Summary")
    print("-" * 50)
    print(f"Total Return          : {total_pnl:.2f}")
    print(f"Annualized Return     : {annual_return:.2f}")
    print(f"Volatility (Ann.)     : {volatility:.4f}")
    print(f"Sharpe Ratio          : {sharpe:.4f}")
    print(f"Sortino Ratio         : {sortino:.4f}")
    print(f"Max Drawdown          : {max_drawdown:.2f}")
    print(f"Calmar Ratio          : {calmar_ratio:.4f}")
    print(f"Win Rate (Trades)     : {win_rate:.2%}")
    print(f"Number of Trades      : {num_trades}")
    print(f"Avg Trade Duration    : {avg_trade_duration:.2f} days")
    print(f"Avg Win per Trade     : {avg_win:.4f}")
    print(f"Avg Loss per Trade    : {avg_loss:.4f}")
    print(f"Max Win per Trade     : {max_win:.4f}")
    print(f"Max Loss per Trade    : {max_loss:.4f}")
    print("-" * 50)

    return {
        "TotalPnL": total_pnl,
        "AnnualReturn": annual_return,
        "Volatility": volatility,
        "SharpeRatio": sharpe,
        "SortinoRatio": sortino,
        "MaxDrawdown": max_drawdown,
        "CalmarRatio": calmar_ratio,
        "WinRate": win_rate,
        "NumTrades": num_trades,
        "AvgTradeDuration": avg_trade_duration,
        "AvgWin": avg_win,
        "AvgLoss": avg_loss,
        "MaxWin": max_win,
        "MaxLoss": max_loss
    }


def plot_pnl(df):
    df["CumulativePnL"].plot(figsize=(10, 4), title="Cumulative PnL", grid=True)
    plt.xlabel("Date")
    plt.ylabel("PnL")
    plt.tight_layout()
    plt.show()


def main():
    print("=== Loading Data ===")
    df = load_data(live=True)
    df_data = feature_engineer(df)
    target_spreads = [f"{i}-{i+1}" for i in range(1, 8)][2:]

    model_paths = {
        spread: f"data/models/linear_model_{spread}.pkl"
        for spread in target_spreads
    }

    best_feats = pd.read_parquet('data/optimal_models.parquet')
    maximizing_by = 'DirectionalAcc'
    feature_cols_dict = {
        spread: list(best_feats[best_feats['Spread'] == spread].loc[
            best_feats[best_feats['Spread'] == spread][maximizing_by].idxmax()
        ]['SubsetBlocks'])
        for spread in target_spreads
    }

    models = load_models(model_paths)

    df_preds = generate_all_predictions(models, df_data, feature_cols_dict)
    df_result = ranked_strategy_vol_adjusted(
        df_preds, target_spreads, vol_lookback=60, z_lookback=30, z_thresh=1.0, exit_z_thresh=0.25)

    utils.make_dir("data/backtest_ranked")
    df_result.to_parquet("data/backtest_ranked/pnl_timeseries.parquet")
    # utils.pdf(df_result[(df_result.index.year==2020)&(df_result.index.month==3)])
    # utils.pdf(df_result[df_result.index.year==2025])
    utils.pdf(df_result.tail(10))

    performance = evaluate_performance(df_result)
    plot_pnl(df_result)


if __name__ == "__main__":
    main()