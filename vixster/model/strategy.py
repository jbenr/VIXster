"""Ranked long/short calendar-spread strategy.

Each day: z = (pred - spot) / rolling std of daily spread changes.
  * Enter when the highest z >= z_thresh and the lowest z <= -z_thresh
    (long the highest, short the lowest), unless today is a roll day.
  * Exit on a roll day, or when the long's z drops below exit_z or the
    short's z rises above -exit_z. A new entry can follow an exit the same day.
  * Daily P&L is the next day's change: long leg minus short leg.

Z-scores and vols are computed for all spreads at once as arrays. Position
state is path-dependent, so that part is a loop, but over plain numpy arrays
(a few ms for 13 years) instead of pandas rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vixster import config


def _argmax_first(row: np.ndarray, largest: bool) -> int:
    """Index Python's max()/min() would pick over a dict (first wins ties; a
    NaN in first place sticks), so results match the old implementation."""
    best = 0
    for k in range(1, len(row)):
        if (row[k] > row[best]) if largest else (row[k] < row[best]):
            best = k
    return best


def run(preds: pd.DataFrame, data: pd.DataFrame, spreads: list[str] = config.TRADED,
        vol_lookback: int = 60, z_lookback: int = 30, z_thresh: float = 1.0,
        exit_z_thresh: float = 0.25) -> pd.DataFrame:
    df = pd.concat([data[spreads + ["DTR", "DFR"]], preds[[f"Pred_{s}" for s in spreads]]], axis=1)
    df = df.dropna(subset=[f"Pred_{s}" for s in spreads] + spreads).sort_index()

    S = df[spreads].to_numpy(float)
    P = df[[f"Pred_{s}" for s in spreads]].to_numpy(float)
    dS = pd.DataFrame(S).diff()
    vol = dS.rolling(vol_lookback).std().to_numpy()
    vol_z = dS.rolling(z_lookback).std().to_numpy()
    Z = (P - S) / vol_z

    dtr = df["DTR"].to_numpy(float)
    dfr = df["DFR"].to_numpy(float)
    roll = (dtr == 0) | (dfr == 0)
    can_enter = (dtr != 0) & (dfr != 0)

    n, k = S.shape
    nanrow = np.isnan(Z).any(axis=1)
    hi = np.where(nanrow, -1, np.argmax(np.nan_to_num(Z, nan=-np.inf), axis=1))
    lo = np.where(nanrow, -1, np.argmin(np.nan_to_num(Z, nan=np.inf), axis=1))
    for t in np.flatnonzero(nanrow):
        hi[t], lo[t] = _argmax_first(Z[t], True), _argmax_first(Z[t], False)

    nxt = np.vstack([S[1:], S[-1:]])          # last day: no next day, so zero P&L
    chg = nxt - S

    long_i = np.full(n, -1)
    short_i = np.full(n, -1)
    trade_id = np.zeros(n, dtype=int)
    cur_l = cur_s = -1
    tid = 0
    for t in range(n):
        z = Z[t]
        if cur_l >= 0 and (roll[t] or z[cur_l] < exit_z_thresh or z[cur_s] > -exit_z_thresh):
            cur_l = cur_s = -1
        if cur_l < 0 and z[hi[t]] >= z_thresh and z[lo[t]] <= -z_thresh and can_enter[t]:
            cur_l, cur_s = hi[t], lo[t]
            tid += 1
        if cur_l >= 0:
            long_i[t], short_i[t], trade_id[t] = cur_l, cur_s, tid

    active = long_i >= 0
    rows = np.arange(n)
    li, si = np.where(active, long_i, 0), np.where(active, short_i, 0)
    pnl_long = np.where(active, chg[rows, li], np.nan)
    pnl_short = np.where(active, -chg[rows, si], np.nan)
    daily = np.where(active, pnl_long + pnl_short, 0.0)

    names = np.array(spreads, dtype=object)
    out = pd.DataFrame({
        "Trade": active,
        "TradeID": trade_id,
        "LongSpread": np.where(active, names[li], None),
        "ShortSpread": np.where(active, names[si], None),
        "PnL_Long": pnl_long,
        "PnL_Short": pnl_short,
        "DailyPnL": daily,
        "Vol_Long": np.where(active, vol[rows, li], np.nan),
        "Vol_Short": np.where(active, vol[rows, si], np.nan),
    }, index=df.index)
    for j, s in enumerate(spreads):
        out[f"Z_{s}"] = Z[:, j]
        out[f"Pred_{s}"] = P[:, j]
        out[f"Spread_{s}"] = S[:, j]
        out[f"Vol_{s}"] = vol[:, j]
        out[f"VolZ_{s}"] = vol_z[:, j]
    out["CumulativePnL"] = out["DailyPnL"].cumsum()
    out["TradeCumulativePnL"] = np.where(active, out.groupby("TradeID")["DailyPnL"].cumsum(), np.nan)
    out.index.name = "Date"
    return out


def trades(result: pd.DataFrame) -> pd.DataFrame:
    """One row per trade: dates, legs, length and P&L."""
    t = result[result["Trade"]]
    if t.empty:
        return pd.DataFrame(columns=["TradeID", "Entry", "Exit", "Long", "Short", "Days", "PnL"])
    g = t.reset_index().groupby("TradeID")
    out = pd.DataFrame({
        "Entry": g["Date"].first(), "Exit": g["Date"].last(),
        "Long": g["LongSpread"].first(), "Short": g["ShortSpread"].first(),
        "Days": g.size(), "PnL": g["DailyPnL"].sum(),
    }).reset_index()
    return out


def metrics(result: pd.DataFrame) -> dict:
    """Same definitions as the old evaluate_performance (P&L in spread points)."""
    r = result["DailyPnL"]
    cum = result["CumulativePnL"]
    mu, sd = r.mean(), r.std()
    down = r[r < 0].std()
    dd = cum - cum.cummax()
    tp = trades(result)["PnL"]
    n = len(tp)
    wins = tp[tp > 0]
    return {
        "TotalPnL": float(cum.iloc[-1]),
        "AnnualReturn": float(mu * 252),
        "Volatility": float(sd * np.sqrt(252)),
        "Sharpe": float(mu / sd * np.sqrt(252)) if sd else np.nan,
        "Sortino": float(mu * 252 / (down * np.sqrt(252))) if down else np.nan,
        "MaxDrawdown": float(dd.min()),
        "Calmar": float(mu * 252 / abs(dd.min())) if dd.min() else np.nan,
        "Trades": n,
        "WinRate": float(len(wins) / n) if n else np.nan,
        "AvgWin": float(wins.mean()) if len(wins) else np.nan,
        "AvgLoss": float(tp[tp <= 0].mean()) if n > len(wins) else np.nan,
        "AvgDays": float(trades(result)["Days"].mean()) if n else np.nan,
        "TimeInMarket": float(result["Trade"].mean()),
    }


def signal_table(result: pd.DataFrame, spreads: list[str] = config.TRADED,
                 base: str = config.SIZING_BASE) -> pd.DataFrame:
    """Today's signal, one column per spread: spot, fair value, residual, z,
    side, vol and contracts scaled so `base` = 1."""
    last = result.iloc[-1]
    is_trade = bool(last["Trade"])
    vol = {s: last[f"VolZ_{s}"] for s in spreads}
    rows = {
        "Actual": {s: last[f"Spread_{s}"] for s in spreads},
        "Pred": {s: last[f"Pred_{s}"] for s in spreads},
        "Resid": {s: last[f"Pred_{s}"] - last[f"Spread_{s}"] for s in spreads},
        "Z": {s: last[f"Z_{s}"] for s in spreads},
        "Signal": {s: ("Long" if is_trade and s == last["LongSpread"] else
                       "Short" if is_trade and s == last["ShortSpread"] else "") for s in spreads},
        "Vol (30d)": vol,
        "Contracts": {s: (vol[base] / vol[s]) if vol[s] and np.isfinite(vol[s]) else np.nan for s in spreads},
    }
    return pd.DataFrame(rows).T
