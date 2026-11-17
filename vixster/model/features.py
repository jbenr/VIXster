"""Model inputs: adjacent spreads + VIX/SPX + roll timing, then derived features.

Mirrors the old app's live path (prep_data_2.load_data(live=True) followed by
feature_engineer), which keeps only these columns before engineering features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vixster import config
from vixster.io import read_parquet

MODEL_COLS = config.SPREADS + ["VIX", "SP500", "DTR", "DFR"]


def assemble(spreads: pd.DataFrame, curve: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Join spreads, VIX/SPX and DTR/DFR on the dates all three have."""
    spreads, curve, fred = (x.set_axis(pd.to_datetime(x.index), axis=0) for x in (spreads, curve, fred))
    dates = spreads.index.intersection(curve.index).intersection(fred.index).sort_values()
    df = pd.concat([spreads.loc[dates, config.SPREADS],
                    fred.loc[dates, ["VIX", "SP500"]],
                    curve.loc[dates, ["DTR", "DFR"]]], axis=1)
    # FRED leaves SPX/VIX blank on a few exchange holidays when CFE still traded.
    # The old code turned those into 0 (fillna(0)); carrying the last value is right.
    df[["VIX", "SP500"]] = df[["VIX", "SP500"]].ffill()
    df.index.name = "Date"
    return df


def load() -> pd.DataFrame:
    return assemble(read_parquet(config.VX_SPREADS), read_parquet(config.VX_CURVE), read_parquet(config.FRED))


def with_live_row(df: pd.DataFrame, live: dict[str, float], asof: pd.Timestamp) -> pd.DataFrame:
    """Append today's row built from live quotes (spreads + VIX + SP500).

    DTR/DFR roll forward by the calendar days since the last row. Anything the
    live feed didn't supply is carried from the last close, never zero-filled.
    """
    asof = pd.Timestamp(asof).normalize()
    if asof <= df.index[-1]:
        df = df.loc[df.index < asof]
    last = df.iloc[-1]
    days = (asof - df.index[-1]).days
    row = last.copy()
    for k, v in live.items():
        if k in row.index and v is not None and np.isfinite(v):
            row[k] = v
    row["DTR"] = last["DTR"] - days
    row["DFR"] = last["DFR"] + days
    out = pd.concat([df, row.to_frame(asof).T])
    out.index.name = "Date"
    return out.astype(float)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """SPX returns, drawdown, realised vols and month dummies. All column-wise."""
    d = df.copy()
    d.index = pd.to_datetime(d.index)
    spx = d["SP500"]
    d["SP500_5d"] = spx.pct_change(5)
    d["SP500_1d"] = spx.pct_change(1)
    d["SP500_peak"] = spx.cummax()
    d["SP500_drawdown"] = (d["SP500_peak"] - spx) / spx

    months = pd.get_dummies(d.index.month, prefix="Month", drop_first=True).set_axis(d.index)
    d = pd.concat([d, months.astype(float)], axis=1)

    logret = np.log(spx / spx.shift(1))
    for w in (7, 14, 21, 50):
        d[f"SP500_realized_vol_{w}d"] = logret.rolling(w).std() * np.sqrt(252)
    return d.dropna()
