"""PCA on the VX futures curve (constant-rank F1..F8 settles).

Same method as Views' research/curve_pca.py: fit on a trailing window, rebuild
the curve from the first k factors, and read each contract's residual (actual
minus PCA fair value) and its z-score against that contract's own residual
history in the window. Spread residuals follow from the contract residuals,
since spread i-(i+1) = F(i+1) - F(i).

Factor signs are fixed so the charts read the same every day: PC1 (level) has
positive loadings, PC2 (slope) is higher at the back than the front, PC3
(curvature) is positive in the belly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vixster import config
from vixster.io import read_parquet

TENORS = [f"F{i}" for i in range(1, config.N_RANKS + 1)]
WINDOWS = {"1Y": 252, "2Y": 504, "5Y": 1260, "ALL": None}
NAMES = ["Level", "Slope", "Curvature", "PC4", "PC5", "PC6", "PC7", "PC8"]


def load_curve() -> pd.DataFrame:
    df = read_parquet(config.VX_CURVE)
    df.index = pd.to_datetime(df.index)
    return df[TENORS].dropna().sort_index()


def _orient(vecs: np.ndarray) -> np.ndarray:
    n = vecs.shape[0]
    mid = slice(n // 4, n - n // 4)
    checks = [lambda v: v.sum(), lambda v: v[-1] - v[0], lambda v: v[mid].mean() - (v[0] + v[-1]) / 2]
    for k, check in enumerate(checks[: vecs.shape[1]]):
        if check(vecs[:, k]) < 0:
            vecs[:, k] *= -1
    return vecs


def fit(curve: pd.DataFrame, window: str = "2Y", n_components: int = 3, changes: bool = False) -> dict:
    """Loadings, variance shares and factor scores; fair curve and residuals on levels."""
    lookback = WINDOWS[window]
    sample = curve if lookback is None else curve.tail(lookback)
    x = sample.diff().dropna() if changes else sample
    mat = x.to_numpy(float)
    mean = mat.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(mat - mean, rowvar=False))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], _orient(vecs[:, order])
    explained = vals / vals.sum()

    # factor scores over the window, on levels
    lv = sample.to_numpy(float)
    lv_mean = lv.mean(axis=0)
    k = n_components
    scores = pd.DataFrame((lv - lv_mean) @ vecs[:, :k], index=sample.index, columns=NAMES[:k])

    # fair value: each day projected onto the first k factors
    fair = pd.DataFrame(lv_mean + scores.to_numpy() @ vecs[:, :k].T, index=sample.index, columns=TENORS)
    resid = sample - fair
    std = resid.std()
    today = pd.DataFrame({"Actual": sample.iloc[-1], "PCA fair": fair.iloc[-1], "Resid": resid.iloc[-1],
                          "Z": resid.iloc[-1] / std})

    sp_resid = pd.DataFrame({s: resid[f"F{i + 1}"] - resid[f"F{i}"]
                             for i, s in enumerate(config.SPREADS, start=1)})
    spreads = pd.DataFrame({"Actual": [sample.iloc[-1][f"F{i + 1}"] - sample.iloc[-1][f"F{i}"]
                                       for i in range(1, config.N_RANKS)],
                            "Resid": sp_resid.iloc[-1].to_numpy(),
                            "Z": (sp_resid.iloc[-1] / sp_resid.std()).to_numpy()}, index=config.SPREADS)
    spreads["PCA fair"] = spreads["Actual"] - spreads["Resid"]

    return {
        "asof": sample.index[-1], "start": sample.index[0], "n_obs": len(x),
        "loadings": pd.DataFrame(vecs[:, :k], index=TENORS, columns=NAMES[:k]),
        "explained": pd.Series(explained, index=[f"PC{i + 1}" for i in range(len(explained))]),
        "scores": scores, "today": today, "spreads": spreads[["Actual", "PCA fair", "Resid", "Z"]],
        "resid": resid,
    }
