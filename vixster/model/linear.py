"""Per-spread linear models: fit, store as JSON, predict.

Models are plain coefficient tables keyed by feature name (data/models.json),
so there's no pickle/sklearn-version coupling, and predicting every spread
is one matrix multiply per spread.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from vixster import config

SEASONALITY = "SEASONALITY_BLOCK"


def expand(blocks: list[str], columns) -> list[str]:
    months = [c for c in columns if str(c).startswith("Month_")]
    out = []
    for b in blocks:
        out.extend(months if b == SEASONALITY else [b])
    return out


def fit(data: pd.DataFrame, spread: str, blocks: list[str], n_splits: int = 5) -> dict:
    """OLS on the training window of the last TimeSeriesSplit fold, which is
    what the old pipeline kept as its 'final' model."""
    feats = expand(blocks, data.columns)
    sub = data[feats + [spread]].dropna()
    n = len(sub)
    n_train = n - n // (n_splits + 1)
    X = sub[feats].to_numpy(float)[:n_train]
    y = sub[spread].to_numpy(float)[:n_train]
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"features": feats, "intercept": float(beta[0]), "coef": beta[1:].tolist(),
            "blocks": list(blocks), "trained_through": str(pd.Timestamp(sub.index[n_train - 1]).date())
            if isinstance(sub.index, pd.DatetimeIndex) else str(sub.index[n_train - 1])}


def predict(models: dict, data: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for spread, m in models.items():
        X = data[m["features"]].to_numpy(float)
        out[f"Pred_{spread}"] = X @ np.asarray(m["coef"]) + m["intercept"]
    return pd.DataFrame(out, index=data.index)


def save(models: dict, path: Path = config.MODELS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models, indent=1))


def load(path: Path = config.MODELS) -> dict:
    if not Path(path).exists():
        import_legacy()
    return json.loads(Path(path).read_text())


def best_blocks(selection: pd.DataFrame, by: str = config.SELECT_BY) -> dict[str, list[str]]:
    """Feature subset per spread: the row of the subset search that maximises `by`."""
    out = {}
    for spread, g in selection.groupby("Spread"):
        out[spread] = list(g.loc[g[by].idxmax(), "SubsetBlocks"])
    return out


def import_legacy(legacy: Path | None = None) -> dict:
    """Convert the old project's sklearn pickles into models.json (one time)."""
    legacy = Path(legacy or config.LEGACY / "data")
    blocks = best_blocks(pd.read_parquet(legacy / "optimal_models.parquet"))
    models = {}
    for spread in config.SPREADS:
        p = legacy / "models" / f"linear_model_{spread}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as f:
            m = pickle.load(f)
        models[spread] = {"features": [str(c) for c in m.feature_names_in_],
                          "intercept": float(m.intercept_), "coef": [float(c) for c in m.coef_],
                          "blocks": blocks.get(spread, []), "trained_through": "legacy"}
    save(models)
    return models
