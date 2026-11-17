"""All-subsets feature search for each spread's linear model, batched.

Same procedure as the old single_regresh.all_subsets_feature_selection: every
non-empty subset of the candidate blocks, 5-fold expanding-window CV, scored
by MAE / R² / IC / directional accuracy averaged across folds.

The old version fit 32,767 subsets x 5 folds per spread one sklearn model at
a time. Here each fold's X'X and X'y are computed once; a subset's OLS is then
just a small solve on a slice of those, and all subsets of the same size are
solved together in one batched np.linalg.solve.

`python -m vixster.model.search` writes data/selection.parquet and a candidate
models file (data/models_candidate.json). It does NOT replace the live
models.json unless you pass --apply.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd

from vixster import config
from vixster.model import features, linear

BASE_BLOCKS = ["VIX", "DFR", "SP500_realized_vol_7d", "SP500_realized_vol_21d", "SP500_realized_vol_50d",
               "SP500_1d", "SP500_5d", "SP500_drawdown"] + config.SPREADS
CHUNK = 1024


def _folds(n: int, k: int = 5):
    test = n // (k + 1)
    for i in range(k):
        end = n - (k - 1 - i) * test
        yield end - test, end   # train = [0, start), test = [start, end)


def search_spread(data: pd.DataFrame, spread: str, blocks: list[str] | None = None, n_splits: int = 5) -> pd.DataFrame:
    blocks = [b for b in (blocks or BASE_BLOCKS + [linear.SEASONALITY]) if b != spread]
    months = [c for c in data.columns if str(c).startswith("Month_")]
    cols = [b for b in blocks if b != linear.SEASONALITY] + months
    col_of = {b: [cols.index(b)] for b in blocks if b != linear.SEASONALITY}
    col_of[linear.SEASONALITY] = [cols.index(m) for m in months]

    sub = data[cols + [spread]].dropna()
    X = np.column_stack([np.ones(len(sub)), sub[cols].to_numpy(float)])   # col 0 = intercept
    y = sub[spread].to_numpy(float)
    folds = list(_folds(len(sub), n_splits))
    grams = [(X[:s].T @ X[:s], X[:s].T @ y[:s], X[s:e], y[s:e]) for s, e in folds]

    subsets = [c for r in range(1, len(blocks) + 1) for c in itertools.combinations(range(len(blocks)), r)]
    idx_lists = [[0] + sorted(i + 1 for b in c for i in col_of[blocks[b]]) for c in subsets]

    out = np.zeros((len(subsets), 4))
    by_size: dict[int, list[int]] = {}
    for k, cl in enumerate(idx_lists):
        by_size.setdefault(len(cl), []).append(k)
    for p, ks in by_size.items():
        for c0 in range(0, len(ks), CHUNK):
            kk = np.array(ks[c0:c0 + CHUNK])
            I = np.array([idx_lists[k] for k in kk])                       # (B, p)
            acc = np.zeros((len(kk), 4))
            for G, c, Xte, yte in grams:
                Gs = G[I[:, :, None], I[:, None, :]]                        # (B, p, p)
                cs = c[I]                                                    # (B, p)
                beta = np.linalg.solve(Gs + 1e-10 * np.eye(p), cs[..., None])[..., 0]
                pred = np.einsum("tbp,bp->bt", Xte[:, I], beta)              # (B, n_test)
                err = pred - yte
                mae = np.abs(err).mean(1)
                r2 = 1 - (err ** 2).sum(1) / ((yte - yte.mean()) ** 2).sum()
                pc = pred - pred.mean(1, keepdims=True)
                yc = yte - yte.mean()
                ic = (pc @ yc) / np.sqrt((pc ** 2).sum(1) * (yc ** 2).sum())
                da = (np.sign(pred) == np.sign(yte)).mean(1)
                acc += np.column_stack([mae, r2, ic, da])
            out[kk] = acc / len(grams)
    return pd.DataFrame({"Spread": spread, "SubsetBlocks": [[blocks[b] for b in c] for c in subsets],
                         "MAE": out[:, 0], "R2": out[:, 1], "IC": out[:, 2], "DirectionalAcc": out[:, 3]})


def run(spreads: list[str] = config.SPREADS, apply: bool = False, log=print) -> pd.DataFrame:
    data = features.engineer(features.load())
    frames = []
    for s in spreads:
        t0 = time.perf_counter()
        frames.append(search_spread(data, s))
        log(f"{s}: {len(frames[-1]):,} subsets x 5 folds in {time.perf_counter() - t0:.1f}s")
    sel = pd.concat(frames, ignore_index=True)
    sel.to_parquet(config.DATA / "selection.parquet")
    best = linear.best_blocks(sel)
    models = {s: linear.fit(data, s, b) for s, b in best.items()}
    linear.save(models, config.DATA / "models_candidate.json")
    if apply:
        linear.save(models)
        log("models.json updated")
    else:
        log("wrote data/models_candidate.json (live models unchanged; pass --apply to switch)")
    return sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--spreads", nargs="*", default=config.SPREADS)
    a = ap.parse_args()
    run(a.spreads, a.apply)
