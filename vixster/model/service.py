"""What the app asks the model for. History is cached until the data files change."""
from __future__ import annotations

import threading

import pandas as pd

from vixster import config
from vixster.data import live
from vixster.model import features, linear, strategy

_LOCK = threading.Lock()
_CACHE: dict = {}


def _stamp() -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in
                 (config.VX_SPREADS, config.VX_CURVE, config.FRED, config.MODELS))


def history() -> dict:
    """Engineered features, predictions and the full backtest on settlement data."""
    with _LOCK:
        if _CACHE.get("stamp") == _stamp():
            return _CACHE["hist"]
        raw = features.load()
        data = features.engineer(raw)
        models = linear.load()
        preds = linear.predict(models, data)
        res = strategy.run(preds, data, **config.STRATEGY)
        hist = {"raw": raw, "data": data, "models": models, "preds": preds, "result": res}
        _CACHE.update(stamp=_stamp(), hist=hist)
        return hist


def backtest(**params) -> pd.DataFrame:
    h = history()
    return strategy.run(h["preds"], h["data"], **{**config.STRATEGY, **params})


def live_signal(use_live: bool = True) -> dict:
    """Today's signal. With live quotes it appends an intraday row; without, it's
    the signal as of the last settlement."""
    h = history()
    meta = {"asof": h["raw"].index[-1], "mode": "settlement"}
    data = h["data"]
    if use_live:
        vals, m = live.live_inputs()
        meta.update(m)
        today = pd.Timestamp.today().normalize()
        if vals:
            raw = features.with_live_row(h["raw"], vals, today)
            data = features.engineer(raw)
            meta.update(asof=today, mode="live" if m.get("spreads_source") else "live (indexes only)",
                        live_values=vals)
    preds = linear.predict(h["models"], data)
    res = strategy.run(preds, data, **config.STRATEGY)
    return {"result": res, "table": strategy.signal_table(res), "meta": meta, "data": data}
