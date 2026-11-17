"""Live VX quotes from IB Gateway -> data/live_quotes.parquet (once a second).

Streams the 7 adjacent calendar spreads as exchange combos (BAG: sell the
nearer leg, buy the farther, so the price is F_far - F_near like the model),
the 8 outright futures, and the VIX / SPX indexes when the account has them.

Runs forever and reconnects on its own: if Gateway restarts (nightly) or is
not up yet, it waits and tries again, so it never needs a manual restart.
Run with `python -m vixster.ibkr.stream` or `vixster stream`.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from datetime import date, datetime

import pandas as pd
from ib_async import IB, Bag, ComboLeg, Contract, Future, Index

from vixster import config
from vixster.io import write_parquet

log = logging.getLogger("vixster.stream")
PIDFILE = config.DATA / "stream.pid"
COLS = ["Kind", "Key", "Contract", "Expiry", "Bid Size", "Bid", "Ask", "Ask Size", "Last", "Close", "Updated"]


def _num(x) -> float:
    return float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) and x != -1 else float("nan")


async def front_futures(ib: IB, n: int = config.N_RANKS) -> list[Contract]:
    details = await ib.reqContractDetailsAsync(Future("VIX", exchange="CFE", currency="USD", tradingClass="VX"))
    today = date.today().strftime("%Y%m%d")
    contracts = sorted((d.contract for d in details if d.contract.lastTradeDateOrContractMonth >= today),
                       key=lambda c: c.lastTradeDateOrContractMonth)
    return contracts[:n]


def spread_bag(near: Contract, far: Contract) -> Bag:
    return Bag(symbol="VIX", exchange="CFE", currency="USD", comboLegs=[
        ComboLeg(conId=near.conId, ratio=1, action="SELL", exchange="CFE"),
        ComboLeg(conId=far.conId, ratio=1, action="BUY", exchange="CFE"),
    ])


async def run_once(ib: IB) -> None:
    """One connected session: subscribe and write snapshots until disconnected."""
    ib.reqMarketDataType(int(os.getenv("IB_MARKET_DATA_TYPE", "1")))
    futs = await front_futures(ib)
    if len(futs) < 2:
        raise RuntimeError(f"Only {len(futs)} VX contracts found")
    subs: list[tuple[str, str, str, str, object]] = []
    for i, c in enumerate(futs):
        exp = c.lastTradeDateOrContractMonth
        subs.append(("future", f"F{i + 1}", c.localSymbol, exp, ib.reqMktData(c, "", False, False)))
    for i in range(len(futs) - 1):
        a, b = futs[i], futs[i + 1]
        bag = spread_bag(a, b)
        subs.append(("spread", f"{i + 1}-{i + 2}", f"{a.localSymbol}/{b.localSymbol}",
                     f"{a.lastTradeDateOrContractMonth}/{b.lastTradeDateOrContractMonth}",
                     ib.reqMktData(bag, "", False, False)))
    for key, sym in (("VIX", "VIX"), ("SP500", "SPX")):
        try:
            [idx] = await ib.qualifyContractsAsync(Index(sym, "CBOE", "USD"))
            subs.append(("index", key, sym, "", ib.reqMktData(idx, "", False, False)))
        except Exception as e:  # no index entitlement: the app falls back to Yahoo
            log.warning("index %s unavailable: %s", sym, e)
    log.info("streaming %d instruments", len(subs))

    while ib.isConnected():
        now = datetime.now()
        rows = []
        for kind, key, name, exp, t in subs:
            rows.append([kind, key, name, exp, _num(t.bidSize), _num(t.bid), _num(t.ask), _num(t.askSize),
                         _num(t.last), _num(t.close),
                         (t.time.astimezone().replace(tzinfo=None) if t.time else pd.NaT)])
        df = pd.DataFrame(rows, columns=COLS)
        df["Written"] = now
        if df[["Bid", "Ask", "Last"]].notna().any().any():
            write_parquet(df, config.LIVE_QUOTES, index=False)
        await asyncio.sleep(1)


async def main() -> None:
    ib = IB()
    delay = 5
    while True:
        try:
            await ib.connectAsync(config.IB_HOST, config.IB_PORT, clientId=config.IB_CLIENT_ID, timeout=10)
            log.info("connected to %s:%s", config.IB_HOST, config.IB_PORT)
            delay = 5
            await run_once(ib)
            log.warning("disconnected")
        except Exception as e:
            log.warning("not connected (%s); retrying in %ss", e or type(e).__name__, delay)
        finally:
            if ib.isConnected():
                ib.disconnect()
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60)


def cli() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(config.LOG_DIR / "stream.log")])
    logging.getLogger("ib_async").setLevel(logging.WARNING)
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    try:
        asyncio.run(main())
    finally:
        PIDFILE.unlink(missing_ok=True)


if __name__ == "__main__":
    cli()
