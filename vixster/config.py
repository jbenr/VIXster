"""Paths, credentials and strategy constants. Everything else imports from here.

Secrets come from VIXster/.env (never committed).

The site keeps its market data in data/site/ so it never overwrites the files
the old Streamlit app (app.py) reads. The NAV files are shared with data/:
both apps use the same NAV code and format, so a refresh in either shows in both.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT                       # the old app lives in the same folder now

load_dotenv(ROOT / ".env")

SHARED = ROOT / "data"              # the old app's data folder (NAV files live here)
DATA = Path(os.getenv("VIXSTER_DATA", SHARED / "site"))
CBOE_DIR = DATA / "cboe"            # one parquet per VX contract, named by expiry
FLEX_DIR = SHARED / "flex"          # every IBKR Flex download, archived
LOG_DIR = ROOT / "logs"

# Built files
VX_LONG = DATA / "vx_long.parquet"      # every contract/day with DTM, DFR, DTR, Calendar_Num
VX_CURVE = DATA / "vx_curve.parquet"    # constant-rank curve: 1..8 settle + DTR/DFR
VX_SPREADS = DATA / "vx_spreads.parquet"
FRED = DATA / "fred.parquet"
LIVE_QUOTES = DATA / "live_quotes.parquet"   # written by the IBKR streamer
MODELS = DATA / "models.json"
NAV = SHARED / "performance.parquet"
NAV_REPORT = SHARED / "nav_report.csv"
SEED_XML = SHARED / "starting_dataer.xml"

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FLEX_TOKEN = os.getenv("IB_FLEX_TOKEN", "")
FLEX_QUERY_ID = os.getenv("IB_FLEX_QUERY_ID", "1155973")

# IBKR Gateway (driven by IBC). 4001 = Gateway live API, 4002 = Gateway paper.
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "4001"))
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "104"))
IB_TRADING_MODE = os.getenv("IB_TRADING_MODE", "live")
IBC_PATH = Path(os.getenv("IBC_PATH", Path.home() / "ibc"))
TWS_PATH = Path(os.getenv("TWS_PATH", Path.home() / "Jts"))


def _newest_gateway() -> str:
    """Newest IB Gateway installed under ~/Jts/ibgateway (e.g. "1050"). IBC 3.24 needs a
    Gateway that bundles Java 17+, which 10.30 doesn't, so we prefer the newest."""
    try:
        vers = [p.name for p in (TWS_PATH / "ibgateway").iterdir() if p.name.isdigit() and (p / "jars").is_dir()]
        return max(vers, key=int)
    except (OSError, ValueError):
        return "1030"


TWS_MAJOR_VRSN = os.getenv("TWS_MAJOR_VRSN") or _newest_gateway()
IB_DISPLAY = os.getenv("IB_DISPLAY", ":99")

# ── strategy ──────────────────────────────────────────────────────────────────
N_RANKS = 8
SPREADS = [f"{i}-{i + 1}" for i in range(1, N_RANKS)]   # adjacent calendar spreads
TRADED = SPREADS[1:]                                     # 2-3 .. 7-8 (what the old app traded)
BASE_FEATURES = ["VIX", "SP500", "DTR", "DFR"]
SIZING_BASE = "2-3"                                      # contracts are scaled to this spread's vol

STRATEGY = dict(vol_lookback=60, z_lookback=30, z_thresh=1.0, exit_z_thresh=0.25)
SELECT_BY = "DirectionalAcc"                             # how the feature subset per spread was picked

FRED_SERIES = {
    "2Y10Y": "T10Y2Y",
    "5YBEI": "T5YIE",
    "VIX": "VIXCLS",
    "SP500": "SP500",
    "BTC": "CBBTCUSD",
    "HY": "BAMLH0A0HYM2",
    "IG": "BAMLC0A0CM",
}
