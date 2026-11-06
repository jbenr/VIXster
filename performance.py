"""
Fund NAV history, returns and investor ownership.

Data flow:
  IBKR Flex XML  ->  data/flex/*.xml (every download is archived, never overwritten)
                 ->  data/performance.parquet (merged daily NAV history)
                 ->  data/nav_report.csv (human-readable daily report)

Returns are time-weighted: each day's return is measured on the money that was
in the account that day, so deposits/withdrawals never count as gains or losses.
"""
import glob
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

FLEX_DIR = "data/flex"
SEED_XML = "data/starting_dataer.xml"
NAV_PARQUET = "data/performance.parquet"
REPORT_CSV = "data/nav_report.csv"
NAV_COLS = ["Date", "Start NAV", "Net Inflow", "End NAV"]

FLEX_TOKEN = "35829916154420740066444"
FLEX_QUERY_ID = "1155973"
FLEX_BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

# The account owner. Owns everything not bought by an outside investor below,
# and every deposit/withdrawal not listed below is theirs.
OWNER = "Benjin"
OWNER_LOGO = {"logo_url": "https://cdn.freebiesupply.com/images/large/2x/washington-redskins-logo-transparent.png",
              "logo_height": 50}

# Outside investors. On the first deposit of `initial_investment` on/after
# `invest_date`, they get initial_investment / strike of the fund.
INVESTORS = {
    "Chuck": {
        "invest_date":        "2025-03-06",
        "initial_investment": 7000.0,
        "strike":             20699.76,
        "logo_url":           "https://upload.wikimedia.org/wikipedia/commons/archive/5/5c/20240330094136%21Chicago_Bears_logo.svg",
        "logo_height":        40,
    },
    "Luger": {
        "invest_date":        "2025-05-08",
        "initial_investment": 1000.0,
        "strike":             23714.12,
        "logo_url":           "https://images.squarespace-cdn.com/content/v1/57c489118419c295dde4c84a/1495392731348-WRZNA4R1PB910OA70Q3D/peter-luger-logo.jpg",
        "logo_height":        40,
    },
}


# ───────────────────────────── IBKR download ─────────────────────────────

def fetch_flex(from_date: Optional[str] = None, to_date: Optional[str] = None) -> str:
    """
    Download a Flex statement and archive it in data/flex/. Returns the file path.
    Without dates you get the query's default window (last 365 days);
    with dates (YYYYMMDD) you get exactly that range.
    """
    params = {"t": FLEX_TOKEN, "q": FLEX_QUERY_ID, "v": 3}
    if from_date and to_date:
        params.update(fd=from_date, td=to_date)
    resp = requests.get(f"{FLEX_BASE_URL}/SendRequest", params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ref = root.findtext("ReferenceCode")
    if not ref:
        raise RuntimeError(f"IBKR Flex error {root.findtext('ErrorCode', '?')}: "
                           f"{root.findtext('ErrorMessage', 'unknown error')}")

    for _ in range(30):  # a full year can take IBKR ~2 minutes to build
        resp = requests.get(f"{FLEX_BASE_URL}/GetStatement",
                            params={"t": FLEX_TOKEN, "q": ref, "v": 3}, timeout=60)
        resp.raise_for_status()
        if "<FlexStatements" in resp.text:
            os.makedirs(FLEX_DIR, exist_ok=True)
            tag = f"{from_date}_{to_date}" if from_date else "latest"
            path = os.path.join(FLEX_DIR, f"flex_{tag}_{datetime.now():%Y%m%d_%H%M%S}.xml")
            with open(path, "w") as f:
                f.write(resp.text)
            return path
        time.sleep(10)
    raise RuntimeError("Flex statement never became ready.")


def parse_flex_xml_to_df(xml_path: str) -> pd.DataFrame:
    """Parse a Flex XML into daily rows of Date / Start NAV / Net Inflow / End NAV."""
    rows = []
    for stmt in ET.parse(xml_path).getroot().iter("FlexStatement"):
        nav = stmt.find("ChangeInNAV")
        if nav is None:
            continue
        rows.append([
            stmt.get("fromDate"),
            float(nav.get("startingValue", 0) or 0),
            float(nav.get("depositsWithdrawals", 0) or 0),
            float(nav.get("endingValue", 0) or 0),
        ])
    df = pd.DataFrame(rows, columns=NAV_COLS)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Skip the near-empty account days before funding.
    return df.dropna(subset=["Date"])[lambda d: d["Start NAV"] >= 100]


# ───────────────────────────── NAV history ─────────────────────────────

def merge_nav_history() -> pd.DataFrame:
    """Merge the saved parquet, the seed XML and every archived Flex XML."""
    frames = []
    if os.path.exists(NAV_PARQUET) and os.path.getsize(NAV_PARQUET) > 0:
        frames.append(pd.read_parquet(NAV_PARQUET)[NAV_COLS])
    for path in [SEED_XML] + sorted(glob.glob(os.path.join(FLEX_DIR, "*.xml"))):
        if os.path.exists(path):
            frames.append(parse_flex_xml_to_df(path))
    if not frames:
        raise RuntimeError("No NAV data found. Click 'Refresh NAV Data'.")
    df = pd.concat(frames, ignore_index=True).sort_values("Date")
    return df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


def find_gaps(df: pd.DataFrame, tol: float = 0.01) -> list:
    """Return (last_date_before, first_date_after) for every break in the NAV chain."""
    breaks = (df["Start NAV"] - df["End NAV"].shift()).abs() > tol
    return [(df["Date"].iloc[i - 1], df["Date"].iloc[i]) for i in df.index[breaks] if i > 0]


def save_nav_history(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(NAV_PARQUET), exist_ok=True)
    df[NAV_COLS].to_parquet(NAV_PARQUET, index=False)


# ───────────────────────────── returns ─────────────────────────────

def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily return with deposits/withdrawals assumed to land at the start of the day:
        (End - Start - Flow) / (Start + Flow)
    Cumulative Return chains those together (time-weighted return).
    """
    df = df.copy()
    flow = df["Net Inflow"]
    df["Daily Return"] = (df["End NAV"] - df["Start NAV"] - flow) / (df["Start NAV"] + flow)
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1
    return df


def period_returns(df: pd.DataFrame, freq: str) -> pd.Series:
    """Time-weighted return per period ('ME' monthly, 'YE' yearly)."""
    growth = (1 + df.set_index("Date")["Daily Return"]).resample(freq).prod(min_count=1)
    return growth.dropna() - 1


def sharpe(daily: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe from daily returns (0% risk-free rate)."""
    daily = daily.dropna()
    sd = daily.std()
    return float(daily.mean() / sd * periods_per_year ** 0.5) if sd > 0 else float("nan")


def yearly_sharpe(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df["Date"].dt.year)["Daily Return"].apply(sharpe)


def summary(df: pd.DataFrame) -> Dict:
    first, last = df.iloc[0], df.iloc[-1]
    net_deposits = float(df["Net Inflow"].sum())
    this_year = df[df["Date"].dt.year == last["Date"].year]
    return {
        "start_date":    first["Date"].date(),
        "end_date":      last["Date"].date(),
        "nav":           float(last["End NAV"]),
        "money_in":      float(first["Start NAV"]) + net_deposits,
        "pnl":           float(last["End NAV"] - first["Start NAV"]) - net_deposits,
        "total_return":  float(last["Cumulative Return"]),
        "ytd_return":    float((1 + this_year["Daily Return"]).prod() - 1),
        "gaps":          find_gaps(df),
    }


# ───────────────────────────── investors ─────────────────────────────

def compute_investor_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Fund-style unit accounting.
      * The owner starts with all units.
      * An outside investor's buy-in gets them exactly initial_investment / strike
        of the fund (the agreed strike), by issuing new units.
      * Every other deposit/withdrawal is the owner's, at that day's unit price.
    Later deposits/withdrawals therefore dilute or concentrate everyone fairly,
    and investor values always add up to the account NAV.
    """
    df = df.reset_index(drop=True)
    names = [OWNER] + list(INVESTORS)
    units = {n: 0.0 for n in names}
    contributed = {n: 0.0 for n in names}
    units[OWNER] = contributed[OWNER] = float(df.loc[0, "Start NAV"])

    # Which row is each investor's buy-in?
    buy_in_row = {}
    for name, info in INVESTORS.items():
        amt = float(info["initial_investment"])
        hits = df.index[(df["Date"] >= pd.Timestamp(info["invest_date"])) & (df["Net Inflow"] >= amt - 0.01)]
        if len(hits) == 0:
            raise RuntimeError(f"Couldn't find {name}'s ${amt:,.0f} deposit on/after {info['invest_date']}.")
        buy_in_row[hits[0]] = name

    share_cols = {n: [] for n in names}
    for i, row in df.iterrows():
        flow = float(row["Net Inflow"])
        total = sum(units.values())
        if i in buy_in_row:
            name = buy_in_row[i]
            amt = float(INVESTORS[name]["initial_investment"])
            pct = amt / float(INVESTORS[name]["strike"])
            new_units = total * pct / (1 - pct)  # -> they own exactly pct afterwards
            units[name] += new_units
            contributed[name] += amt
            flow -= amt
            total += new_units
        if abs(flow) > 0.005:  # owner's own deposit/withdrawal at market price
            price = float(row["Start NAV"]) / total
            units[OWNER] += flow / price
            contributed[OWNER] += flow
            total += flow / price
        for n in names:
            share_cols[n].append(units[n] / total * float(row["End NAV"]))

    logos = {OWNER: OWNER_LOGO, **INVESTORS}
    results = {}
    for n in names:
        df[f"{n} NAV Share"] = share_cols[n]
        value = share_cols[n][-1]
        pnl = value - contributed[n]
        results[n] = {
            "initial_investment": contributed[n],
            "ownership_pct":      units[n] / sum(units.values()),
            "current_value":      value,
            "pnl":                pnl,
            "return_pct":         pnl / contributed[n] * 100 if contributed[n] else 0.0,
            "logo_url":           logos[n].get("logo_url", ""),
            "logo_height":        logos[n].get("logo_height", 40),
        }
    return df, results


# ───────────────────────────── entry points ─────────────────────────────

def _finish(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = add_daily_returns(df)
    df, investors = compute_investor_results(df)
    return df, investors


def write_report(df: pd.DataFrame) -> None:
    out = df.copy()
    out["Date"] = out["Date"].dt.date
    out.round(4).to_csv(REPORT_CSV, index=False)


def load_performance_from_disk(parquet_path: str = NAV_PARQUET) -> Tuple[pd.DataFrame, Dict]:
    """Saved NAV data only, no IBKR call."""
    if not os.path.exists(parquet_path):
        raise RuntimeError(f"No saved performance data at {parquet_path}. Click 'Refresh NAV Data' first.")
    return _finish(pd.read_parquet(parquet_path)[NAV_COLS])


def fetch_range(start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Download start..end inclusive, in chunks IBKR accepts (max 365 days each)."""
    while start <= end:
        chunk_end = min(end, start + timedelta(days=364))
        fetch_flex(start.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d"))
        start = chunk_end + timedelta(days=1)


def pull_performance(out_parquet: str = NAV_PARQUET, log=print) -> Tuple[pd.DataFrame, Dict]:
    """
    Download only the days since the last saved date (fast), backfill any
    holes, then save the parquet + CSV report. `log` gets progress messages.
    """
    yesterday = pd.Timestamp.today().normalize() - timedelta(days=1)
    try:
        last = merge_nav_history()["Date"].max()
    except RuntimeError:
        last = None

    if last is None:
        log("No saved history: downloading the last year from IBKR (takes ~2 min)…")
        fetch_flex()
    elif last < yesterday:
        start = last - timedelta(days=3)  # small overlap to catch late corrections
        log(f"Downloading {start:%b %d} → {yesterday:%b %d} from IBKR…")
        fetch_range(start, yesterday)
    else:
        log("Already up to date.")

    df = merge_nav_history()
    for before, after in find_gaps(df):
        if (after - before).days < 2:
            continue  # consecutive days: nothing to download
        log(f"Backfilling missing dates {before:%Y-%m-%d} → {after:%Y-%m-%d}…")
        try:
            fetch_range(before + timedelta(days=1), after - timedelta(days=1))
        except Exception as e:
            log(f"Backfill failed: {e}")
    df = merge_nav_history()
    save_nav_history(df)
    df, investors = _finish(df)
    write_report(df)
    return df, investors


if __name__ == "__main__":
    df, inv = load_performance_from_disk()
    s = summary(df)
    print({k: v for k, v in s.items()})
    for k, v in inv.items():
        print(k, {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()})
