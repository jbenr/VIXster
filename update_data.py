import vix_futures_exp_dates, vixy, fredder
import utils
import pandas as pd
import time, requests, os
from datetime import datetime
import xml.etree.ElementTree as ET


def run():
    vixy.pull_vix(vix_futures_exp_dates.run_over_time_frame(2013))
    df = vixy.process_vix()
    df.to_parquet("data/vix.parquet")
    utils.pdf(df.tail(3))

    loader = vixy.create_calendar_spread_df(df)
    loader.to_parquet('data/vix_load.parquet')
    utils.pdf(loader.tail(3))

    df = vixy.pivot_vix(df)
    df.to_parquet('data/vix_pivot.parquet')
    utils.pdf(df.tail(3))

    df = vixy.vix_spreads(df)
    df.to_parquet('data/vix_spreads.parquet')
    utils.pdf(df[['1-2','2-3','3-4','4-5','5-6','6-7','7-8']].tail(3))

    print("Done pulling VIX data.")

    print("Pulling stonk data...")
    fredder.run()
    print("Done pulling stonk data.")

    print('Scraped.')


def live():
    vix = pd.read_parquet(utils.get_abs_path('data/vix_pivot.parquet'))
    # utils.pdf(vix.tail(10))

    rt_vix = pd.read_parquet(utils.get_abs_path('data/bapi/rt_vix.parquet'))
    rt_vix = vixy.rt_vix_cleaner(rt_vix)
    rt_vix.to_parquet(utils.get_abs_path('data/vix_rt.parquet'))
    rt_vix = vixy.pivot_vix(rt_vix)
    rt_vix.to_parquet(utils.get_abs_path('data/vix_pivot_rt.parquet'))
    # utils.pdf(rt_vix)

    vix.loc[rt_vix.index[0]] = rt_vix.loc[rt_vix.index[0]]
    vix = vixy.vix_spreads(vix)

    vix.index = pd.to_datetime(vix.index)
    vix.to_parquet(utils.get_abs_path('data/vix_spreads_rt.parquet'))

    # utils.pdf(vix.tail(10))

def pull_performance(
    out_parquet: str = "data/performance.parquet"
) -> tuple[pd.DataFrame, dict]:
    """
    1) Downloads the latest Summ.xml from IBKR Flex WebService.
    2) Parses each <FlexStatement> into daily NAV rows.
    3) Computes Cumulative NAV for each date.
    4) Defines an `investors` dictionary up top; each key (except 'Benjin')
       must have: invest_date (YYYY-MM-DD), initial_investment (float), strike (float),
       plus a logo_url and logo_height for display.
    5) Calculates each non-Benjin investor’s ownership_pct = initial_investment / strike.
    6) Computes total_inflows, then:
         • Benjin’s initial_investment = total_inflows − sum_of_all_other_initials
         • Benjin’s ownership_pct = 1 − sum_of_all_other_ownerships
         • Benjin’s logo_url & logo_height can be supplied here as well.
    7) Builds a column "<Name> NAV Share" = ownership_pct * End NAV for each investor
    8) Computes current_value, pnl, return_pct for each investor
    9) Saves daily NAV DataFrame (with all "<Name> NAV Share" columns) to out_parquet
    10) Returns (df, investor_results)

    To add a new investor, simply insert them (with invest_date, initial_investment,
    strike, logo_url, logo_height) into the `investors` dict below. Everything else adapts automatically.

    Raises RuntimeError if the Flex WebService fails or if no valid NAV rows remain.
    """

    # ─── 0) DEFINE YOUR INVESTORS HERE (except Benjin) ────────────────────────────
    # Format: "Name": {
    #     "invest_date": "YYYY-MM-DD",
    #     "initial_investment": float,
    #     "strike": float,
    #     "logo_url": str,        # URL to the investor’s logo
    #     "logo_height": int      # desired height when showing the logo
    # }
    investors = {
        "Chuck": {
            "invest_date":        "2024-03-06",
            "initial_investment": 7000.0,
            "strike":             20699.76,
            "logo_url":           "https://upload.wikimedia.org/wikipedia/commons/archive/5/5c/20240330094136%21Chicago_Bears_logo.svg",
            "logo_height":        40
        },
        "Luger": {
            "invest_date":        "2024-05-08",
            "initial_investment": 1000.0,
            "strike":             23714.12,
            "logo_url":           "https://images.squarespace-cdn.com/content/v1/57c489118419c295dde4c84a/1495392731348-WRZNA4R1PB910OA70Q3D/peter-luger-logo.jpg",
            "logo_height":        40
        },
        # To add another investor (e.g. “Alice”), copy/paste below:
        # "Alice": {
        #     "invest_date":        "2024-07-01",
        #     "initial_investment": 5000.0,
        #     "strike":             25000.00,
        #     "logo_url":           "https://example.com/alice_logo.png",
        #     "logo_height":        40
        # },
    }
    # ──────────────────────────────────────────────────────────────────────────────

    # ─── A) Download Summ.xml from IBKR FlexWebService ─────────────────────────────

    token = "154979803551046183567560"
    queryId = "1155973"
    base_url = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

    # 1) Request a ReferenceCode
    send_url = f"{base_url}/SendRequest?t={token}&q={queryId}&v=3"
    resp = requests.get(send_url)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ref_elem = root.find("ReferenceCode")
    if ref_elem is None or ref_elem.text is None:
        raise RuntimeError("Could not obtain ReferenceCode from IBKR response.")
    ref_code = ref_elem.text

    # 2) Poll until Summ.xml is ready, then write it to disk
    get_url = f"{base_url}/GetStatement?t={token}&q={ref_code}&v=3"
    summ_path = "sheet/Summ.xml"

    for attempt in range(10):
        resp = requests.get(get_url)
        resp.raise_for_status()
        if resp.content.strip():
            os.makedirs(os.path.dirname(summ_path), exist_ok=True)
            with open(summ_path, "wb") as f:
                f.write(resp.content)
            break
        time.sleep(10)
    else:
        raise RuntimeError("Flex Statement never became ready after 10 attempts.")

    # ─── B) Parse Summ.xml into daily NAV rows ────────────────────────────────────

    tree = ET.parse(summ_path)
    root = tree.getroot()

    rows = []
    for stmt in root.iter("FlexStatement"):
        date_str = stmt.get("fromDate")  # e.g. "2025-06-03"
        change_nav = stmt.find("ChangeInNAV")
        if change_nav is None:
            continue

        start_nav = float(change_nav.get("startingValue", 0))
        net_inflow = float(change_nav.get("depositsWithdrawals", 0))
        end_nav = float(change_nav.get("endingValue", 0))
        rows.append([date_str, start_nav, net_inflow, end_nav])

    if not rows:
        raise RuntimeError("No <FlexStatement> rows found in Summ.xml.")

    df = pd.DataFrame(rows, columns=["Date", "Start NAV", "Net Inflow", "End NAV"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # ─── C) Filter tiny NAVs ─────────────────────────────

    df = df[df["Start NAV"] >= 100]
    if df.empty:
        raise RuntimeError("No valid NAV rows remain after dropping small NAVs.")

    # ─── D) Save daily NAV DataFrame to disk (so Streamlit can read it) ────────────

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    # ─── E) Compute each non-Benjin investor’s ownership_pct ──────────────────────

    sum_of_initials = 0.0
    sum_of_ownerships = 0.0
    ownership_map = {}  # name -> ownership_pct

    for name, info in investors.items():
        initial_cap = float(info["initial_investment"])
        strike = float(info["strike"])
        pct = initial_cap / strike
        ownership_map[name] = pct

        sum_of_initials += initial_cap
        sum_of_ownerships += pct

    # ─── F) Now define Benjin as the “leftover” investor ─────────────────────────

    # Benjin’s initial_investment = total_inflows − sum_of_all_other_initials
    benjin_initial = df.iloc[0]['Start NAV']
    if benjin_initial < 0:
        raise RuntimeError(f"Leftover capital (for Benjin) is negative: {benjin_initial}")

    # Benjin’s ownership_pct = 1 − sum_of_all_other_ownerships
    benjin_pct = 1.0 - sum_of_ownerships
    if benjin_pct < 0:
        raise RuntimeError(f"Benjin’s ownership_pct is negative ({benjin_pct}). Check your strikes & investments.")

    # Add Benjin to the investors dictionary:
    strategy_start_date = pd.to_datetime(datetime(2024, 8, 27))
    investors["Benjin"] = {
        "invest_date":        strategy_start_date.strftime("%Y-%m-%d"),
        "initial_investment": benjin_initial,
        "logo_url":           "https://cdn.freebiesupply.com/images/large/2x/washington-redskins-logo-transparent.png",
        "logo_height":        50
    }
    ownership_map["Benjin"] = benjin_pct

    # ─── G) Build each investor’s "<Name> NAV Share" column & compute final stats ─

    investor_results = {}

    for name, info in investors.items():
        invest_date = pd.to_datetime(info["invest_date"])
        initial_cap = float(info["initial_investment"])
        pct = ownership_map[name]

        # 1) Create "<Name> NAV Share" column = pct * End NAV
        df[f"{name} NAV Share"] = pct * df["End NAV"]

        # 2) Current value = last row of "<Name> NAV Share"
        current_val = df[f"{name} NAV Share"].iloc[-1]
        pnl = current_val - initial_cap
        return_pct = (pnl / initial_cap) * 100 if initial_cap != 0 else 0.0

        investor_results[name] = {
            "invest_date":        invest_date.date(),
            "initial_investment": initial_cap,
            "ownership_pct":      pct,
            "current_value":      current_val,
            "pnl":                pnl,
            "return_pct":         return_pct,
            "logo_url":           info.get("logo_url", ""),
            "logo_height":        info.get("logo_height", 40)
        }

    return df, investor_results


def main():
    run()


if __name__ == '__main__':
    # main()
    perf = pull_performance()
    utils.pdf(perf[0])
    for key in perf[1]:
        print(key,perf[1][key])
