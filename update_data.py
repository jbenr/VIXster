import vix_futures_exp_dates, vixy, fredder
import utils
import pandas as pd
import time, requests, os
from datetime import datetime

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

def pull_performance(out_parquet: str = "data/performance.parquet") -> pd.DataFrame:
    strategy_start_date = datetime(2024, 8, 27)
    chuck_initial_investment = 7000
    chuck_strike = 20699.76
    chuck_pct = chuck_initial_investment / chuck_strike
    chuck_invest_date = datetime(2024, 3, 6)

    """
    1) Downloads the latest Summ.xml from IBKR.
    2) Parses each <FlexStatement> into daily NAV rows.
    3) Computes Cumulative NAV, Ben NAV, Chuck NAV, etc.
    4) Saves the full daily‐level DataFrame to `out_parquet`.
    5) Returns that DataFrame.

    Raises an exception if the XML download or parsing fails.
    """
    import xml.etree.ElementTree as ET
    token = "154979803551046183567560"
    queryId = "1155973"
    base_url = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

    # STEP A: 1) Request Flex Statement reference code
    send_url = f"{base_url}/SendRequest?t={token}&q={queryId}&v=3"
    resp = requests.get(send_url)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ref_code = root.find("ReferenceCode")
    if ref_code is None or ref_code.text is None:
        raise RuntimeError("Could not obtain ReferenceCode from IBKR response.")
    ref_code = ref_code.text

    get_url = f"{base_url}/GetStatement?t={token}&q={ref_code}&v=3"
    summ_path = "data/Summ.xml"

    for attempt in range(10):
        resp = requests.get(get_url)
        resp.raise_for_status()
        if resp.content.strip():
            os.makedirs(os.path.dirname(summ_path), exist_ok=True)
            with open(summ_path, "wb") as f:
                f.write(resp.content)
            break
        time.sleep(1)
    else:
        raise RuntimeError("IBKR Flex Statement never became ready after 10 attempts.")

    # STEP B: Parse Summ.xml into a daily NAV DataFrame
    tree = ET.parse(summ_path)
    root = tree.getroot()

    rows = []
    for statement in root.iter("FlexStatement"):
        # Each <FlexStatement> has attributes fromDate, toDate, etc.
        date_str = statement.get("fromDate")  # e.g. "2025-06-03"
        change_nav = statement.find("ChangeInNAV")
        if change_nav is None:
            continue

        start_nav = float(change_nav.get("startingValue", 0))
        net_inflow = float(change_nav.get("depositsWithdrawals", 0))
        end_nav = float(change_nav.get("endingValue", 0))
        rows.append([date_str, start_nav, net_inflow, end_nav])

    if not rows:
        raise RuntimeError("No <FlexStatement> rows found in Summ.xml.")

    df = pd.DataFrame(rows, columns=["Date", "Start NAV", "Net Inflow", "End NAV"])

    # Convert Date → datetime, drop any bad rows, sort ascending
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter out anything before strategy_start_date, and any Start NAV < 100
    df = df[(df["Date"] >= pd.to_datetime(strategy_start_date)) & (df["Start NAV"] >= 100)]
    if df.empty:
        raise RuntimeError("No valid NAV rows remain after filtering by date/NAV.")

    # STEP C: Compute Cumulative and split NAVs
    # “Cumulative NAV” is just End NAV (no adjustments here)
    df["Cumulative NAV"] = df["End NAV"]

    # Benjin vs Chuck splits:
    df["Ben NAV"]   = df["Cumulative NAV"] * (1 - chuck_pct)
    df["Chuck NAV"] = df["Cumulative NAV"] * chuck_pct

    # STEP D: Resample to monthly & yearly returns, if needed (optional)
    # We’ll store these in some separate DataFrames if you want to display later,
    # but the primary DataFrame (df) is daily‐level. We’ll compute returns on the fly
    # in Streamlit using df itself.

    # STEP E: Save df to disk as Parquet for later loading in Streamlit
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return df


def main():
    run()


if __name__ == '__main__':
    main()

