import vix_futures_exp_dates, vixy, fredder
import utils
import pandas as pd
import time, requests, os
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import Tuple, Dict


# ───────────────────────────── existing flows ─────────────────────────────

def run(log=print):
    """Re-pull VX futures + FRED and rebuild the spread files.
    `log` gets progress messages; returns a short summary for the UI."""
    t0 = time.perf_counter()
    pull = vixy.pull_vix(vix_futures_exp_dates.run_over_time_frame(2013), log=log)

    log("Building the futures curve and spreads…")
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
    last = pd.Timestamp(df.index.max())
    log(f"VX spreads through {last:%a %b %d, %Y} ({len(df):,} days).")

    fredder.run(log=log)

    seconds = time.perf_counter() - t0
    log(f"Done in {seconds:.1f} s.")
    return {'through': last, 'seconds': seconds, 'failed': pull['failed']}


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


# ───────────────────────────── performance ─────────────────────────────
# Lives in performance.py; re-exported here so existing imports keep working.
from performance import (pull_performance, load_performance_from_disk,  # noqa: F401
                         compute_investor_results, parse_flex_xml_to_df, INVESTORS)


def main():
    run()


if __name__ == '__main__':
    # main()
    perf = pull_performance()
    utils.pdf(perf[0])
    for key in perf[1]:
        print(key, perf[1][key])
