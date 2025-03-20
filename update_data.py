import vix_futures_exp_dates, vixy, fredder
import utils
import pandas as pd


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


def main():
    run()


if __name__ == '__main__':
    main()

