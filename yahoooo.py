import random
import time
import atexit
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import utils

# 1) Spin up the gateway
gateway = ApiGateway(
    "https://query1.finance.yahoo.com",
    regions=EXTRA_REGIONS,
    access_key_id=utils.AWS_ACCESS,
    access_key_secret=utils.AWS_SECRET
)
gateway.start()
atexit.register(lambda: gateway.shutdown())

# 2) Mount it on both query1 AND query2 (yfinance uses both)
rotating_session = requests.Session()
rotating_session.mount("https://query1.finance.yahoo.com", gateway)
rotating_session.mount("https://query2.finance.yahoo.com", gateway)

# 3) User‑Agent pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]


def fetch_yahoo_finance_data(symbols, start_date, end_date):
    """
    Fetch historical daily stock data for given symbols from Yahoo Finance.
    """
    data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    data = filter_adjusted_close(data)
    return data

def filter_adjusted_close(data):
    """
    Filter the DataFrame to only include the adjusted close columns for each ticker.
    """
    adj_close_columns = [(symbol, 'Adj Close') for symbol in data.columns.levels[0]]
    filtered_data = data.loc[:, adj_close_columns]
    # Flatten the multi-index columns
    filtered_data.columns = [symbol for symbol, _ in filtered_data.columns]
    return filtered_data

def get_price(symbol):
    """
    Fetch the latest close for a single ticker, rotating both IP and UA,
    and sleeping a bit to avoid hammering the server.
    """
    try:
        # small random delay
        time.sleep(random.uniform(0.5, 1.5))

        # rotate UA
        rotating_session.headers.update({
            "User-Agent": random.choice(USER_AGENTS)
        })

        # pass our rotating session into yfinance
        t = yf.Ticker(symbol, session=rotating_session)
        df = t.history(period="1d", timeout=10)

        df.index = pd.to_datetime(df.index).date
        return df[["Close"]].tail(1)

    except Exception as e:
        print(f"Error retrieving {symbol}: {e}")
        return None


def run():
    # List of stock symbols to fetch data for
    stonks = [
        '^VIX', '^GSPC', '^VVIX',
        '^IRX','^FVX','^TNX','^TYX',
        'GC=F','HG=F','CL=F'
    ]
    tickers = stonks

    # Calculate the start and end date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14 * 365)  # 15 years back

    print(f"Fetching data for {tickers} from {start_date.date()} to {end_date.date()}...")
    data = fetch_yahoo_finance_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    data.index = pd.to_datetime(data.index).date

    if isinstance(data, pd.DataFrame):
        utils.pdf(data.tail(10))
        data.to_parquet('data/yahoo.parquet')
    else:
        print("Failed to fetch data")

def main():
    v = get_price("^VIX")
    print(v)

    s = get_price("^GSPC")
    print(s)

if __name__ == "__main__":
    main()
