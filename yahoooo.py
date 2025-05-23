import random
import time
import atexit
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import utils



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

        # # rotate UA
        # rotating_session.headers.update({
        #     "User-Agent": random.choice(USER_AGENTS)
        # })

        # pass our rotating session into yfinance
        t = yf.Ticker(symbol
                      # , session=rotating_session
                      )
        df = t.history(period="1d", timeout=10)

        df.index = pd.to_datetime(df.index).date
        return df[["Close"]].tail(1)

    except Exception as e:
        print(f"Error retrieving {symbol}: {e}")
        return None


def scrape_vix():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By

    service = Service("/usr/local/bin/chromedriver")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=service, options=options)

    driver.get("https://www.cboe.com/tradable_products/vix/")
    # time.sleep(2)  # you may need longer depending on your connection

    # 3. Inspect in your browser to find the right CSS selector.
    #    For example (this will almost certainly change!):
    price_elem = driver.find_element(
        By.CSS_SELECTOR,
        "#charts-tile > div > div > div:nth-child(1) > div.Box-cui__sc-6h7t1h-0.BorderBox-cui__sc-100l55d-0.eLdhlz.lewxc > div.Box-cui__sc-6h7t1h-0.Text-cui__sc-1owhlvg-0.khbfga.cSVxuZ")
    print("Live VIX price:", price_elem.text)

    driver.quit()
    return int(price_elem.text)


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
