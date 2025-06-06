import random
import time
import atexit
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import utils
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


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

# def get_price(symbol):
#     """
#     Fetch the latest close for a single ticker, rotating both IP and UA,
#     and sleeping a bit to avoid hammering the server.
#     """
#     try:
#         # small random delay
#         time.sleep(random.uniform(0.5, 1.5))
#
#         # # rotate UA
#         # rotating_session.headers.update({
#         #     "User-Agent": random.choice(USER_AGENTS)
#         # })
#
#         # pass our rotating session into yfinance
#         t = yf.Ticker(symbol
#                       # , session=rotating_session
#                       )
#         df = t.history(period="1d", timeout=10)
#
#         df.index = pd.to_datetime(df.index).date
#         return df[["Close"]].tail(1)
#
#     except Exception as e:
#         print(f"Error retrieving {symbol}: {e}")
#         return None
#
#
# def scrape_vix():
#     from selenium import webdriver
#     from selenium.webdriver.chrome.service import Service
#     from selenium.webdriver.common.by import By
#
#     service = Service("/usr/local/bin/chromedriver")
#     options = webdriver.ChromeOptions()
#     options.add_argument("--headless")
#     options.add_argument("--disable-gpu")
#     driver = webdriver.Chrome(service=service, options=options)
#
#     driver.get("https://www.cboe.com/tradable_products/vix/")
#     # time.sleep(2)  # you may need longer depending on your connection
#
#     # 3. Inspect in your browser to find the right CSS selector.
#     #    For example (this will almost certainly change!):
#     price_elem = driver.find_element(
#         By.CSS_SELECTOR,
#         "#charts-tile > div > div > div:nth-child(1) > div.Box-cui__sc-6h7t1h-0.BorderBox-cui__sc-100l55d-0.eLdhlz.lewxc > div.Box-cui__sc-6h7t1h-0.Text-cui__sc-1owhlvg-0.khbfga.cSVxuZ")
#     print("Live VIX price:", price_elem.text)
#
#     driver.quit()
#     return int(price_elem.text)


def scraper(url: str, css_selector: str) -> float:
    """
    Spins up a headless Chrome driver, opens `url`, finds the element
    matching `css_selector`, and returns its text parsed as float.
    Raises an exception if anything goes wrong.
    """
    # Path to your chromedriver binary; adjust if it’s elsewhere
    service = Service("/usr/local/bin/chromedriver")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get(url)
        # tiny sleep in case the page needs a moment to render
        time.sleep(1)

        elem = driver.find_element(By.CSS_SELECTOR, css_selector)
        text = elem.text.strip().replace(",", "")  # remove commas if present
        price = float(text)
    finally:
        driver.quit()

    return price


def get_price(symbol: str) -> pd.DataFrame | None:
    """
    Fetch the latest close for `symbol`. If symbol is "^VIX" or "^GSPC",
    first attempt to scrape a live price; if that fails, fall back to yfinance.
    Otherwise, just use yfinance.
    Returns a DataFrame with a single-row index = today’s date, column "Close".
    """
    # Define scraping rules for special tickers
    scraping_info = {
        "^VIX": {
            "url": "https://www.cboe.com/tradable_products/vix/",
            "css": (
                "#charts-tile > div > div > div:nth-child(1) "
                "> div.Box-cui__sc-6h7t1h-0.BorderBox-cui__sc-100l55d-0."
                "eLdhlz.lewxc > div.Box-cui__sc-6h7t1h-0.Text-cui__sc-1owhlvg-0."
                "khbfga.cSVxuZ"
            ),
        },
        "^GSPC": {
            "url": "https://www.cnbc.com/quotes/.SPX",
            "css": "#quote-page-strip > div.QuoteStrip-dataContainer "
                   "> div.QuoteStrip-lastTimeAndPriceContainer "
                   "> div.QuoteStrip-lastPriceStripContainer "
                   "> span.QuoteStrip-lastPrice",
            # on Yahoo Finance, this selector grabs the SPX live price
        },
    }

    # If symbol is in our scrape‐list, try scraping first
    if symbol in scraping_info:
        info = scraping_info[symbol]
        try:
            # small random delay to avoid hammering
            time.sleep(random.uniform(0.5, 1.5))
            live_price = scraper(info["url"], info["css"])
            # wrap into a 1x1 DataFrame with today's date as index
            today = datetime.now().date()
            df_live = pd.DataFrame({"Close": [live_price]}, index=[today])
            return df_live

        except Exception as e:
            print(f"Scraping fallback for {symbol} failed: {e}. Falling back to yfinance...")

    # Fallback (or non‐special symbol): use yfinance
    try:
        time.sleep(random.uniform(0.5, 1.5))
        t = yf.Ticker(symbol)
        df = t.history(period="1d", timeout=10)

        # ensure date index is just a date (no timestamps)
        df.index = pd.to_datetime(df.index).date
        return df[["Close"]].tail(1)

    except Exception as e:
        print(f"Error retrieving {symbol} via yfinance: {e}")
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
